""" 
Generate paraphrase during the text generation process
"""

import random
import torch
import numpy as np
import logging
import os
from tqdm import tqdm
import pandas as pd
import yaml

from src.utils import load_config, split_into_sentences, replace_multiple_periods

from src.paraphrase_generation import generate_paraphrase_from_model_name, ParaphraseGenerator
from src.text_generation import load_model, generate_random_temperature, clean_generated_text



def generate_from_prompt_with_paraphrases(
    prompt,
    model,
    tokenizer,
    paraphrase_generator,
    nb_paraphrase_max = 2,
    do_sample=True,
    prospective_span=5,
    retrospective_span=200,
    target_length=200,
    top_p=0.95,   # Lower values make the output more focused#
    top_k=50,  #Higher values make the output more random
    temperature=0.9,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    punc_tokens=np.arange(0, 250, 1),
    excluded_tokens_list=["\n", "http", "www.", "UESPWiki", "\n\n", "\n\n\n", "50256"],
    device=None,
    p_paraphrase=0.1,

):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    excluded_tokens = tokenizer(excluded_tokens_list).input_ids
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)
    input_ids_temp = input_ids[
        :, max(0, input_ids.shape[1] - retrospective_span) : input_ids.shape[1]
    ]
    #initilization
    output_all = input_ids
    cont = True
    previous = True
    sentence_count = 0  # Keep track of the number of sentences generated
    paraphrased_index = [0]
    nb_paraphrase = 0

    while cont:
        temp_output = model.generate(
            input_ids_temp,
            do_sample=do_sample,
            max_length=prospective_span + input_ids_temp.shape[1],
            min_length=int(prospective_span + input_ids_temp.shape[1]),
            top_k = top_k,
            top_p=top_p,
            bad_words_ids=excluded_tokens,
            temperature=temperature,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        output_only = temp_output[:, input_ids_temp.shape[1] : temp_output.shape[1]]
       
        # Count the number of sentences
        temp_text = replace_multiple_periods(tokenizer.decode(output_all[0], skip_special_tokens=False))
        #print(temp_text)
        sentence_count = len(split_into_sentences(temp_text)) # How many sentences are in the output for now

        new_sentences = split_into_sentences(tokenizer.decode(output_all[0]))
        
        

        # Introduce a paraphrase acording to the probability p_paraphrase and if not the first sentence and if not too many paraphrases
        # At each genration process
        if random.random() < p_paraphrase  and sentence_count > 1 and nb_paraphrase < nb_paraphrase_max and paraphrased_index[-1] + 2 < sentence_count : # if we did not do a paraphrase in the previous step
           
            print("doing paraphrase...")
            # extract the last sentence and it's ID
            if new_sentences[-1][-1] == ".": # If the last sentence ends with a "." then its complete
                last_sentence = new_sentences[-1]
                new_sentences = new_sentences + [paraphrase_generator.paraphrase(last_sentence)] # we add a paraphrase after the last sentence
                paraphrased_index.append(sentence_count)
                print("last sentence ended with a .")
            else: # If the last sentence does not end with a "." then it is incomplete
                if len(new_sentences) > 1: # If there is more than one sentence in the output
                    last_sentence = new_sentences[-2] # the last complete sentence is the second to last sentence
                    #print(tokenizer.decode(output_only[0]))
                    #print(new_sentences)
                    paraphrase = paraphrase_generator.paraphrase(last_sentence)
                    new_sentences = new_sentences[:-1] + [paraphrase] + [new_sentences[-1]] # we add a paraphrase after the last sentence
                    paraphrased_index.append(sentence_count-1)
                    print("inserting paraphrase between two sentences")
                    print(paraphrase, last_sentence)
                else :
                    last_sentence = new_sentences[0] # there is only one sentence in the output, so the last complete sentence is the first sentence
                    new_sentences = new_sentences + [paraphrase_generator.paraphrase(last_sentence)] # we add a paraphrase after the last sentence even if incomplete
                    paraphrased_index.append(sentence_count)
                    print("inserting paraphrase after the only sentence")
                        
            altered_text = " ".join(new_sentences)
            altered_text = replace_multiple_periods(altered_text)
            altered_ids = tokenizer.encode(altered_text, return_tensors="pt").to(device)
            output_all = altered_ids
              # The number of sentences before paraphrase
            nb_paraphrase +=1 # we only do one paraphrase per output
                
            
        else: #just add the output to the output_all
            output_all = torch.cat((output_all, output_only), 1)
                
        input_ids_temp = output_all[
            :, max(0, output_all.shape[1] - retrospective_span) : output_all.shape[1]
        ]

        if output_all.shape[1] >= target_length:
            cont = False

        input_length = retrospective_span
        nPunct = sum(
            sum(output_all[0, -input_length:] == i for i in punc_tokens).bool()
        )
        input_length = input_length + nPunct.item()
        input_length = min(input_length, output_all.shape[1])
        input_ids_temp = output_all[:, -input_length:]

      
    results = tokenizer.decode(output_all[0], skip_special_tokens=False)
    return results, paraphrased_index[1:]



def generate_online_paraphrase_dataset_from_config(
    config, logger=logging.getLogger(__name__), save=False, clean=False
):
    model_name = config['generate_dataset']["model_name"]
    saving_folder = os.path.join(config['paraphrase_dataset']["saving_folder"],f"online_{model_name}")
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    prompt_list = config['generate_dataset']["prompt_list"]
    saving_path = os.path.join(saving_folder, f"altered_dataset.csv")
    n_sim = len(prompt_list) * config["generate_dataset"]["n_simulations"]

    quality_control_kwargs  = config['paraphrase_dataset']["quality_control_kwargs"]
    para_model_name = config['paraphrase_dataset']["model_name"]
    generator = ParaphraseGenerator(model_name=para_model_name, quality_control_kwargs=quality_control_kwargs)

    logger.info(
        f"Generating {n_sim} simulations for model {model_name} and saving it to {saving_path}"
    )
    logger.info(f"Prompt list : {prompt_list}")
    logger.info(
        f"Temperature range : {config['generate_dataset']['min_temperature']} - {config['generate_dataset']['max_temperature']}"
    )
    logger.info("Paraphrase generator : %s", para_model_name)
    logger.info("Quality control kwargs : %s", quality_control_kwargs)

    # Init dataset
    dataset = pd.DataFrame(
        columns=["prompt", "generated_text", "model_name", "temperature", "num_beams","index_paraphrase"]
    )

    temperature_list = generate_random_temperature(
        n_sim=n_sim, min=config['generate_dataset']["min_temperature"], max=config['generate_dataset']["max_temperature"]
    )
    num_beams_list = [random.randint(3, 5) for _ in range(n_sim)]

    # Load model
    model, tokenizer = load_model(model_name=model_name)

    # Generate dataset
    k = 0
    for prompt in prompt_list:
        logger.info(f"Generating simulations for prompt : {prompt} ...")
        for i in tqdm(range(config['generate_dataset']["n_simulations"])):
            temperature = temperature_list.pop()
            num_beams = num_beams_list.pop()
            generated_text, paraphrase_index = generate_from_prompt_with_paraphrases(
                prompt = prompt,
                model = model,
                tokenizer = tokenizer,
                paraphrase_generator = generator,
                nb_paraphrase_max = config['online_param']["nb_paraphrase_max"],
                temperature=temperature,
                num_beams=num_beams,
                target_length=config['generate_dataset']["target_length"],
                retrospective_span=config['generate_dataset']["retrospective_span"],
                top_p=config['generate_dataset']["top_p"],
                top_k=config['generate_dataset']["top_k"],
                p_paraphrase=config['online_param']["p_paraphrase"],
            )
            try :
                index_paraphrase = paraphrase_index[0]
            except:
                index_paraphrase = 0
            dataset.loc[k] = pd.Series(
                {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "model_name": model_name,
                    "temperature": temperature,
                    "num_beams": num_beams,
                    "index_paraphrase": index_paraphrase-1,
                }
            )
            k += 1

    if clean:
        logger.info("Cleaning text dataset")
        dataset["altered_text"] = dataset["generated_text"].apply(
            lambda x: clean_generated_text(x)
        )

    if save:
        logger.info(f"Saving dataset to {saving_path}")
        dataset.to_csv(saving_path, index=False, sep="\t")

        param_path = os.path.join(saving_folder, "dataset_model_kwargs.yaml")
        with open(param_path, "w") as file:
            documents = yaml.dump(config['paraphrase_dataset'], file)

    return dataset