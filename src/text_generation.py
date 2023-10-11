# standard imports
import random
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm

# transformers imports
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np

from src.utils import clean_generated_text


def clean_gpu():
    """
    Clean the GPU memory
    """
    try:
        del model
        del tokenizer
    except:
        pass
    torch.cuda.empty_cache()


def load_model(model_name="gpt2", logger=logging.getLogger(__name__)):
    """
    Load the model and the tokenizer based on the model name

    Args:
        model_name (str, optional): _description_. Defaults to "gpt2".
        logger (_type_, optional): _description_. Defaults to logging.getLogger(__name__).
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == "falcon-1b":
        tokenizer = AutoTokenizer.from_pretrained("euclaise/falcon_1b_stage2")
        model = AutoModelForCausalLM.from_pretrained(
            "euclaise/falcon_1b_stage2", pad_token_id=tokenizer.eos_token_id
        ).to(device)

    elif model_name == "gpt2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2-large", add_prefix_space=False)
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2-large", pad_token_id=tokenizer.eos_token_id
        ).to(device)

    else:
        logger.error("Unknown model name")
        raise ValueError("Unknown model name")

    logger.info(f"Model {model_name} loaded on {device}")

    return model, tokenizer


def generate_random_temperature(n_sim=200, min=0.9, max=5):
    """
    Generate random temperature for the text generation

    Args:
        n_sim (int, optional): _description_. Defaults to 200.
        min (float, optional): _description_. Defaults to 0.9.
        max (int, optional): _description_. Defaults to 5.

    Returns:
        _type_: _description_
    """
    vector_length = n_sim  # You can change this to your desired length

    # Initialize an empty list to store the random values
    random_vector = []

    # Generate random values for the vector
    for _ in range(vector_length):
        if random.random() < 0.5:
            # 50% chance for values between 0.95 and 1.05
            value = random.uniform(0.95, 1.05)
        else:
            # 50% chance for values between 0.9 and 5 (excluding 0.95 to 1.05 range)
            value = random.uniform(min, max)
        random_vector.append(value)

    print("mean:", np.mean(random_vector))
    print("std:", np.std(random_vector))
    return random_vector


def generate_from_prompt(
    prompt,
    model,
    tokenizer,
    do_sample=True,
    prospective_span=5,
    retrospective_span=200,
    target_length=200,
    top_p=0.95,
    temperature=0.9,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    punc_tokens=np.arange(0, 250, 1),
    excluded_tokens_list=["\n", "http", "www.", "UESPWiki", "\n\n", "\n\n\n", "50256"],
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    excluded_tokens = tokenizer(excluded_tokens_list).input_ids
    input_ids = tokenizer.encode(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(device)
    # input_ids=input_ids.input_ids
    input_ids_temp = input_ids[
        :, max(0, input_ids.shape[1] - retrospective_span) : input_ids.shape[1]
    ]
    output_all = input_ids
    cont = True

    while cont:
        temp_output = model.generate(
            input_ids_temp,
            do_sample=do_sample,
            max_length=prospective_span + input_ids_temp.shape[1],
            min_length=int(prospective_span + input_ids_temp.shape[1]),
            top_p=top_p,
            bad_words_ids=excluded_tokens,
            # top_k=top_k,
            temperature=temperature,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        output_only = temp_output[:, input_ids_temp.shape[1] : temp_output.shape[1]]
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
    return results


def generate_dataset_from_config(
    config, logger=logging.getLogger(__name__), save=False, clean=False
):
    saving_folder = config["saving_folder"]
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    prompt_list = config["prompt_list"]
    model_name = config["model_name"]
    saving_path = os.path.join(saving_folder, f"{model_name}_{config['dataset_name']}")
    n_sim = len(prompt_list) * config["n_simulations"]

    logger.info(
        f"Generating {n_sim} simulations for model {model_name} and saving it to {saving_path}"
    )
    logger.info(f"Prompt list : {prompt_list}")
    logger.info(
        f"Temperature range : {config['min_temperature']} - {config['max_temperature']}"
    )

    # Init dataset
    dataset = pd.DataFrame(
        columns=["prompt", "generated_text", "model_name", "temperature", "num_beams"]
    )

    temperature_list = generate_random_temperature(
        n_sim=n_sim, min=config["min_temperature"], max=config["max_temperature"]
    )
    num_beams_list = [random.randint(3, 5) for _ in range(n_sim)]

    # Load model
    model, tokenizer = load_model(model_name=model_name)

    # Generate dataset
    k = 0
    for prompt in prompt_list:
        logger.info(f"Generating simulations for prompt : {prompt} ...")
        for i in tqdm(range(config["n_simulations"])):
            temperature = temperature_list.pop()
            num_beams = num_beams_list.pop()
            generated_text = generate_from_prompt(
                prompt,
                model,
                tokenizer,
                temperature=temperature,
                num_beams=num_beams,
                target_length=config["target_length"],
                retrospective_span=config["retrospective_span"],
                top_p=config["top_p"],
            )

            dataset.loc[k] = pd.Series(
                {
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "model_name": model_name,
                    "temperature": temperature,
                    "num_beams": num_beams,
                }
            )
            k += 1

    if clean:
        logger.info("Cleaning text dataset")
        dataset["text"] = dataset["generated_text"].apply(
            lambda x: clean_generated_text(x)
        )

    if save:
        logger.info(f"Saving dataset to {saving_path}")
        dataset.to_csv(saving_path, index=False, sep="\t")

    return dataset
