# PARAPHRASE GENERATOR AND DETECTOR

This repository was build to achieve the following goals:
    1. Generate dataset of story using small LLM such as GPT-2, falcon-1b etc.
    2. Introduce a paraphrase at a specific place in the story. (at the i sentence)
        a) The first step is to create a paraphrase generator using SOTA models
        b) The second step is to create a streamlit app to evaluate the quality of the paraphrase
        c) It will enable us to have a human controlled and labeled dataset of semantic repetition
    3. Evaluate how speech disorder technics found and evaluate this semantic repetition.  


## 0. Install

1. Clone this repository to your local machine.
2. Install the required packages: `pip install -r requirements.txt`.
3. Place your dataset in the project directory and update the `config.yaml`.
4. Run the main script: `python train_model.py 10`. If you want to train 10 models  for 10 random seed
5. Analyze results in the `results` folder and using the log file: `deep_classification_text.log`.

## 1. Generate dataset of story using small LLM such as GPT-2, falcon-1b etc.

If you do not have a paraphrase dataset, you can generate one using the `main_generate_dataset.py` script.

To control the generation, you can use the following arguments in config.yaml:

- model_name: the LLM you want to use to generate the dataset (for now, gpt2, and falcon-1b are available)
- prompt_list: the list of prompts you want to use to generate the dataset
- n_samples: the number of samples you want to generate for each prompt
- target_length: the maximum length of the generated text
- min_temperature: the temperature min of the generation
- max_temperature: the temperature max of the generation

if clean is set to True, the generated dataset will be cleaned using the following rules:
- split into sentences
- remove last incomplete sentence
- remove special characters


## 2. Introduce a paraphrase at a specific place in the story. 


### a) The first step is to create a paraphrase generator using SOTA models


### b) The second step is to create a streamlit app to evaluate the quality of the paraphrase


## 3. Evaluate how speech disorder technics found and evaluate this semantic repetition.


