[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)


[![Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/binbin83/synthetic_deraillement_and_repetition)



# PARAPHRASE GENERATOR AND DETECTOR

This repository was build to achieve the following goals:
    1. Generate dataset of story using small LLM such as GPT-2, falcon-1b etc. with different parameters setting (temperature etc.)
    2. Introduce a paraphrase at a specific place in the story. (at the i sentence)
        a) The first step is to create a paraphrase generator using SOTA models
        b) The second step is to create a streamlit app to evaluate the quality of the paraphrase
        c) It will enable us to have a human controlled and labeled dataset of semantic repetition
    3. Evaluate how speech disorder technics found and evaluate this semantic repetition.  

## Overview

This repository provides the computational framework for analyzing **semantic perseveration** and **incoherence** in psychiatric conditions through theory-driven generative language simulations. The codebase implements novel Natural Language Processing (NLP) metrics designed to capture the paradoxical interplay between semantic repetitiveness and derailment in Formal Thought Disorder (FTD).


## Associated Research Paper

**"Characterizing the Paradoxical Interplay of Semantic Perseveration and Incoherence in Psychiatry using Theory-Driven Generative Language Simulations"**

*Authors: Robin Quillivic¹'², Raymond J. Dolan³'⁴, Isaac Fradkin⁵*

### Key Research Contributions

- **Novel Density-Based Metrics**: Development of semantic density metrics that dissociate repetitiveness from derailment in formal thought disorder
- **Online Paraphrasing Method**: Innovative real-time paraphrase insertion during text generation to simulate downstream effects of repetition
- **Synthetic Dataset**: Controlled dataset with independent manipulation of semantic repetitiveness and derailment
- **Empirical Validation**: Superior performance in detecting repetitive speech patterns across psychiatric dimensions

### Research Problem

Traditional NLP metrics used in psychiatry, particularly cosine-based semantic distances, suffer from **interpretive ambiguity**:
- **Increased semantic distances** → Could indicate derailment/disorganization
- **Decreased semantic distances** → Could indicate either normal coherence OR pathological repetitiveness

This creates a fundamental challenge in clinical applications where both phenomena may co-occur, as the same metric result can have opposite clinical interpretations.


## 0. Install

1. Clone this repository to your local machine.
2. Install the required packages: `pip install -r requirements.txt`.
3. Place your dataset in the project directory and update the `config.yaml`.
4. Run the main script: `python train_model.py 10`. If you want to train 10 models  for 10 random seed
5. Analyze results in the `results` folder and using the log file: `deep_classification_text.log`.

---

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

--- 

### a) The first step is to create a paraphrase generator using SOTA models

In order to realize this task, we sectioned various SOTA models for paraphrase generation on huggingfaceHub.

#### Available Paraphrase Models

The paraphrase generator supports various models for generating paraphrases. Here are the currently supported models:

1. **Parrot Models:**
   - "Vamsi/T5_Paraphrase_Paws": A T5-based paraphrase model. https://huggingface.co/Vamsi/T5_Paraphrase_Paws 
   - "humarin/chatgpt_paraphraser_on_T5_base": Another T5-based model for paraphrasing.
   https://huggingface.co/humarin/chatgpt_paraphraser_on_T5_base 
   - "prithivida/parrot_paraphraser_on_T5": A T5-based model specialized in paraphrasing. https://huggingface.co/prithivida/parrot_paraphraser_on_T5
   - "stanford-oval/paraphraser-bart-large": A BART-based paraphrase model. https://huggingface.co/stanford-oval/paraphraser-bart-large 

2. **Quality Control Models:**
Paper: https://aclanthology.org/2022.acl-long.45/
code: https://github.com/ibm/quality-controlled-paraphrase-generation 

   - "ibm/qcpg-sentences": A quality control model for sentences.
   - "ibm/qcpg-captions": A quality control model for captions.
   - "ibm/qcpg-questions": A quality control model for generating questions.

You can select any of these models to generate paraphrases based on your specific needs.


#### Usage

To use the paraphrase generator, you can call the generate_paraphrase_from_model_name function and provide a model name and an input sentence. Here's an example:

```python

from src.paraphrase_generation import generate_paraphrase_from_model_name
model_name = "Vamsi/T5_Paraphrase_Paws"
input_sentence = "Input sentence to paraphrase."
paraphrase = generate_paraphrase_from_model_name(input_sentence, model_name)
print("Paraphrase:", paraphrase)
```

or use it to introduce a paraphrase at the i sentence in a story

```python
text = "long text. with multiple sentences"
index = 2 # the index of the sentence where you want to introduce a paraphrase
paraphrase_generator = ParaphraseGenerator(model_name="ibm/qcpg-sentences")
altered_text = introduce_paraphrase(text, index, paraphrase_generator)
```

or use it to transform a dataset of text. The dataframe should contain a column named "text" with the text to paraphrase, a columns index_paraphrase with the index of the sentence where you want to introduce a paraphrase.


```python
df = introduce_paraphrases_to_dataset(df =data, model_name="ibm/qcpg-sentences")
```



#### Running Tests

The project includes a sample test suite in the tests folder. To run the tests, use the following commands:

```sh
python -m unittest discover tests  # Using unittest
```

---

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://paraphrase-evaluation.streamlit.app/)

### b) The second step is to create a streamlit app to evaluate the quality of the paraphrase

Certainly, here's a template for a README section for the Streamlit app. You can include this in a larger README for your project:



#### Paraphrase Quality Evaluation App

The **Paraphrase Quality Evaluation App** is a Streamlit application designed to help evaluate the quality of paraphrases in a given dataset. The app allows users to select a dataset, view each sentence in the dataset one by one, emphasize the paraphrased sentence, and make modifications as needed. Users can validate the modified paraphrase, and the changes are saved to a new file.

#### Features

- **Dataset Selection**: Users can select a dataset and a specific altered dataset created with a model and parameters.
- **Sentence-by-Sentence Review**: The app displays each sentence in the dataset one by one, emphasizing the paraphrased sentence for review.
- **Paraphrase Modification**: Users can modify the paraphrased sentence as necessary.
- **Validation**: A validation button allows users to save their changes for each sentence.

#### Usage

1. **Select Dataset and Model**: Choose a dataset and a corresponding altered dataset using the sidebar.
Altered datasets are stored in the `data/altered` folder. if you want the App to work with your altered dataset, you need to place it in the `data/altered` folder. You can generate a new alterered dataset using the `main_generate_dataset.py` script.

2. **Review and Modify**: Review sentences one by one, with the paraphrased sentence emphasized. Modify the paraphrase as needed.

3. **Validation**: Click the "Validate" button to save the changes for the current sentence.

4. **Continue**: Review and validate each sentence. Once all examples are checked, a message will indicate that all examples have been reviewed.

#### Running the App
1. Run the Streamlit app on local.

```bash
streamlit run paraphrase_evaluation_app.py
```

2. A deployed version is available on Streamlit

link to the app: https://paraphrase-evaluation.streamlit.app/

/!\ the app may not be up to date

---


## 3. Online Paraphrasing. 

In the config file if config['generate_dataset']['online'] is set to True, then the generation process will add a paraphrase with a probability of config['online_param']['p_paraphrase'] at each step of the process.


Our **online paraphrasing insertion** method is the an innovation that enables realistic simulation of semantic repetition during text generation:

**Method**: Instead of post-hoc paraphrasing, we integrate paraphrases **during** the text generation process itself, mimicking how repetitive thinking constrains and biases subsequent discourse in real human speech.

**Parameters**:
- **P**: Probability of inserting a paraphrase at each generation step [0.1]
- **Q**: Probability of paraphrasing recent vs. distant sentences [0.1, 0.5, 0.9]  
- **α**: Pareto distribution parameter controlling paraphrase distance [0.5, 0.7]

**Paraphrase Generation**: Uses GPT-3.5 turbo via OpenAI API with the prompt:
> *"Paraphrase the following sentence, minimizing the words in common with the original sentence."*

This approach captures the **downstream effects** of repetitive thinking on narrative flow, unlike simple word repetition or post-hoc modifications.

In consequences, the next step of text generation will be influenced by this repetition. This will mimic more precisely the human speech disorder.
--- 

## 4. Evaluate how speech disorder technics found and evaluate this semantic repetition.

To evaluate speech disorder technics we reproduce well known metrics (cosine based) and implement new metrics.

To reproduce the result presented in the paper, the notebooks to be used are: 
- reproduce_results_data_analysis.ipynb
- reproduce_results_story_dataset.ipynb
- reproduce_results_story_dataset.ipynb






## Acknowledgments

Special thanks to the computational psychiatry and NLP communities for their foundational work in this area, and to all participants in the empirical studies that made this research possible.