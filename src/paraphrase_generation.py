# standard imports
import random
import pandas as pd
import numpy as np
import os
import logging
from tqdm import tqdm


# specific imports
from parrot import Parrot

import logging
from transformers import pipeline


class ParaphraseGenerator:
    def __init__(
        self, model_name, logger=None, parrot_kwargs=None, quality_control_kwargs=None
    ):
        self.model_name = model_name
        self.generator = None
        self.parrot_kwargs = parrot_kwargs or {
            "diversity_ranker": "levenshtein",
            "do_diverse": 9,
            "max_return_phrases": 5,
            "adequacy_threshold": 0.1,
            "fluency_threshold": 0.1,
        }
        self.quality_control_kwargs = quality_control_kwargs or {
            "lexical": 0.3,
            "syntactic": 0.5,
            "semantic": 0.8,
        }
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        self.load_generator()

    def load_generator(self):
        if self.model_name in [
            "Vamsi/T5_Paraphrase_Paws",
            "humarin/chatgpt_paraphraser_on_T5_base",
            "prithivida/parrot_paraphraser_on_T5",
            "stanford-oval/paraphraser-bart-large",
        ]:
            self.generator = ParrotGenerator(
                model_name=self.model_name, kwargs=self.parrot_kwargs
            )
            self.logger.info("Loaded ParrotGenerator")
        elif self.model_name in [
            "ibm/qcpg-sentences",
            "ibm/qcpg-captions",
            "ibm/qcpg-questions",
        ]:
            self.generator = QualityControlGenerator(
                model_name=self.model_name, kwargs=self.quality_control_kwargs
            )
            self.logger.info("Loaded QualityControlGenerator")
        else:
            self.logger.error(
                "Invalid model_name. Supported values: Parrot or Quality Control models."
            )
            raise ValueError(
                "Invalid model_name. Supported values: Parrot or Quality Control models."
            )

    def paraphrase(self, input_phrase):
        if self.generator is not None:
            self.logger.info(
                "Generating paraphrase using {} for input: {}".format(
                    self.model_name, input_phrase
                )
            )
            return self.generator.paraphrase(input_phrase)
        else:
            self.logger.error(
                "Generator has not been loaded. Call load_generator first."
            )
            raise ValueError(
                "Generator has not been loaded. Call load_generator first."
            )


class ParrotGenerator:
    def __init__(self, model_name, kwargs=None):
        self.parrot = Parrot(model_tag=model_name, use_gpu=True)
        self.kwargs = kwargs

    def paraphrase(self, input_phrase):
        return self.parrot.augment(input_phrase=input_phrase, **self.kwargs)[0][0]


class QualityControlGenerator:
    def __init__(self, model_name, kwargs=None):
        self.quality_control = QualityControlPipeline(model_name=model_name, **kwargs)

    def paraphrase(self, input_phrase):
        return self.quality_control(input_phrase)


class QualityControlPipeline:
    def __init__(self, model_name, lexical, syntactic, semantic):
        self.pipe = pipeline("text2text-generation", model=model_name)
        self.ranges = {
            "captions": {"lex": [0, 90], "syn": [0, 80], "sem": [0, 95]},
            "sentences": {"lex": [0, 100], "syn": [0, 80], "sem": [0, 95]},
            "questions": {"lex": [0, 90], "syn": [0, 75], "sem": [0, 95]},
        }[model_name.split("-")[-1].lower()]

        self.lexical = lexical
        self.syntactic = syntactic
        self.semantic = semantic

    def __call__(self, text):
        assert all(
            [0 <= val <= 1 for val in [self.lexical, self.syntactic, self.semantic]]
        ), f" control values must be between 0 and 1, got {self.lexical}, {self.syntactic}, {self.semantic}"
        names = ["semantic_sim", "lexical_div", "syntactic_div"]
        control = [
            int(5 * round(val * 100 / 5))
            for val in [self.semantic, self.lexical, self.syntactic]
        ]
        control = {
            name: max(min(val, self.ranges[name[:3]][1]), self.ranges[name[:3]][0])
            for name, val in zip(names, control)
        }
        control = [f"COND_{name.upper()}_{control[name]}" for name in names]
        assert all(
            cond in self.pipe.tokenizer.additional_special_tokens for cond in control
        )
        text = (
            " ".join(control) + text
            if isinstance(text, str)
            else [" ".join(control) for t in text]
        )
        max_length = len(text.split(" ")) + 10
        return self.pipe(text, max_new_tokens=max_length)[0]["generated_text"]


def generate_paraphrase_from_model_name(sentence, model_name):
    generator = ParaphraseGenerator(model_name=model_name)
    return generator.paraphrase(sentence)


def introduce_paraphrase(text, index, paraphrase_generator):
    """
    Introduce a semantic repetition in the text at the specified index.
    Args:
        text (_type_): _description_
        index (_type_): _description_
        paraphrase_generator (_type_): _description_

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    # Split the text into sentences
    sentences = text.split(".")  # Assuming sentences are separated by period and space

    # Check if I is within the valid range
    if index < 0 or index >= len(sentences):
        raise ValueError(
            "Invalid value of I. It should be within the range [0, {}].".format(
                len(sentences) - 1
            )
        )

    # Generate a paraphrase for the sentence at index I
    paraphrase = paraphrase_generator.paraphrase(sentences[index])

    # Keep the original sentence
    original_sentence = sentences[index]

    # Insert the paraphrase alongside the original sentence
    sentences[index] = f"{original_sentence}. {paraphrase}"

    # Reconstruct the modified text
    modified_text = ".".join(sentences)

    return modified_text


def introduce_paraphrases_to_dataset(df, model_name, save_path=None, logger=None):
    if logger is None:
        logger = logging.getLogger(__name__)

    logger.info("Introducing paraphrases to the dataset using model: %s", model_name)

    # Create a ParaphraseGenerator with the specified model
    generator = ParaphraseGenerator(model_name=model_name, logger=logger)

    # Create a new column named "altered_text" to store the modified text
    df["altered_text"] = df["text"]

    # Iterate through the DataFrame rows
    for index, row in df.iterrows():
        text = row["text"]
        index_paraphrase = row["index_paraphrase"]

        try:
            # Introduce the paraphrase into the text
            modified_text = introduce_paraphrase(
                index=index_paraphrase, paraphrase_generator=generator, text=text
            )
            df.at[index, "altered_text"] = modified_text
            logger.info("Paraphrase introduced for row %d: %s", index, modified_text)

        except ValueError as e:
            logger.error("Error in row %d: %s", index, str(e))

    if save_path:
        df.to_csv(save_path, index=False)
        logger.info("Altered dataset saved to: %s", save_path)

    return df
