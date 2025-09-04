"""
Script of the streamlit application.

"""

import streamlit as st
import pandas as pd
import os
import ast

from src.utils import load_config

# Define the path to the dataset folder

altered_data_folder = "data/altered_corpus/"


def load_dataset(file_path):
    try:
        return pd.read_csv(file_path)
    except:
        return pd.read_csv(file_path, sep="\t")


def save_dataset(data, file_path):
    data.to_csv(file_path, index=False)


def insert_paraphrase(original_text, paraphrase):
    return original_text.replace(
        paraphrase, f"<span style='background-color: yellow;'>{paraphrase}</span>"
    )


def main():
    st.title("Paraphrase Quality Evaluation App")

    st.sidebar.header("Select Dataset and Model")

    # Select the dataset file
    dataset_folder = st.sidebar.selectbox(
        "Select Dataset", [x[0] for x in os.walk(altered_data_folder)][1:]
    )

    if dataset_folder:
        # Load dataset or previously validated dataset

        validated_file = os.path.join(dataset_folder, "validated_dataset.csv")
        if os.path.exists(validated_file):
            dataset = load_dataset(validated_file)
            dataset = dataset[dataset["verified"] == 0]  # load only unverified examples
        else:
            dataset = load_dataset(os.path.join(dataset_folder, "altered_dataset.csv"))
            dataset["evaluation"] = 0
            dataset["verified"] = 0
            dataset["verified_text"] = dataset["altered_text"]

        config_dataset = load_config(
            os.path.join(dataset_folder, "dataset_model_kwargs.yaml")
        )

        st.sidebar.write("Dataset Summary:")
        st.sidebar.write(f"Number of Examples: {len(dataset)}")
        st.sidebar.write(f"Text was generate using : {config_dataset['source_dataset'].split('/')[-1]}")
        st.sidebar.write(
            f"The paraphrases were generated using the model: {config_dataset['model_name']}"
        )

        if "ibm" in config_dataset["model_name"]:
            str_param = [
                f"{k} : {v}"
                for k, v in config_dataset["quality_control_kwargs"].items()
            ]
        else:
            str_param = [
                f"{k} : {v}" for k, v in config_dataset["parrot_kwargs"].items()
            ]

        st.sidebar.write(f"Parameter were: \n")
        for elt in str_param:
            st.sidebar.write(elt)

        st.header("Loaded Dataset")
        try: 
            dataset = dataset.rename(columns={"paraphase_type": "paraphrase_type"})
            dataset['index_paraphrase'] = dataset['index_paraphrase'].apply(lambda x: list(ast.literal_eval(x)))
        except:
            pass
        st.dataframe(dataset[["altered_text", "index_paraphrase", "evaluation","paraphrase_type"]].head(5))

        st.sidebar.write("**Work In Progres**:")
        example_index = st.sidebar.empty()

        if "current_example" not in st.session_state:
            st.session_state.current_example = 0

        current_example = st.session_state.current_example
        st.sidebar.write(f"current example: {current_example}")
        st.sidebar.write(f"{current_example/len(dataset)*100:.2f}% done")

        # Split the text into sentences
        sentences = dataset.iloc[current_example]["altered_text"].split(".")
        para_index = dataset.iloc[current_example]["index_paraphrase"]
        print(para_index)
        st.header("Example to validate")

        # Create a text input for paraphrase validation
        for i, sentence in enumerate(sentences):
            if len(sentence) > 2:
                if i == para_index[0] or i == para_index[1]:
                    st.markdown(
                        f"{i+1}. **{sentence.strip()}**",
                    )
                else:
                    st.write(f"{i+1}. {sentence}")

        st.subheader(
            f"The sentence n°{int(para_index[0] + 1)} was paraphrased into n°{int(para_index[1] + 1)}"
        )
        st.write("Please evaluate the quality of the paraphrase.")

        # Create a text input for modifying the paraphrase
        col1,col2 = st.columns(2)

        with col1:
            good = st.button("Good enough")
        with col2:
            not_good = st.button("Not good enough")

        if good:
      
            dataset.at[current_example, "evaluation"] = 1
            dataset.at[current_example, "verified"] = 1
            save_dataset(dataset, os.path.join(validated_file))

            current_example += 1
            st.session_state.current_example = current_example
        
        elif not_good:
            dataset.at[current_example, "evaluation"] = 0
            dataset.at[current_example, "verified"] = 1
            save_dataset(dataset, os.path.join(validated_file))

            current_example += 1
            st.session_state.current_example = current_example



        if current_example == len(dataset):
            st.write("You have checked all examples.")

        st.sidebar.write("Download your work at any time:")
        download_button = st.sidebar.download_button(
            label="Download Verified Dataset",
            data=dataset.to_csv(index=False),
            key="download_verified_dataset",
            file_name="verified_dataset.csv",
        )
        if download_button:
            st.write("File is ready for download.")
            
if __name__ == "__main__":
    main()
