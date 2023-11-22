import streamlit as st
import pandas as pd
import os

from src.utils import load_config

from src.paraphrase_generation import generate_paraphrase_from_model_name, ParaphraseGenerator


# Function to paraphrase text based on user input parameters
def paraphrase_text(text, lexical, syntactic, semantic):
    quality_control_kwargs = {
        "lexical": lexical,
        "syntactic": syntactic,
        "semantic": semantic,
    }
    model_name = "ibm/qcpg-sentences"
    
    # Generate paraphrase
    generator = ParaphraseGenerator(model_name=model_name, quality_control_kwargs=quality_control_kwargs)
    paraphrase = generator.paraphrase(text)
    
    return paraphrase

# Streamlit app
def main():
    st.title("Text Paraphraser")

    st.subheader("Paraphrase text using IBM's Quality Controled Paraphrase Generation (QCPG) model.")

    st.write(f"<p style='font-size:20px'><strong> Parameters:</strong> </p>", unsafe_allow_html=True)
            
    # User input for parameters
    lexical_param = st.slider("Lexical Parameter", 0.0, 1.0, 0.5, 0.01)
    syntactic_param = st.slider("Syntactic Parameter", 0.0, 1.0, 0.5, 0.01)
    semantic_param = st.slider("Semantic Parameter", 0.0, 1.0, 0.5, 0.01)

    # Text input
    text_input = st.text_area("## Enter Text to Paraphrase")

    # Paraphrase button
    if st.button("Paraphrase"):
        if text_input:
            # Call the paraphrase function
            paraphrased_text = paraphrase_text(text_input, lexical_param, syntactic_param, semantic_param)
            
            # Display original and paraphrased text
            st.write(f"<p style='font-size:20px'><strong>Original Text:</strong> {text_input}</p>", unsafe_allow_html=True)
            st.write(f"<p style='font-size:20px'><strong>Paraphrased Text:</strong> {paraphrased_text}</p>", unsafe_allow_html=True)
        else:
            st.warning("Please enter text to paraphrase.")

if __name__ == "__main__":
    main()


