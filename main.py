from openai import OpenAI
import deepeval
from decouple import config
import streamlit as st
import pandas as pd
import numpy as np

client = OpenAI(
    base_url = "https://api.groq.com/openai/v1",
    api_key = config('GROQ_API_KEY')
)

def fetch_response(model_name, model_id):
    with st.spinner(f"Fetching response from {model_name}..."):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}]
            )
            # Assuming 'choices[0].message.content' contains the response text
            return response.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Error with {model_name}: {e}")
            return None

# User prompt input
prompt = st.text_area("Enter your prompt:", "")

# Collect responses
gemini_response = fetch_response("Gemini", "gemma2-9b-it")
mixtral_response = fetch_response("Mixtral", "mixtral-8x7b-32768")
llama_response = fetch_response("Llama", "llama-3.1-8b-instant")

# Submit button
if st.button("Generate Responses"):
    if prompt.strip():
        st.subheader("Responses")
        
    if prompt:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Gemini")
            if gemini_response:
                st.markdown(gemini_response)
            else:
                st.markdown("*No response*")

        with col2:
            st.markdown("### Mixtral")
            if mixtral_response:
                st.markdown(mixtral_response)
            else:
                st.markdown("*No response*")

        with col3:
            st.markdown("### Llama")
            if llama_response:
                st.markdown(llama_response)
            else:
                st.markdown("*No response*")

        #Evaluation section

    else:
        st.error("Please enter a valid prompt.")