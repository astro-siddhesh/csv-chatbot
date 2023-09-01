import streamlit as st
import pandas as pd
from pandasai.llm.openai import OpenAI
from pandasai import PandasAI
import os
from matplotlib.backends.backend_agg import RendererAgg
import matplotlib.pyplot as plt
import matplotlib

# Initialize the Matplotlib lock
_lock = RendererAgg.lock

# Load   CSV data
df = pd.read_csv("random_data.csv")

# Initialize PandasAI and OpenAI
llm = OpenAI()
pandas_ai = PandasAI(llm, conversational=True)

# Streamlit App Configuration
st.set_page_config(
    page_title="CSV Analysis",
    page_icon="📊",
)

# Custom CSS Styles
st.markdown(
    """
    <style>
    .title {
        font-size: 36px !important;
        color: #3366ff !important;
    }
    
    .sidebar .sidebar-content {
        background-color: #f0f0f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
st.title("CSV Data Analysis")

# User input for prompts
user_prompt = st.text_input("Enter your prompt:")

if user_prompt:
    # Generate a response based on the user's prompt
    response = pandas_ai.run(df, prompt=user_prompt)
    
    # Display the response to the user
    st.subheader("Response:")
    st.write(response)
    