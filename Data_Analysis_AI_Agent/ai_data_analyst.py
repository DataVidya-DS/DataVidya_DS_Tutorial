import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools
import re
from streamlit_lottie import st_lottie

# Utility function to handle file preprocessing and temporary storage
def process_uploaded_file(uploaded_file):
    try:
        # Read the file based on its extension
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif uploaded_file.name.endswith('.xlsx'):
            data = pd.read_excel(uploaded_file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file type. Please upload a CSV or Excel file.")
            return None, None, None

        # Clean string columns and ensure proper quoting
        for column in data.select_dtypes(include=['object']):
            data[column] = data[column].astype(str).replace({r'"': '""'}, regex=True)

        # Handle date and numeric parsing
        for column in data.columns:
            if 'date' in column.lower():
                data[column] = pd.to_datetime(data[column], errors='coerce')
            elif data[column].dtype == 'object':
                try:
                    data[column] = pd.to_numeric(data[column])
                except (ValueError, TypeError):
                    pass  # Retain original values if parsing fails

        # Save the processed data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            file_path = temp_file.name
            data.to_csv(file_path, index=False, quoting=csv.QUOTE_ALL)

        return file_path, data.columns.tolist(), data
    except Exception as err:
        st.error(f"File processing failed: {err}")
        return None, None, None

# Function to load Lottie animations from a JSON file
def load_lottie_animation(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

# Streamlit app: Interactive Data Exploration
lottie_animation = load_lottie_animation("Robot.json")

# App header with columns for alignment
col_left, col_right = st.columns([3, 1], gap="medium")
with col_left:
    st.markdown(
        """
        <h1 style="display: inline-block; vertical-align: middle;">🤖 <b>DataSense</b> - Smarter Insights</h1>
        <p style="font-size: 16px;">
        Welcome to <b>DataSense</b>, your AI-powered assistant for data exploration!  
        Upload your dataset, ask natural language questions, and let AI handle the rest. 🚀  
        </p>
        """,
        unsafe_allow_html=True,
    )
with col_right:
    st_lottie(
        lottie_animation,
        speed=1,
        reverse=False,
        loop=True,
        quality="high",
        height=None,
        width=None,
        key=None,
    )

# Sidebar for API key input
with st.sidebar:
    st.header("🔑 API Key Configuration")
    api_key = st.text_input("Enter your OpenAI API key:", type="password")
    if api_key:
        st.session_state.openai_key = api_key
        st.success("API key saved successfully!")
    else:
        st.warning("An API key is required to proceed.")

# File upload section
st.header("📂 Upload Dataset")
file_uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if file_uploaded and "openai_key" in st.session_state:
    # Process the uploaded file
    processed_path, column_names, dataframe = process_uploaded_file(file_uploaded)

    if processed_path and column_names and dataframe is not None:
        # Display the processed data
        st.write("Dataset Preview:")
        st.dataframe(dataframe)
        st.write("Columns Detected:", column_names)

        # Configure the semantic model for query generation
        semantic_config = {
            "tables": [
                {
                    "name": "uploaded_dataset",
                    "description": "User-uploaded dataset.",
                    "path": processed_path,
                }
            ]
        }

        # Initialize the DuckDbAgent with required configurations
        db_agent = DuckDbAgent(
            model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
            semantic_model=json.dumps(semantic_config),
            tools=[PandasTools()],
            markdown=True,
            add_history_to_messages=False,
            followups=False,
            read_tool_call_history=False,
            system_prompt="You are an expert data analyst. Generate SQL queries for the given user query. Return SQL queries enclosed in ```sql ``` along with the final answer.",
        )

        # Session state for storing generated SQL
        if "sql_code" not in st.session_state:
            st.session_state.sql_code = None

        # Input box for user queries
        query_input = st.text_area("Enter your query about the data:")

        if st.button("Run Query"):
            if query_input.strip() == "":
                st.warning("Please enter a valid query.")
            else:
                try:
                    with st.spinner("Analyzing your query..."):
                        agent_response = db_agent.run(query_input)

                        # Extract content from the response
                        response_text = agent_response.content if hasattr(agent_response, 'content') else str(agent_response)
                        
                        # Display response in Streamlit
                        st.markdown(response_text)

                except Exception as error:
                    st.error(f"Failed to process the query: {error}")
