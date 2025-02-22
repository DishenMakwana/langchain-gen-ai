# --- Using LLM locally, here are running llama3.2 model locally ---

# ref: https://python.langchain.com/docs/integrations/llms/ollama/

# Step 1: Download ollama application. (https://github.com/ollama/ollama)
# Step 2: Download modal locally (https://github.com/ollama/ollama?tab=readme-ov-file#model-library)
# Step 3: Get LANGCHAIN_API_KEY from langchain website (https://smith.langchain.com/)
# Step 4: Then in terminal run > `streamlit run ollamaLocalApp.py`

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

## streamlit framework
st.title('Langchain Demo With LLAMA2 API')
input_text=st.text_input("Search the topic u want")

# ollama LLAma2 LLm 
llm=Ollama(model="llama3.2")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))