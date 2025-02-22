# End To End Langchain Project Using Groq Inference Engine

import streamlit as st  # For building the interactive web application
import os  # For accessing environment variables
from langchain_groq import ChatGroq  # Groq inference engine integration for LLMs
from langchain_community.document_loaders import WebBaseLoader  # To load documents from web URLs
from langchain.embeddings import OllamaEmbeddings  # To create vector embeddings for documents
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split documents into chunks
from langchain.chains.combine_documents import create_stuff_documents_chain  # To combine retrieved documents for the LLM
from langchain_core.prompts import ChatPromptTemplate  # To create structured prompts for LLMs
from langchain.chains import create_retrieval_chain  # To create a retriever and document chain
from langchain_community.vectorstores import FAISS  # For storing and querying vectorized document embeddings
import time  # For measuring response times

from dotenv import load_dotenv  # To load environment variables from a .env file
load_dotenv()

## Load the Groq API key from environment variables
groq_api_key = os.environ['GROQ_API_KEY']

# Initialize vector embedding process only if not already in session state
if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model="llama3.2:latest")  # Initialize embeddings
    st.session_state.loader = WebBaseLoader("https://docs.smith.langchain.com/")  # Load documents from URL
    st.session_state.docs = st.session_state.loader.load()  # Load the documents

    # Split documents into smaller chunks for processing
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    
    # Create vector embeddings and store them in a FAISS vector database
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit app title
st.title("ChatGroq Demo")

# Initialize ChatGroq LLM with the specified model
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama3-8b-8192"
)

# Define a structured prompt for the LLM
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

# Create a document chain that processes retrieved documents
document_chain = create_stuff_documents_chain(llm, prompt)

# Convert the FAISS vector store into a retriever
retriever = st.session_state.vectors.as_retriever()

# Create a retrieval chain that connects the retriever and document chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Capture user input via Streamlit
prompt = st.text_input("Input your prompt here")

if prompt:
    start = time.process_time()  # Start measuring response time
    response = retrieval_chain.invoke({"input": prompt})  # Get the response from the chain
    print("Response time:", time.process_time() - start)  # Log the response time
    st.write(response['answer'])  # Display the answer in the app

    # Show relevant document chunks in an expandable section
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):  # Iterate through the retrieved chunks
            st.write(doc.page_content)  # Display the content of each chunk
            st.write("--------------------------------")