import streamlit as st  # For creating a web-based interactive interface
import os  # For managing environment variables
from langchain_groq import ChatGroq  # Groq inference engine integration
from langchain_openai import OpenAIEmbeddings  # OpenAI embeddings for vectorization
from langchain.text_splitter import RecursiveCharacterTextSplitter  # To split large text into smaller chunks
from langchain.chains.combine_documents import create_stuff_documents_chain  # Combines retrieved documents for processing
from langchain_core.prompts import ChatPromptTemplate  # For creating structured prompts
from langchain.chains import create_retrieval_chain  # Combines retrievers and chains for query handling
from langchain_community.vectorstores import FAISS  # FAISS vector store for similarity search
from langchain_community.document_loaders import PyPDFDirectoryLoader  # To load PDFs from a directory
from dotenv import load_dotenv  # For loading environment variables from .env file
from langchain_ollama import OllamaEmbeddings

# Load environment variables from .env file
load_dotenv()

# Set OpenAI and Groq API keys from environment variables
groq_api_key = os.getenv('GROQ_API_KEY')

# Set the title of the Streamlit app
st.title("ChatGroq With Llama3 Demo")

# Initialize the ChatGroq LLM with a specific model name
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Create a structured prompt template for the LLM
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

# Function to handle vector embedding for documents
def vector_embedding():
    # Check if embeddings and vectors are already in session state
    if "vectors" not in st.session_state:
        # Initialize embeddings using OpenAI
        st.session_state.embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
        # Load documents from the ./us_census directory
        st.session_state.loader = PyPDFDirectoryLoader("./us_census")
        st.session_state.docs = st.session_state.loader.load()  # Load all documents
        # Split large documents into smaller chunks with some overlap
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        # Create vector embeddings for the document chunks and store them in FAISS
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Input field for the user to ask questions
prompt1 = st.text_input("Enter Your Question From Documents")

# Button to trigger document embedding and vector creation
if st.button("Documents Embedding"):
    vector_embedding()  # Call the embedding function
    st.write("Vector Store DB Is Ready")  # Display success message

import time  # For tracking the response time

# If the user has entered a prompt
if prompt1:
    # Create a document chain using the LLM and structured prompt
    document_chain = create_stuff_documents_chain(llm, prompt)
    # Create a retriever from the FAISS vector store
    retriever = st.session_state.vectors.as_retriever()
    # Create a retrieval chain combining the retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # Start tracking response time
    start = time.process_time()
    # Get the response by invoking the retrieval chain with the user input
    response = retrieval_chain.invoke({'input': prompt1})
    print("Response time :", time.process_time() - start)  # Log the response time
    st.write(response['answer'])  # Display the answer to the user

    # Expandable section to display document similarity search results
    with st.expander("Document Similarity Search"):
        # Iterate through relevant document chunks in the response
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)  # Display the content of each chunk
            st.write("--------------------------------")  # Separator for readability