import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google.cloud import aiplatform
from dotenv import load_dotenv
import pdfplumber
import time

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="Gemma Model Document Q&A", layout="wide")

st.title("Gemma Model Document Q&A")
st.markdown("""
    Welcome to the **Gemma Model Document Q&A** application! 
    Upload your PDF documents, create a search index, and ask questions based on the contents.
    """, unsafe_allow_html=True)

llm = ChatGroq(groq_api_key=groq_api_key, model="Gemma-7b-it")

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

def vector_embedding(uploaded_files):
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Handle file uploads
        if not uploaded_files:
            st.error("Please upload PDF files.")
            return
        
        # Create a list to store document texts
        docs = []
        for uploaded_file in uploaded_files:
            with pdfplumber.open(uploaded_file) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text()
                # Append document with text
                docs.append(CustomDocument(page_content=text, metadata={"source": uploaded_file.name}))
        
        if not docs:
            st.error("No documents loaded. Please check the files.")
            return
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(docs)
        
        if not st.session_state.final_documents:
            st.error("Document splitting failed. No chunks created.")
            return
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        
        st.success("Vector store created successfully!")

# Custom document class
class CustomDocument:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Create a sidebar with file uploader
st.sidebar.header("Upload PDF Documents")
uploaded_files = st.sidebar.file_uploader("Upload your PDFs", type=["pdf"], accept_multiple_files=True)

# Add button to create vector store
if st.sidebar.button("Create Vector Store"):
    with st.spinner("Creating vector store..."):
        vector_embedding(uploaded_files)

# Input for user query
st.markdown("### Ask a Question")
prompt1 = st.text_input("Enter your Prompt")

if st.button("Submit"):
    if prompt1:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])
        
        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("---------------------------------")
    else:
        st.warning("Please enter a prompt to get a response.")