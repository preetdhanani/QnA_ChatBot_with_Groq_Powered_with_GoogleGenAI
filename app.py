import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Streamlit App
st.title("Q&A ChatBot with Custom Context with GROQ and Powerd with GoogleGenAI")

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

prompt = ChatPromptTemplate.from_template("""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions: {input}
""")
def vector_embedding(uploaded_file):
    if "vectors" not in st.session_state:
        # Create tempDir if it doesn't exist
        if not os.path.exists("tempDir"):
            os.makedirs("tempDir")
        
        # Save uploaded file temporarily
        temp_file_path = os.path.join("tempDir", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load the uploaded file
        st.session_state.loader = PyPDFDirectoryLoader("tempDir")  # Directory with the uploaded file
        st.session_state.docs = st.session_state.loader.load()  # Document Loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])  # Splitting
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # Embedding model
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)  # Vector embeddings



uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")


if uploaded_file:
    if st.button("Process Document"):
        vector_embedding(uploaded_file)
        st.write("Vector Store DataBase Is Ready")

prompt1 = st.text_input("From Documents ask Questions")


if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write(f"Response time: {time.process_time() - start} seconds")
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
