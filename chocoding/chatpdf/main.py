# import torch
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # Use GPU for computation
# else:
#     device = torch.device("cpu")  # Fallback to CPU
# print(f"device: {device}")
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import RetrievalQA

import os
import streamlit as st
import tempfile


V_STORE_PATH = "./v_store"

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    # Load
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

st.title("ChatPDF with Ollama")
st.write("---")
st.write("Upload a PDF file to start chatting with it!")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    # Save the uploaded file to disk
     ###### LOAD -------------------------------------------------------------------
    pages = pdf_to_document(uploaded_file)


    ###### SPLIT ------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.split_documents(pages)
    ###### TOKENIZE  --------------------------------------------------------------
    embedding_model = OllamaEmbeddings(
        model="llama3.3:latest",
        base_url="http://172.17.0.2:11434",
    )
    ### STORE (Vector Store)  --------------------------------------------------------
    db = Chroma.from_documents(texts, embedding_model, persist_directory=V_STORE_PATH)
    st.write("File uploaded and store chroma successfully!")


###### Retreve (QUERY) ------------------------------------------------------------------
st.header("Ask your PDF")
question = st.text_input("Enter your question here")

try:
    db = Client()
    db.load_from_directory(V_STORE_PATH)
except Exception as e:
    print(f"Error loading Chroma DB from directory: {e}")

if st.button('Ask'):
    with st.spinner("Thinking..."):
        qa_chain = RetrievalQA.from_chain_type(
            retriever = db.as_retriever(),
            llm = ChatOllama(
                    base_url="http://172.17.0.2:11434",
                    model = "llama3.3:latest",
                    temperature = 0,
                    num_predict = 256,
                )
        )
        # 질문에 대한 답변 출력하기
        answer = qa_chain.invoke({"query": question})
        print(answer)
        st.write(answer['result'])


