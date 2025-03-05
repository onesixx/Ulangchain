from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!!")

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

import faiss
from langchain_community.vectorstores import FAISS as lc_faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
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

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

embedding_model = OllamaEmbeddings(
    base_url="http://172.17.0.2:11434",
    model= "llama3.3:latest", #"all-minilm:l6-v2",
)
# 임베딩 차원 크기를 계산
# dimension_size = len(embedding_model.embed_query("hello world"))
dimension_size = 8192 # 384


llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)

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
    texts = text_splitter.split_documents(pages)
    ### STORE (Vector Store)  --------------------------------------------------------

    res = faiss.StandardGpuResources()  # GPU 리소스 초기화
    cpu_index = faiss.IndexFlatL2(dimension_size)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

    vectorstore = lc_faiss(
        embedding_function=embedding_model,
        index=gpu_index,
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
    vectorstore.add_documents(documents=texts)

    # db = Chroma.from_documents(
    #     texts,
    #     embedding_model,
    #     persist_directory=V_STORE_PATH,
    #     #collection_dim=384  # 임베딩 모델의 출력 차원과 일치시킵니다.
    # )
    st.write("File uploaded and store chroma successfully!")

    ###### Retreve (QUERY) ------------------------------------------------------------------
    st.header("Ask your PDF")
    question = st.text_input("Enter your question here")

    if st.button('Ask'):
        with st.spinner("Thinking..."):
            qa_chain = RetrievalQA.from_chain_type(
                retriever = vectorstore.as_retriever(),
                llm = llm
            )
            # add
            results = vectorstore.similarity_search(question, k=5)
            for result in results:
                print(result.page_content)


            # 질문에 대한 답변 출력하기
            answer = qa_chain.invoke({"query": question})
            print(answer)
            st.write(answer['result'])
