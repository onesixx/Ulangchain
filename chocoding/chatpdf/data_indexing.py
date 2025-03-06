from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

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

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "nlpai-lab/KoE5"
# Initialize embedding_model based on model_name using if-else
if model_name == "nomic-embed-text:latest":
    embedding_model = OllamaEmbeddings(
        base_url="http://172.17.0.2:11434",
        model=model_name
    )
elif model_name == "all-minilm:l6-v2":
    embedding_model = OllamaEmbeddings(
        base_url="http://172.17.0.2:11434",
        model=model_name
    )
elif model_name == "nlpai-lab/KoE5":
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name
    )
else:
    raise ValueError(f"Unsupported model_name: {model_name}. Supported models are: ['nomic-embed-text:latest', 'all-minilm:l6-v2', 'jhgan/ko-sroberta-multitask']")
dimension_size = len(embedding_model.embed_query("hello world"))



# ------ Load ------
loader = PyPDFLoader(r"./data/aLuckyDay.pdf")
pages = loader.load_and_split()

# tokenizer
# import tiktoken
# tokenizer = tiktoken.get_encoding("cl100k_base")
# def tiktoken_len(text):
#     tokens = tokenizer.encode(text)
#     return len(tokens)


# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", " ", "", "."],
#     chunk_size=300,
#     chunk_overlap=20,
#     length_function=len, #tiktoken_len,
#     is_separator_regex=False,
# )

from langchain_experimental.text_splitter import SemanticChunker
text_splitter = SemanticChunker(
    OllamaEmbeddings(
        base_url="http://172.17.0.2:11434",
        model= model_name #"llama3.3:latest",
    )
)

docs = text_splitter.split_documents(pages)
print( [len(txt.page_content) for txt in docs] )

docs[0].page_content
docs[8].page_content

# ----------------------------------------------------------

# toke
#  V_STORE_PATH = "./v_store"
Chroma().delete_collection()
db = Chroma.from_documents(docs, embedding_model,
        persist_directory="./chroma_db")


#



#------------------------------------------------------------
llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)

if os.path.exists("v_store/index.faiss"):
    st.write("Vector store is already loaded.")
    vectorstore = lc_faiss.load_local(V_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
    logger.info(f"vectorstore is {vectorstore}")
else:
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
        vectorstore.save_local(V_STORE_PATH)
        # vectorstore = lc_faiss.from_documents(texts, embedding_model)

        st.write("File uploaded and store chroma successfully!")
