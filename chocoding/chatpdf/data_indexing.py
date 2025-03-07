#from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


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

from util import *
from langchain_experimental.text_splitter import SemanticChunker


# ------ Load ------
loader = PyPDFLoader(r"./data/aLuckyDay.pdf")
pages = loader.load_and_split()

# tokenizer
# import tiktoken
# tokenizer = tiktoken.get_encoding("cl100k_base")
# def tiktoken_len(text):
#     tokens = tokenizer.encode(text)
#     return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", "", "."],
    chunk_size=300,
    chunk_overlap=20,
    length_function=len, #tiktoken_len,
    #is_separator_regex=False,
)

embedding_model = load_LLM_embeddings("nomic-embed-text:latest")
# embedding_model = load_LLM_embeddings("all-minilm:l6-v2")
# text_splitter = SemanticChunker(embedding_model)

docs = text_splitter.split_documents(pages)
print( [len(txt.page_content) for txt in docs] )

logger.info(docs[0].page_content)
logger.info(docs[8].page_content)

# ----------------------------------------------------------

# toke
#  V_STORE_PATH = "./v_store"
Chroma().delete_collection()
db = Chroma.from_documents(docs, embedding_model,
        persist_directory="./chroma_db")
