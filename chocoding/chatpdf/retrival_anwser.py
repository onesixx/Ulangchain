from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
import os
import streamlit as st
import tempfile
# Define the model name you want to use
#model_name = "nomic-embed-text:latest"
model_name = "nlpai-lab/KoE5"
#model_name = "jhgan/ko-sroberta-multitask"
 # or any other supported model like all-MiniLM-L6-v2, jhgan/ko-sroberta-multitask
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

db = Chroma(persist_directory="./chroma_db",
        embedding_function=embedding_model)

llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)

qa_chain = RetrievalQA.from_chain_type(
    retriever = db.as_retriever(),
    llm = llm
)

#==============================================================================
st.title("Demo for 동한")
st.write("---")
st.header("Ask about 'a lucky day'")
question = st.text_input("Enter your question here")

if st.button('Ask'):
    with st.spinner("Thinking..."):
        # question = "아내가 먹고 싶은 음식은?"
        # qry = db.similarity_search(question, k=3)
        # qry_and_scores = db.similarity_search_with_score(question, k=3)
        # logger.info(f"answer is from {qry[0].page_content}")
        answer = qa_chain.invoke({"query": question})
        st.write(answer['result'])