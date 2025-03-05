# from langchain_anthropic import ChatAnthropic
# chat = ChatAnthropic(
#     model_name ="claude-3-opus-20240229",
#     anthropic_api_key="YOUR_API_KEY"
# )
# chat.invoke("안녕~ 너를 소개해줄래?")

# ----------
from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!! vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
# ----------
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_ollama import OllamaEmbeddings
# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
#openai_embedding=OpenAIEmbeddings(model = 'text-embedding-3-small')
embedding_model = OllamaEmbeddings(
    base_url="http://172.17.0.2:11434",
    model= "nomic-embed-text:latest", # "all-minilm:l6-v2" , "llama3.3:latest",
)

model_name = "jhgan/ko-sroberta-multitask"
embedding_model= HuggingFaceEmbeddings(
    model_name=model_name
)


loader = PyPDFLoader(r"./data/대한민국 헌법.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=0
)
docs = text_splitter.split_documents(pages)

db0 = Chroma.from_documents(docs, embedding_model, persist_directory="./chroma_db")
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)
query = "대통령의 임기는"
docs = db.similarity_search(query, k=5)
for doc in docs:
    print(doc)

Chroma().delete_collection()
