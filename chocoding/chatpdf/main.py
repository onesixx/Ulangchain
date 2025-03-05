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

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    # ------ Load ------
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

# tokenizer
import tiktoken
tokenizer = tiktoken.get_encoding("cl100k_base")
def tiktoken_len(text):
    tokens = tokenizer.encode(text)
    return len(tokens)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=tiktoken_len,
    is_separator_regex=False,
)
# toke
V_STORE_PATH = "./v_store"
embedding_model = OllamaEmbeddings(
    base_url="http://172.17.0.2:11434",
    model= "all-minilm:l6-v2", # "llama3.3:latest",
)
dimension_size = len(embedding_model.embed_query("hello world"))
# dimension_size = 8192 # 384

llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)

st.title("ChatPDF with Ollama")
st.write("---")
st.write("Upload a PDF file to start chatting with it!")

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


###### Retreve (QUERY) ------------------------------------------------------------------
st.header("Ask your PDF")
question = st.text_input("Enter your question here")

if st.button('Ask'):
    with st.spinner("Thinking..."):
        if vectorstore is not None:
            qa_chain = RetrievalQA.from_chain_type(
                retriever = vectorstore.as_retriever(),
                llm = llm
            )

            docs = vectorstore.similarity_search(question)
            docs_and_scores = vectorstore.similarity_search_with_score(question, k=3)
            logger.info(f"docs_and_scores are {docs_and_scores}")
            for doc in docs_and_scores:
                logger.info(f"meta ==> {doc[0].metadata}")
                logger.info(f"page ==> {doc[0].page_content}")

            logger.info(f"answer is from {docs[0].page_content}")

            answer = qa_chain.invoke({"query": question})
            print(answer)
            st.write(answer['result'])
        else:
            st.error("No file uploaded yet.")
