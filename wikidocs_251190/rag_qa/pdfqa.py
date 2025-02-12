import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device in use: {device}")

# LangSmith 추적을 설정합니다. https://smith.langchain.com
# https://wikidocs.net/250954
from dotenv import load_dotenv
load_dotenv(".env")
from langchain_teddynote import logging
logging.langsmith("CH12-RAG")
# logging.langsmith("CH12-RAG", set_enable=False)

from langchain_community.document_loaders import PyMuPDFLoader #PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS #Chroma
from langchain_community.vectorstores import Chroma

from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import RetrievalQA

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import os
import streamlit as st
import tempfile

V_STORE_PATH = "./v_store"

# def pdf_to_document(uploaded_file):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     # Load
#     loader = PyMuPDFLoader(temp_filepath)
#     pages = loader.load_and_split()
#     return pages

# st.title("ChatPDF with Ollama")
# st.write("---")
# st.write("Upload a PDF file to start chatting with it!")

# File uploader
# uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
uploaded_file = True
if uploaded_file is not None:
    # Save the uploaded file to disk
    ###### 1. LOAD Document -------------------------------------------------------------------
    # loader = pdf_to_document(uploaded_file)

    loader = PyMuPDFLoader("data/SPRI_AI_Brief_2023년12월호_F.pdf")
    docs = loader.load()


    ###### 2. SPLIT ------------------------------------------------------------------
    # text_splitter = RecursiveCharacterTextSplitter(
    #     chunk_size=300,
    #     chunk_overlap=20,
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    # split_docs  = text_splitter.split_documents(loader)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    ###### 3. 임베딩(Embedding) 생성   :: TOKENIZE  --------------------------------------------------------------
    embedding_model = OllamaEmbeddings(
        model="llama3.3:latest",
        base_url="http://172.17.0.2:11434",
    )

    ### 4. STORE (Vector Store)  --------------------------------------------------------
    # vectorstore = Chroma.from_documents(split_docs , embedding_model, persist_directory=V_STORE_PATH)
    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embedding_model)

    # st.write("File uploaded and store chroma successfully!")

    ###### 5. 검색기 생성 Retrever (QUERY) ------------------------------------------------------------------
    retriever = vectorstore.as_retriever()

    ###### 6 프롬프트 생성(Create Prompt) ------------------------------------------------------------------
    prompt = PromptTemplate.from_template(
        """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in Korean.

    #Question:
    {question}
    #Context:
    {context}

    #Answer:"""
    )
    # st.header("Ask your PDF")
    # question = st.text_input("Enter your question here")

    # if st.button('Ask'):
    #     with st.spinner("Thinking..."):
    #         qa_chain = RetrievalQA.from_chain_type(
    #             retriever = vectorstore.as_retriever(),
    #             llm = ChatOllama(
    #                     apiBase="http://172.17.0.2:11434",
    #                     model = "llama3.3:latest",
    #                     temperature = 0,
    #                     num_predict = 256,
    #                 )
    #         )
    #         # 질문에 대한 답변 출력하기
    #         answer = qa_chain.invoke({"query": question})
    #         st.write(answer['result'])


    # 단계 7: 언어모델(LLM) 생성
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ChatOllama(
        base_url="http://172.17.0.2:11434",
        model = "llama3.3:latest",
        temperature = 0,
        # num_predict = 256,
    )
    ###### 단계 8: QA 체인(Chain) 생성 ------------------------------------------------------------------
    chain = RetrievalQA.from_chain_type(
        retriever = retriever,
        llm = llm,

    )
    # chain = (
    #     {"context": retriever, "question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    ###### 단계 9: 체인 실행(Run Chain) : 질문 입력,  답변 출력하기 --------------------------------------------------------------
    question = "삼성전자가 자체 개발한 AI 의 이름은?"
    response = chain.invoke({"query": question})
    print(response)


# ---------------------------------------------------------------------------------------------------------------------
# import logging
# from langchain.chains import RetrievalQA
# import httpx
# logging.basicConfig(level=logging.DEBUG)

# # httpx의 요청을 로깅
# client = httpx.Client(
#   event_hooks={
#       "request": [lambda request: print(f"LangChain Request: {request.url}")],
#       "response": [lambda response: print(f"LangChain Response: {response.status_code}")]})

# # LangChain 실행
# response = chain.invoke({"query": question})
# print(response)
