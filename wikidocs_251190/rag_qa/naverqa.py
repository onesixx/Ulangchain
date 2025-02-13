from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!!")
# LangSmith 추적을 설정합니다. https://smith.langchain.com
# Chain 이나 Agent 내부에서 정확히 무슨 일이 일어나고 있는지 조사
from dotenv import load_dotenv
load_dotenv(".env")
from langchain_teddynote import logging
from langchain_teddynote.messages import stream_response
logging.langsmith("CH12-RAG")



import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS   # $ lscpu | grep avx2
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_ollama import ChatOllama

###### 로드-분할-임베딩-저장
MODEL="llama3.3:latest"
BASE_URL="http://172.17.0.2:11434"

# 1 - Load the document --------------------------------------------------------
loader = WebBaseLoader(
    web_paths=("https://n.news.naver.com/article/437/0000378416",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            "div",
            attrs={"class": ["newsct_article _article_body", "media_end_head_title"]},
        )
    ),
)
docs = loader.load()
# logger.info(f"문서의 수: {len(docs)}")

# 2 - Split the document into smaller parts ------------------------------------
txt_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
)
split_docs = txt_splitter.split_documents(docs)
# logger.info(f"분할된 문서의 수: {len(split_docs)}")

# 3 - Store the embeddings -----------------------------------------------------
vectorstore = FAISS.from_documents(
    documents=split_docs,
    embedding=OllamaEmbeddings(model=MODEL, base_url=BASE_URL)
)
retriever = vectorstore.as_retriever()

###### prompt - llm - chain - query - parse - output
# 1. Define the prompt template --------------------------------------------------
prompt = PromptTemplate.from_template(
    """당신은 질문-답변(Question-Answering)을 수행하는 친절한 AI 어시스턴트입니다. 당신의 임무는 주어진 문맥(context) 에서 주어진 질문(question) 에 답하는 것입니다.
검색된 다음 문맥(context) 을 사용하여 질문(question) 에 답하세요. 만약, 주어진 문맥(context) 에서 답을 찾을 수 없다면, 답을 모른다면 `주어진 정보에서 질문에 대한 정보를 찾을 수 없습니다` 라고 답하세요.
한글로 답변해 주세요. 단, 기술적인 용어나 이름은 번역하지 않고 그대로 사용해 주세요.

#Question:
{question}

#Context:
{context}

#Answer:"""
)
# 2. Chain ---------------------------------------------------
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0)
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# rag_chain = RetrievalQA.from_chain_type(
#     retriever=retriever,
#     chain_type="stuff",
#     chain_type_kwargs={"prompt": prompt},
#     return_source_documents=True,
#     llm = ChatOllama(model=MODEL, base_url=BASE_URL, temperature=0)
# )

# 3. Query ---------------------------------------------------
question = "부영그룹의 출산 장려 정책에 대해 설명해주세요."

answer = rag_chain.stream(question)
stream_response(answer)

answer = rag_chain.invoke(question)
print(answer)