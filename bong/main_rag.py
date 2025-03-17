
from rosie.log import logger
logger.info("start main_rag!!!")
from rosie.raptor_me import *

from langchain.chat_models import init_chat_model
model = init_chat_model(
    "llama3.3",
    model_provider="ollama",
    base_url="172.17.0.2:11434"
)

# ============  load doc ============
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
# FILE_PATH = "https://arxiv.org/pdf/2408.09869"
FILE_PATH = "./SPRI_AI_Brief_2023년12월호_F.pdf"
loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=ExportType.MARKDOWN
)
documents = loader.load()

# from langchain_text_splitters import RecursiveCharacterTextSplitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000, chunk_overlap=200, add_start_index=True
# )
# all_splits = text_splitter.split_documents(documents)
# # len(all_splits)

# =========== RAPTOR ===================
docs_texts = [d.page_content for d in documents]
# 트리 구축
leaf_texts = docs_texts  # 문서 텍스트를 리프 텍스트로 설정
results = recursive_embed_cluster_summarize(
    leaf_texts, level=1, n_levels=3
)  # 재귀적으로 임베딩, 클러스터링 및 요약을 수행하여 결과를 얻음


# ========== Vector Store ======================
from langchain_community.vectorstores import FAISS
# leaf_texts를 복사하여 all_texts를 초기화합니다.
all_texts = leaf_texts.copy()
# 각 레벨의 요약을 추출하여 all_texts에 추가하기 위해 결과를 순회합니다.
for level in sorted(results.keys()):
    # 현재 레벨의 DataFrame에서 요약을 추출합니다.
    summaries = results[level][1]["summaries"].tolist()
    # 현재 레벨의 요약을 all_texts에 추가합니다.
    all_texts.extend(summaries)
# 이제 all_texts를 사용하여 FAISS vectorstore를 구축합니다.
vectorstore = FAISS.from_texts(texts=all_texts, embedding=embeddings)

import os
DB_INDEX = "RAPTOR"
# 로컬에 FAISS DB 인덱스가 이미 존재하는지 확인하고, 그렇다면 로드하여 vectorstore와 병합한 후 저장합니다.
if os.path.exists(DB_INDEX):
    local_index = FAISS.load_local(DB_INDEX, embeddings)
    local_index.merge_from(vectorstore)
    local_index.save_local(DB_INDEX)
else:
    vectorstore.save_local(folder_path=DB_INDEX)

# retriever 생성
retriever = vectorstore.as_retriever()


# =============================================================
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_ollama import OllamaLLM
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_core.vectorstores import VectorStoreRetriever
from langchain.vectorstores import FAISS
import faiss
# 프롬프트 생성
prompt = hub.pull("rlm/rag-prompt")
# 문서 포스트 프로세싱

def format_docs(docs):
    # 문서의 페이지 내용을 이어붙여 반환합니다.
    return "\n\n".join(doc.page_content for doc in docs)

llm = OllamaLLM(model="llama3.3")

document_content_description = "인공지는 산업의 최신동향"
metadata_field_info = [
    AttributeInfo(
        name="manual",
        description="SPRi AI Brief",
        type="string",
    ),
    # Additional metadata fields...
]

retriever = vectorstore.as_retriever()
# RAG 체인 정의
rag_chain = (
    # 검색 결과를 포맷팅하고 질문을 처리합니다.
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt  # 프롬프트를 적용합니다.
    | model  # 모델을 적용합니다.
    | StrOutputParser()  # 문자열 출력 파서를 적용합니다.
)




rag_chain.invoke("첨단 AI 시스템의 위험 관리를 위한 국제 행동강령에 대해 알려줘")

for token in rag_chain.stream("첨단 AI 시스템의 위험 관리를 위한 국제 행동강령에 대해 알려줘. 내용은 아주 상세하게 markdown으로 한글로 작성해줘. "):
    # print(token.content, end="")
    print(token, end="")

for token in rag_chain.stream("삼성전자의 자체 개발 생성 AI에 대해 알려줘. 내용은 아주 상세하게 markdown으로 한글로 작성해줘. "):
    # print(token.content, end="")
    print(token, end="")