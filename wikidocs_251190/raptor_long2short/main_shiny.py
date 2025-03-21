from rosie.log import setup_logging
from rosie.log import logger
logger.info("main.py start!!!")
# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
# LLM의 정보검색능력을 향상시키기 위한 방식
import os
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import tiktoken
import matplotlib.pyplot as plt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
#from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama

import pickle

# # LCEL 문서 로드
# 1 - Load the document --------------------------------------------------------
# urls = [
#     ("https://shiny.posit.co/py/docs/overview.html", "overview"),
#     ("https://shiny.posit.co/py/api/core/", "api_core"),
#     ("https://shiny.posit.co/py/components/", "components"),
#     ("https://shiny.posit.co/py/gallery/", "gallery"),
# ]
# docs = []
# for url_info in urls:
#     url, description = url_info
#     logger.info(f"Loading {description} from {url}")
#     loader = RecursiveUrlLoader(
#         url=url,
#         max_depth=6, #20 if description == "LCEL" else 1,
#         extractor=lambda x: Soup(x, "html.parser").text
#     )
#     loaded_docs = loader.load()
#     docs.extend(loaded_docs)
# docs_texts = [d.page_content for d in docs]
# def find_page_not_found_indices(lst):
#     indices = []
#     for index, element in enumerate(lst):
#         if "Page not found" in element:
#             indices.append(index)
#     return indices
# exclude_indices =find_page_not_found_indices(docs_texts)
# docs = [element for index, element in enumerate(docs) if index not in exclude_indices]


# with open('temp/docs.pkl', 'wb') as file:
#     pickle.dump(docs, file)

# ------------------------------------------------------
path = "/workspaces/wikidocs_251190/raptor_long2short/temp/docs.pkl"
with open(path, 'rb') as file:
    docs = pickle.load(file)
docs_texts = [d.page_content for d in docs]


##### 2. 각 문서에 대한 토큰 수 계산
# def num_tokens_from_string(string: str, encoding_name: str) -> int:
#     # 주어진 문자열에서 토큰의 개수를 반환합니다.
#     encoding = tiktoken.get_encoding(encoding_name)
#     num_tokens = len(encoding.encode(string))
#     return num_tokens

# counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

# fig = plt.figure()
# fig, ax = plt.subplots()
# ax.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
# ax.set_title("Token Counts in LCEL Documents")
# ax.set_xlabel("Token Count")
# ax.set_ylabel("Frequency")
# ax.grid(axis="y", alpha=0.75)
# plt.show()

# fig = plt.figure()
# fig, ax = plt.subplots()
# ax.scatter(range(len(counts)), counts, color="green", alpha=0.5, s=6)
# ax.set_title("Token Counts in LCEL Documents")
# ax.set_xlabel("Document Index")
# ax.set_ylabel("Token Count")
# ax.grid(alpha=0.75)
# plt.show()

# docs[0].__dict__.keys()
# docs[0].__dict__

# 문서 텍스트를 연결합니다. (출처 메타데이터 기준으로 정렬)
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))
# # 역순으로 배열된 문서의 내용을 연결합니다.
concatenated_content = "\n\n\n --- \n\n\n".join(
    [ doc.page_content for doc in d_reversed]
)
# print(
#     "Num tokens in all context: %s"  # 모든 문맥에서의 토큰 수를 출력합니다.
#     % num_tokens_from_string(concatenated_content, "cl100k_base")
# )

# 2 - Split the document into smaller parts ------------------------------------
# 재귀적 문자 텍스트 분할기를 초기화합니다.
#  => 토큰 인코더를 사용하여 청크 크기와 중복을 설정합니다.
chunk_size_tok = 2000
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok,
    chunk_overlap=0
)
texts_split = text_splitter.split_text(
    concatenated_content
)

# 3 - Store the embeddings -----------------------------------------------------

# embeddings 인스턴스를 생성합니다.
# from langchain_openai import OpenAIEmbeddings
# embd = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())

embd = OllamaEmbeddings(
    model="all-minilm:l6-v2", # 더 가벼운 모델로 교체
    base_url="http://172.17.0.2:11434/"
)

# cache_dir = "./cache/"
# if not os.path.exists(cache_dir):
#     os.makedirs(cache_dir)
# store = LocalFileStore(cache_dir)

store = LocalFileStore("./cache/")

cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embd,
    store,
    namespace=embd.model  #.replace(":", "_")
)
# cached_embeddings.embed_documents(["test text"])

# 모델을 초기화 합니다. --------------------------------------------------
class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

model = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0,
    streaming=True,
    #callbacks=[StreamCallback()],
    # num_predict = 256,
)

from langchain_core.output_parsers import StrOutputParser
from raptor_.fn_raptor import recursive_embed_cluster_summarize

# import importlib
# from raptor_ import fn_raptor
# importlib.reload(fn_raptor.recursive_embed_cluster_summarize)

# 트리 구축 ------------------------------------------------------
leaf_texts = docs_texts  # 문서 텍스트를 리프 텍스트로 설정
results = recursive_embed_cluster_summarize(
    leaf_texts, level=1, n_levels=3, model=model
)  # 재귀적으로 임베딩, 클러스터링 및 요약을 수행하여 결과를 얻음



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
vectorstore = FAISS.from_texts(texts=all_texts, embedding=embd)




# DB 를 로컬에 저장합니다.
DB_INDEX = "v_store"


# 로컬에 FAISS DB 인덱스가 이미 존재하는지 확인하고, 그렇다면 로드하여 vectorstore와 병합한 후 저장합니다.
if os.path.exists(DB_INDEX):
    local_index = FAISS.load_local(DB_INDEX, embd, allow_dangerous_deserialization=True)
    local_index.merge_from(vectorstore)
    local_index.save_local(DB_INDEX)
else:
    vectorstore.save_local(folder_path=DB_INDEX)
retriever = vectorstore.as_retriever()
# --------------------------------------------------------------------------
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
# 프롬프트 생성
prompt = hub.pull("rlm/rag-prompt")
# 문서 포스트 프로세싱
def format_docs(docs):
    # 문서의 페이지 내용을 이어붙여 반환합니다.
    return "\n\n".join(doc.page_content for doc in docs)
# RAG 체인 정의
rag_chain = (
    # 검색 결과를 포맷팅하고 질문을 처리합니다.
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt  # 프롬프트를 적용합니다.
    | model  # 모델을 적용합니다.
    | StrOutputParser()  # 문자열 출력 파서를 적용합니다.
)

# 추상적인 질문 실행
_ = rag_chain.invoke("shiny를 기준으로 전체 문서의 핵심 주제에 대해 설명해주세요.")
print(_)
# Low Level 질문 실행
_ = rag_chain.invoke("shiny for python을 사용하여 Restaurant tips dashboard을 활용한 예시 코드를 작성해 주세요.")
print(_)
# Low Level 질문 실행
_ = rag_chain.invoke("shiny에서 사용하는 reactive 방법과 예시 코드를 작성해 주세요.")
print(_)

logger.info("RAG Chain Test Completed")
