from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!!")
# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
# LLM의 정보검색능력을 향상시키기 위한 방식
import os
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from bs4 import BeautifulSoup as Soup
import tiktoken
import matplotlib.pyplot as plt

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.storage import LocalFileStore
from langchain.embeddings import OllamaEmbeddings
from langchain.embeddings import CacheBackedEmbeddings

from langchain.callbacks.base import BaseCallbackHandler
from langchain_ollama import ChatOllama

# # LCEL 문서 로드
# url = "https://python.langchain.com/docs/expression_language/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=20, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs = loader.load()

# # PydanticOutputParser를 사용한 LCEL 문서 로드 (기본 LCEL 문서 외부)
# url = "https://python.langchain.com/docs/how_to/output_parser_structured/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs_pydantic = loader.load()

# # Self Query를 사용한 LCEL 문서 로드 (기본 LCEL 문서 외부)
# url = "https://python.langchain.com/docs/how_to/self_query/"
# loader = RecursiveUrlLoader(
#     url=url, max_depth=1, extractor=lambda x: Soup(x, "html.parser").text
# )
# docs_sq = loader.load()


# # 문서 텍스트
# docs.extend([*docs_pydantic, *docs_sq])
# docs_texts = [d.page_content for d in docs]

# 1 - Load the document --------------------------------------------------------
urls = [
    ("https://python.langchain.com/docs/expression_language/", "LCEL"),
    ("https://python.langchain.com/docs/how_to/output_parser_structured/", "PydanticOutputParser"),
    ("https://python.langchain.com/docs/how_to/self_query/", "SelfQuery")
]
docs = []
for url_info in urls:
    url, description = url_info
    loader = RecursiveUrlLoader(
        url=url,
        max_depth=20 if description == "LCEL" else 1,
        extractor=lambda x: Soup(x, "html.parser").text
    )
    loaded_docs = loader.load()
    docs.extend(loaded_docs)
docs_texts = [d.page_content for d in docs]


##### 2. 각 문서에 대한 토큰 수 계산
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    # 주어진 문자열에서 토큰의 개수를 반환합니다.
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

counts = [num_tokens_from_string(d, "cl100k_base") for d in docs_texts]

plt.figure(figsize=(10, 6))
plt.hist(counts, bins=30, color="blue", edgecolor="black", alpha=0.7)
plt.title("Token Counts in LCEL Documents")
plt.xlabel("Token Count")
plt.ylabel("Frequency")
plt.grid(axis="y", alpha=0.75)
plt.show

# 문서 텍스트를 연결합니다. (출처 메타데이터 기준으로 정렬)
d_sorted = sorted(docs, key=lambda x: x.metadata["source"])
d_reversed = list(reversed(d_sorted))  # 정렬된 문서를 역순으로 배열합니다.
# 역순으로 배열된 문서의 내용을 연결합니다.
concatenated_content = "\n\n\n --- \n\n\n".join(
    [ doc.page_content for doc in d_reversed]
)
print(
    "Num tokens in all context: %s"  # 모든 문맥에서의 토큰 수를 출력합니다.
    % num_tokens_from_string(concatenated_content, "cl100k_base")
)

# 2 - Split the document into smaller parts ------------------------------------
chunk_size_tok = 2000
# 재귀적 문자 텍스트 분할기를 초기화합니다. 토큰 인코더를 사용하여 청크 크기와 중복을 설정합니다.
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size_tok,
    chunk_overlap=0
)
texts_split = text_splitter.split_text(
    concatenated_content
)

# 3 - Store the embeddings -----------------------------------------------------

# embeddings 인스턴스를 생성합니다.
# embd = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=())
MODEL="llama3.3:latest"
BASE_URL="http://172.17.0.2:11434"
embd = OllamaEmbeddings(
    model=MODEL,
    base_url=BASE_URL,
)
store = LocalFileStore("./cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
    embd,
    store,
    namespace=embd.model
)

# 모델을 초기화 합니다. --------------------------------------------------
class StreamCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs):
        print(token, end="", flush=True)

# from langchain_anthropic import ChatAnthropic
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# ChatAnthropic 모델을 초기화합니다. 온도는 0으로 설정하고, 모델은 "claude-3-opus-20240229"를 사용합니다.
# model = ChatAnthropic(temperature=0, model="claude-3-opus-20240229")
model = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0,
    streaming=True,
    callbacks=[StreamCallback()],
    # num_predict = 256,
)


from langchain_core.output_parsers import StrOutputParser
from raptor_.fn_raptor import recursive_embed_cluster_summarize

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

DB_INDEX = "RAPTOR"

# 로컬에 FAISS DB 인덱스가 이미 존재하는지 확인하고, 그렇다면 로드하여 vectorstore와 병합한 후 저장합니다.
if os.path.exists(DB_INDEX):
    local_index = FAISS.load_local(DB_INDEX, embd)
    local_index.merge_from(vectorstore)
    local_index.save_local(DB_INDEX)
else:
    vectorstore.save_local(folder_path=DB_INDEX)

retriever = vectorstore.as_retriever()
logger.info(f"number of docs_texts: {len(docs_texts)}")
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
_ = rag_chain.invoke("전체 문서의 핵심 주제에 대해 설명해주세요.")
# Low Level 질문 실행
_ = rag_chain.invoke("PydanticOutputParser 을 활용한 예시 코드를 작성해 주세요.")
# Low Level 질문 실행
_ = rag_chain.invoke("self-querying 방법과 예시 코드를 작성해 주세요.")
