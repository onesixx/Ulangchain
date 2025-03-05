
# import torch
# from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma

from langchain_ollama import ChatOllama
from langchain.retrievers import MultiQueryRetriever

from langchain.chains import RetrievalQA
import faiss
import pickle
from langchain_community.vectorstores import FAISS as lc_faiss
from langchain.docstore import InMemoryDocstore

import os
import tempfile
# 현재 디렉토리아래 data폴더에 있는 pdf파일만을 선별하고
#  pyPDFLoader로 불러온다.
from pdfplumber import open as pdf_open


V_STORE_PATH = "./v_store"

# splitter 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

embedding_model = OllamaEmbeddings(
    base_url="http://172.17.0.2:11434",
    model="all-minilm:l6-v2",
)
# sample_text = "This is a test sentence."
# embedding = embedding_model.embed_query(sample_text)
# print(f"Embedding dimension: {len(embedding)}")
# 임베딩 차원 크기를 계산
# dimension_size = len(embedding_model.embed_query("hello world"))
# print(dimension_size)

llm = ChatOllama(
    apiBase="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0,
    num_predict = 256,
    streaming = True,
)

# ------   load pdf file  ------
dir_path = "./data"
for filename in os.listdir(dir_path):
    if not filename.endswith('.pdf'):
        continue  # Skip non-PDF files
    fname = os.path.join(dir_path, filename)
    # Load the extracted text into pyPDFLoader
    loader = PyPDFLoader(fname)
    pages = loader.load_and_split()
# print(len(pages))

# def pdf_to_document(uploaded_file):
#     temp_dir = tempfile.TemporaryDirectory()
#     temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#     with open(temp_filepath, "wb") as f:
#         f.write(uploaded_file.getvalue())
#     # Load
#     loader = PyPDFLoader(temp_filepath)
#     pages = loader.load_and_split()
#     return pages

###### SPLIT ------------------------------------------------------------------
texts = text_splitter.split_documents(pages)
# all_texts = [doc.page_content for doc in texts]
# print(texts[0])

###### TOKENIZE  --------------------------------------------------------------
### (EMBEDDINGS)

### STORE (Vector Store)  --------------------------------------------------------
# if os.path.exists(vector_store_path):
#     # 기존 벡터 저장소 로드
#     vectorstore = FAISS.load_local(vector_store_path, embeddings)
#     print("기존 벡터 저장소를 로드했습니다.")
# else:
vectorstore = lc_faiss.from_documents(
    documents=texts,
    embedding=embedding_model,
)

res = faiss.StandardGpuResources()  # GPU 리소스 생성
gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
vectorstore.index = gpu_index

## 사용 : vectorstore.similarity_search(query)

cpu_index = faiss.index_gpu_to_cpu(vectorstore.index)
vectorstore.index = cpu_index
vectorstore.save_local(V_STORE_PATH)



# 저장된 인덱스 로드 (CPU 상태)
vectorstore = lc_faiss.load_local(V_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
res = faiss.StandardGpuResources()  # GPU 리소스 생성
# CPU → GPU 변환
gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
vectorstore.index = gpu_index
query = "example query text"
results = vectorstore.similarity_search(query, k=5)  # 상위 5개 결과 검색
for result in results:
    print(result.page_content)  # 검색된 문서 출력


# ---------------------------------------------------------------------
dimemsion = 384
res = faiss.StandardGpuResources()  # GPU 리소스 초기화
cpu_index = faiss.IndexFlatL2(dimemsion)
gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)

# FAISS 벡터 저장소 생성
vectorstore = lc_faiss(
    embedding_function=embedding_model,
    index=gpu_index,
    docstore=InMemoryDocstore({}),
    index_to_docstore_id={}
)
# pure_texts = [doc.page_content for doc in texts]
# metadatas = [doc.metadata for doc in texts]
# vectorstore.add_texts(texts=pure_texts, metadatas=metadatas)
vectorstore.add_documents(documents=texts)

cpu_index = faiss.index_gpu_to_cpu(vectorstore.index)
vectorstore.index = cpu_index
# vectorstore.save_local("faiss_index")
vectorstore.save_local(V_STORE_PATH)
# ------------------------------------------------------------------



# #------
# # 저장된 인덱스 로드
# vectorstore = FAISS.load_local("faiss_index", embedding_model)

# # 다시 GPU로 변환
# gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
# vectorstore.index = gpu_index

# # vectorstore = lc_faiss.from_documents(
# #     documents=texts,
# #     embedding=embedding_model,
# # )
# # db.index_to_docstore_id      # # 문서 저장소 ID 확인
# # db.docstore._dict            # 저장된 문서의 ID: Document 확인

# # 새로운 문서 추가
# # vectorstore.add_texts(all_texts)

# vectorstore.save_local(V_STORE_PATH)



# 저장된 인덱스 로드 (CPU 상태)
vectorstore = lc_faiss.load_local(V_STORE_PATH, embedding_model, allow_dangerous_deserialization=True)
gpu_index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
vectorstore.index = gpu_index

results = vectorstore.similarity_search(query, k=5)  # 상위 5개 결과 검색
for result in results:
    print(result.page_content)  # 검색된 문서 출력

# vectorstore = FAISS.load_local("faiss_index", embedding_model)
###### Retreve (QUERY) ------------------------------------------------------------------
question = "아내가 먹고 싶어하는 음식은 무엇인가요?"



retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm,
)

# 관련된 문서 찾기
docs = retriever_from_llm.get_relevant_documents(question)
print(docs)
print(len(docs))

qa_chain = RetrievalQA.from_chain_type(
    retriever = vectorstore.as_retriever(),
    llm = llm,
)

# 질문에 대한 답변 출력하기
answer = qa_chain.invoke({"query": question})
print(f"Answer:{answer['result']}")
