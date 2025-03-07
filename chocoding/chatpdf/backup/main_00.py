
import torch

# from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
# from langchain_community.vectorstores import Chroma

from langchain_ollama import ChatOllama
from langchain.retrievers import MultiQueryRetriever

from langchain.chains import RetrievalQA

pdf_path = "/workspace/git/Ulangchain/chocoding/chatpdf/aLuckyDay.pdf"
###### LOAD -------------------------------------------------------------------
# loader = PyPDFLoader("./aLuckyDay.pdf")
try:
    loader = PyPDFLoader(pdf_path)
except ValueError as e:
    print(f"PDF 파일 경로 오류: {e}")
    raise

###### SPLIT ------------------------------------------------------------------
# splitter 초기화
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
# 페이지 단위로 분할
pages = loader.load_and_split()
# print(len(pages))
texts = text_splitter.split_documents(pages)
# print(texts[0])

###### TOKENIZE  --------------------------------------------------------------
### (EMBEDDINGS)
# tictoken..
embedding_model = OllamaEmbeddings(
    model="llama3.3:latest",
    base_url="http://172.17.0.2:11434"
)

### STORE (Vector Store)  --------------------------------------------------------
db = Chroma.from_documents(texts, embedding_model)

'''
import chromadb
from chromadb.utils import embedding_functions
cl = chromadb.Client()
col_nms = cl.list_collections()
col_nm = col_nms[0]

langchain_collection = cl.get_or_create_collection(name=col_nm[0], embedding_function=embedding_functions.DefaultEmbeddingFunction())
langchain_collection
'''

###### Retreve (QUERY) ------------------------------------------------------------------
question = "아내가 먹고 싶어하는 음식은 무엇인가요?"

llm = ChatOllama(
    apiBase="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0,
    num_predict = 256,
)

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=db.as_retriever(),
    llm=llm,
)

# 관련된 문서 찾기
docs = retriever_from_llm.get_relevant_documents(question)
print(docs)
print(len(docs))

qa_chain = RetrievalQA.from_chain_type(
    retriever = db.as_retriever(),
    llm = llm,
)

# 질문에 대한 답변 출력하기
answer = qa_chain.invoke({"query": question})
print(f"Answer:{answer['result']}")
