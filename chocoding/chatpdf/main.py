#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
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

texts = text_splitter.split_documents(pages)
# print(texts[0])

###### TOKENIZE  --------------------------------------------------------------
### (EMBEDDINGS)
# embedding_model = OllamaEmbeddings(
#     model = "llama3.3:latest",
# )
# tictoken..
embedding_model = OllamaEmbeddings(
    model="smollm2:135m",
    api_base="http://203.235.199.198:13434/"
)
### STORE (Vector Database)  --------------------------------------------------------
db = Chroma.from_documents(texts, embedding_model)

###### Retreve (QUERY) ------------------------------------------------------------------
question = "아내가 먹고 싶어하는 음식은 무엇인가요?"

llm = ChatOllama(
    apiBase="http://203.235.199.198:13434/",  # Replace with your Ollama API base URL if different
    model = "llama3.3",
    temperature = 0,
    num_predict = 256,
    #disable_streaming=False,
    # other params ...
)

# RetrievalQA 체인 초기화
retrieval_qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    chain_type="stuff"
)

retriever_from_llm = MultiQueryRetriever(
    llm_chain=retrieval_qa_chain,
    retriever=db.as_retriever(),
    #db = db,
    chain_type="stuff"
)

retriever_from_llm = MultiQueryRetriever(
    llm_chain=retrieval_qa_chain,
    retriever=db.as_retriever(),
    #db = db,
)

docs = retriever_from_llm.get_relevant_documents(query=question)
print(docs)
print(len(docs))

result = retriever_from_llm.retrieve(question)
print(f'Q: {question}')
print(f'A: {result}')