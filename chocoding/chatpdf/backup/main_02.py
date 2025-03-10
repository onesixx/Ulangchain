# import torch
# if torch.cuda.is_available():
#     device = torch.device("cuda")  # Use GPU for computation
# else:
#     device = torch.device("cpu")  # Fallback to CPU
# print(f"device: {device}")


# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
#from langchain.vectorstores import Chroma
#from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain.retrievers import MultiQueryRetriever
from langchain.chains import RetrievalQA
import os
import streamlit as st
import tempfile

from langchain_community.vectorstores import FAISS

V_STORE_PATH = "./v_store"

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    # Load
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

st.title("ChatPDF with Ollama")
st.write("---")
st.write("Upload a PDF file to start chatting with it!")

# File uploader
# loader = PyPDFLoader("./aLuckyDay.pdf")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
# 로컬에 FAISS DB 인덱스가 이미 존재하는지 확인하고, 그렇다면 로드하여 vectorstore와 병합한 후 저장합니다.
DB_INDEX = "vector_store"


###### TOKENIZE  --------------------------------------------------------------
embedding_model = OllamaEmbeddings(
    model="llama3.3:latest",
    base_url="http://172.17.0.2:11434",
)


if os.path.exists(DB_INDEX):
# if uploaded_file is not None:
    ###### LOAD -------------------------------------------------------------------
    # Save the uploaded file to disk
    pages = pdf_to_document(uploaded_file)
    # pages = loader.load_and_split()

    ###### SPLIT ------------------------------------------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    # texts = text_splitter.split_documents(pages)
    docs = text_splitter.split_documents(pages)
    texts = [doc.page_content for doc in docs]

    ### STORE (Vector Store)  --------------------------------------------------------
    # db = Chroma.from_documents(texts, embedding_model, persist_directory=V_STORE_PATH)
    db = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
    )

    local_index = FAISS.load_local(DB_INDEX, embedding_model, allow_dangerous_deserialization=True)
    # local_index.merge_from(db)    #  id 중복
    # local_index.save_local(DB_INDEX)
else:
    #db.save_local(folder_path=DB_INDEX)
    db = FAISS.load_local(DB_INDEX, embedding_model, allow_dangerous_deserialization=True)
    st.write("File uploaded and store chroma successfully!")


###### Retreve (QUERY) ------------------------------------------------------------------
st.header("Ask your PDF")
question = st.text_input("Enter your question here")

# try:
#     db = Client()
#     db.load_from_directory(V_STORE_PATH)
# except Exception as e:
#     print(f"Error loading Chroma DB from directory: {e}")

if st.button('Ask'):
    with st.spinner("Thinking..."):
        qa_chain = RetrievalQA.from_chain_type(
            retriever = db.as_retriever(),
            llm = ChatOllama(
                    base_url="http://172.17.0.2:11434",
                    model = "llama3.3:latest",
                    temperature = 0,
                    num_predict = 256,
                )
        )
        # 질문에 대한 답변 출력하기
        answer = qa_chain.invoke({"query": question})
        print(answer)
        st.write(answer['result'])


