import warnings
warnings.filterwarnings("ignore")

from langchain_ollama import ChatOllama
reasoning_llm = ChatOllama(
    model = "deepseek-r1:7b",
    stop = ["</think>"],
    base_url="http://172.17.0.2:11434"
)
answer_llm = ChatOllama(
    model = "exaone3.5:latest",
    temperature=0,
    base_url="http://172.17.0.2:11434"
)




from typing import Annotated, List, TypedDict, Literal
from langgraph.graph.message import add_messages
from langchain_core.documents import Document

# define RAG status
class RAGState(TypedDict):
    query: str                               # user question
    thinking: str                            # area for reasoning by LLM
    documents: List[Document]                # retrieved document
    answer: str                              # final answer
    message: Annotated[List, add_messages]   # all process
    mode: str                                # simple / complex needed to doc


# ============  load doc ============
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

FILE_PATH = "https://arxiv.org/pdf/2408.09869"

loader = DoclingLoader(
    file_path=FILE_PATH,
    export_type=ExportType.MARKDOWN
)

docs = loader.load()

# ttt
# from IPython.display import Markdown
# len(docs)
# type(docs[0])
# Markdown(docs[0].page_content)

# ============ split to chunk ============
from langchain_text_splitters import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Headers_3")
    ]
)
splits = [split for doc in docs for split in splitter.split_text(doc.page_content)]


# ttt
# for d in splits[:3]:
#     print(f'- {d.page_content=}')
# print("...")




# ============ vector DB (Chroma, faiss, Milvus..) ============
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model= "bge-m3:latest",
    base_url="http://172.17.0.2:11434"
)

from langchain_qdrant import QdrantVectorStore
from langchain_qdrant import RetrievalMode

vector_store = QdrantVectorStore.from_documents(
    documents = splits,
    embedding = embeddings,
    path = ":memory:",
    # url = "http://172.17.0.5:6333",
    collection_name = "rag_collection_allAI",
    retrieval_mode = RetrievalMode.DENSE # HNSW
)
retriever = vector_store.as_retriever(search_kwargs={'k':10})


# ============ re-ranking ============
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

model = HuggingFaceCrossEncoder(model_name = "BAAI/bge-reranker-base")

compressor = CrossEncoderReranker(model=model, top_n=5)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
    #top_k=5,
    # cross_encoder_tokenizer="all-MiniLM-L6-v2",
    # cross_encoder_model=model
)

