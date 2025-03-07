from langchain_ollama import OllamaEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_LLM_embeddings(model:str, option='Ollama'):
    if option=="Ollama":
        return OllamaEmbeddings(
            base_url="http://172.17.0.2:11434",
            model=model
        )
    elif option=='HuggingFace':
        return HuggingFaceEmbeddings(
            model_name = model
        )
    else:
        raise ValueError(f"Unsupported model_name: {model}")

# embedding_model = load_LLM_embeddings("nomic-embed-text:latest")
# embedding_model = load_LLM_embeddings("all-minilm:l6-v2")
# embedding_model = load_LLM_embeddings("nlpai-lab/KoE5", "HuggingFace")
# dimension_size = len(embedding_model.embed_query("hello world"))