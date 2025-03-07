from rosie.log import setup_logging
from rosie.log import logger
logger.info("Here we go!!! vvvvvvvvvvvvvvvvvvvvvvvv")

from util import *
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

import os
import streamlit as st
import tempfile
# Define the model name you want to use
logger.debug("Starting the script...")

embedding_model = load_LLM_embeddings("nomic-embed-text:latest")
db = Chroma(persist_directory="./chroma_db",
        embedding_function=embedding_model)


# retriever = db.as_retriever()
# retriever.invoke("Hello, how are you?")

llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)


#==============================================================================
st.title("Demo for FA")
st.write("---")
st.header("Ask about 'a lucky day'")
question = st.text_input("Enter your question here")

# Define the primary prompt template
prompt0 = PromptTemplate.from_template(
    """The retrieved context does not contain relevant information to answer the user's query.
       Please generate a response based on general knowledge while clearly indicating
       that no specific context was found.
       If needed, suggest alternative ways the user can refine their query
       or provide additional details for better results.

    #Question:
    {question}

    #Context:
    {context}

    #Answer:"""
)

prompt_primary = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in Korean.

    #Question:
    {question}

    #Context:
    {context}

    #Answer:"""
)

# Define the alternate prompt for when no relevant context is found
prompt_alternate = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
    No relevant context was retrieved to answer this question.
    Generate a response based on your general knowledge.
    Clearly state that no specific context was available and provide a meaningful answer.

    #Question:
    {question}

    #Context: No relevant context found

    #Answer:"""
)

# Create a function to check if the context is empty or irrelevant
def has_relevant_context(context):
    # Implement your logic here to determine if the context is useful
    return len(context) > 0

qa_chain = RetrievalQA.from_chain_type(
    retriever = db.as_retriever(),
    llm = llm
)

if st.button('Ask'):
    with st.spinner("Thinking..."):
        # question = "아내가 먹고 싶은 음식은?"
        # qry = db.similarity_search(question, k=3)
        # qry_and_scores = db.similarity_search_with_score(question, k=3)
        # logger.info(f"answer is from {qry[0].page_content}")

        context = db.similarity_search(question, k=3)  # Adjust based on actual retrieval logic
        if has_relevant_context(context):
            combined_query = f"{prompt_primary.template}"
        else:
            combined_query = f"{prompt_alternate.template}"

        answer = qa_chain.invoke({"query": combined_query})

        # answer = qa_chain.invoke({"query": question})
        st.write(answer['result'])