import streamlit as st
# from langchain_community.llms import CTransformers
# llm = CTransformers(
#     model="./llama-2-7b-chat.ggmlv3.q6_K.bin",
#     model_type="llama",
#     quqntize=True,
# )
from langchain_ollama import OllamaLLM
llm = OllamaLLM( model = "llama3.3:latest")

from langchain_ollama import ChatOllama
llm = ChatOllama(
    model = "llama3.3:latest",
    temperature = 0.8,
    num_predict = 256,
    # other params ...
)
messages = [
    ("system", "You are a helpful translator. Translate the user sentence to French."),
    ("human", "I love programming."),
]
llm.invoke(messages)
# content = "coding"
# result = llm.invoke('write poem about' + content )
# print(result)

# ----------------------------------------------------------------------
st.title('인공지능 시인')

content = st.text_input('시의 주제를 제시해주세요.')

if st.button('시 작성 요청하기'):
    with st.spinner('시 작성 중...'):
        # result = llm.invoke("write a poem about " + content + ": ")
        result = llm.invoke(content + "에 대한 시를 써주세요.")
        st.write(result)