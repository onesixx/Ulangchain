from rosie.log import logger
from datetime import datetime
current_time = datetime.now().strftime("%H:%M:%S")
logger.info(f"start ch22!!! at {current_time}")

import streamlit as st
from langchain_ollama import ChatOllama

chat_llm = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1
)

st.title("Bife Chatbot")
if "messages" not in st.session_state:
    st.session_state["messages"] =[{
        'role': 'assistant',
        'content': 'Hello, Sixx!!  how can I help you today??'
    }]
for msg in st.session_state['messages']:
    st.chat_message(msg['role']).write(msg['content'])


if query := st.chat_input():
    st.session_state['messages'].append({'role': 'user', 'content': query})
    st.chat_message('user').write(query)

    with st.spinner("잠만 기다려 ~"):
        response = chat_llm.invoke(query)
        answer = response.content

    st.session_state['messages'].append({'role': 'asistant', 'content': answer})
    st.chat_message('assistant').write(answer)

# user_input = st.chat_input("Type something...")
# if user_input:
#     st.session_state["messages"].append({'role': 'user', 'content': user_input})
#     st.chat_message('user').write(user_input)

