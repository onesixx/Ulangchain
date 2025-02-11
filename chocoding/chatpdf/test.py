import requests

url = "http://172.17.0.2:11434/api/tags"
# url = "http://172.17.0.2:11434"
response = requests.get(url)

if response.status_code == 200:
    tags = response.json()

    names = [model['name'] for model in tags['models']]
    print("Available Models:")
    print(f"- {names} \n")
else:
    print(f"Failed to retrieve models. Status code: {response.status_code}")


import logging
import httpx

logging.basicConfig(level=logging.DEBUG)

client = httpx.Client()
response = client.get("http://172.17.0.2:11434")

print(response)



from langchain.chains import RetrievalQA
import httpx

def debug_httpx():
    with httpx.Client(event_hooks={"request": [lambda request: print(f"Request: {request.url}")],
                                   "response": [lambda response: print(f"Response: {response.status_code}")]}) as client:
        response = client.get("http://172.17.0.2:11434")
        print(response.text)

debug_httpx()






import logging
from langchain.chains import RetrievalQA
import httpx

logging.basicConfig(level=logging.DEBUG)

# httpx의 요청을 로깅
client = httpx.Client(event_hooks={"request": [lambda request: print(f"LangChain Request: {request.url}")],
                                   "response": [lambda response: print(f"LangChain Response: {response.status_code}")]})

# LangChain 실행
response = qa_chain.invoke({"query": question})
print(response)


import httpx
response = httpx.get("http://172.17.0.2:11434")
print(response.status_code, response.text)

#------
import logging
from langchain.chains import RetrievalQA
import httpx

logging.basicConfig(level=logging.DEBUG)

# httpx의 요청을 로깅
client = httpx.Client(event_hooks={"request": [lambda request: print(f"LangChain Request: {request.url}")],
                                   "response": [lambda response: print(f"LangChain Response: {response.status_code}")]})

# LangChain 실행
response = qa_chain.invoke({"query": question})
print(response)

