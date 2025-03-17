from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

#LLM 호출
#------
# model = ChatOpenAI(model="gpt-4o-mini")
#-----
# from langchain_cursor import ChatCursor  # Cursor AI 모델 사용
# model = ChatCursor(model="cursor-default")  # 기본 모델 사용
#------
model = ChatOllama(
    base_url="http://172.17.0.2:11434",
    model = "llama3.3:latest",
    temperature = 0.1,
    num_predict = 256,
)

#프롬프트 템플릿 설정
prompt = ChatPromptTemplate.from_template("tell me a short joke about {topic}")

#출력 파서 설정
output_parser = StrOutputParser()
#LCEL로 프롬프트템플릿-LLM-출력 파서 연결하기
chain = prompt | model | output_parser

#invoke함수로 chain 실행하기
chain.invoke({"topic": "ice cream"})

# =============================================================================
chain = prompt | model

#Chain의 stream()함수를 통해 스트리밍 기능 추가
for s in chain.stream({"topic": "bears"}):
    print(s.content, end="", flush=True)

# =============================================================================
prompt = ChatPromptTemplate.from_template("다음 한글 문장을 영어로 번역해줘 {sentence}")
chain = prompt | model

chain.batch([
    {"sentence": "그녀는 매일 아침 책을 읽습니다."},
    {"sentence": "오늘 날씨가 참 좋네요."},
    {"sentence": "저녁에 친구들과 영화를 볼 거예요."},
    {"sentence": "그 학생은 매우 성실하게 공부합니다."},
    {"sentence": "커피 한 잔이 지금 딱 필요해요."}
])
# =============================================================================
from langchain_core.runnables import RunnablePassthrough
RunnablePassthrough().invoke("안녕하세요")