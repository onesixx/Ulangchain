from langchain_anthropic import ChatAnthropic
chat = ChatAnthropic(
    model_name ="claude-3-opus-20240229",
    anthropic_api_key="YOUR_API_KEY"
)
chat.invoke("안녕~ 너를 소개해줄래?")