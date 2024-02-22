import os

from langchain_openai import ChatOpenAI

llm: ChatOpenAI = None  # type: ignore

if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

    llm = ChatOpenAI(base_url="http://localhost:8080/v1")
else:
    llm = ChatOpenAI(temperature=0)
