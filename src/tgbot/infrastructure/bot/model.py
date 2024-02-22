import os

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

model = ChatOpenAI(base_url="http://localhost:8080/v1", temperature=0)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un servicial bot de telegram que responde de forma concisa las preguntas de los usuarios.",
        ),
        ("user", "{input}"),
    ]
)


output_parser = StrOutputParser()

chain = prompt | model | output_parser


def general_question(message: str) -> str:
    response = chain.invoke({"input": message})

    return response
