from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .llm import llm

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

chain = prompt | llm | output_parser


def general_question(message: str) -> str:
    response = chain.invoke({"input": message})

    return response
