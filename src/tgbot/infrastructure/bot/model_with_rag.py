import os

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from ..chroma import database

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

model = ChatOpenAI(base_url="http://localhost:8080/v1", temperature=0)


output_parser = StrOutputParser()


prompt = ChatPromptTemplate.from_template(
    """Answer the following question based only on the provided context. You must answer in Spanish:

<context>
{context}
</context>

Question: {input}"""
)


retriever = database.as_retriever(search_kwargs={"k": 10})
document_chain = create_stuff_documents_chain(model, prompt, output_parser=output_parser)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


def tfg_question(message: str) -> str:
    response = retrieval_chain.invoke({"input": message})
    return response["answer"]  # type: ignore
