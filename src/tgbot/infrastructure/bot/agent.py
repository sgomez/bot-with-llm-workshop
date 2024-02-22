import os

from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "OPENAI_API_KEY"

llm = ChatOpenAI(base_url="http://localhost:8080/v1", temperature=0)


def getWordLength(word: str) -> int:
    """Returns the length of a word."""
    print(f"{word=}")
    return len(word)


def getSentenceLength(sentence: str) -> str:
    """Returns the number of words in a sentence."""
    print(f"{sentence=}")
    return f"The sentence has {len(sentence.split(' '))} words"


tools = [
    Tool(
        name="Calculate sentence length",
        func=getSentenceLength,
        return_direct=True,
        description="use this tool to calculate the number of words in a sentence",
    ),
    Tool(
        name="Calculate word length",
        func=getWordLength,
        description="use this tool to calculate the length of a word",
        return_direct=True,
    ),
]


prompt = hub.pull("hwchase17/react")

llm_with_tools = llm.bind_tools(tools)


agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)  # type: ignore


def agent_question(message: str) -> str:
    response = agent_executor.invoke({"input": message})
    print(response)
    return response["output"]  # type: ignore
