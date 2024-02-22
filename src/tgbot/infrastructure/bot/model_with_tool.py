from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.ddg_search.tool import DuckDuckGoSearchResults

from .llm import llm

# Fetches the latest version of this prompt
prompt = hub.pull("wfh/langsmith-agent-prompt:5d466cbc")


tools = [
    DuckDuckGoSearchResults(name="duck_duck_go"),  # General internet search using DuckDuckGo
]

llm_with_tools = llm.bind_functions(tools)

runnable_agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(x["intermediate_steps"]),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(
    agent=runnable_agent, tools=tools, handle_parsing_errors=True, verbose=True
)


def general_question(message: str) -> str:
    response = agent_executor.invoke({"input": message})

    return response["output"]
