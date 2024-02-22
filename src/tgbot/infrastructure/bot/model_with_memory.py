from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from .llm import llm

chat_message_history = SQLChatMessageHistory(
    session_id="test_session_id", connection_string="sqlite:///./data/sessions.db"
)


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
output_parser = StrOutputParser()


chain = prompt | llm | output_parser
chain_with_history = RunnableWithMessageHistory(
    chain,  # type: ignore
    lambda session_id: SQLChatMessageHistory(
        session_id=session_id, connection_string="sqlite:///sqlite.db"
    ),
    input_messages_key="question",
    history_messages_key="history",
)


def general_question_with_memory(chat_id: str, message: str) -> str:
    response: str = chain_with_history.invoke(
        {"question": message}, config={"configurable": {"session_id": chat_id}}
    )

    return response
