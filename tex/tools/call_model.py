from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import MessagesState


def call_model(
    model_with_tools: BaseChatModel,
    state: MessagesState,
):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}
