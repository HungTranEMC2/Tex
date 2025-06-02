from langchain_core.language_models.chat_models import BaseChatModel

# from langgraph.graph import MessagesState
from tex.agents.schemas import FormInput


def call_fill_model(
    form_name: str,
    line: str,
    model_with_tools: BaseChatModel,
    state: FormInput,
):
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response], "forms": state["forms"][form_name].update(response)}
