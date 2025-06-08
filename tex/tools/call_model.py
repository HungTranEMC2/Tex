from langchain_core.runnables import RunnableConfig

from tex.agents.schemas import FormInput
from tex.model import ModelFactory


def call_model(
    state: FormInput,
    config: RunnableConfig,
):
    """
    Follow below script to get specific on run (https://langchain-ai.github.io/langgraph/how-tos/graph-api/#add-runtime-configuration). # noqa
        MODELS = {
            "anthropic": init_chat_model("anthropic:claude-3-5-haiku-latest"),
            "openai": init_chat_model("openai:gpt-4.1-mini"),
        }

        def call_model(state: MessagesState, config: RunnableConfig):
            model = config["configurable"].get("model", "anthropic")
            model = MODELS[model]
            response = model.invoke(state["messages"])
            return {"messages": [response]}

    """
    messages = state["messages"]

    # Get model
    model = ModelFactory.get(config["model"])
    response = model.invoke(messages)
    return {"messages": [response]}
