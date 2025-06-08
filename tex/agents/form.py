from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph  # noqa
from langgraph.prebuilt import create_react_agent

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import ConfigSchema, FormInput
from tex.tools.call_agents import create_handoff_tool

# from tex.tools.call_model import call_model


async def select_form(state: FormInput):
    last_message = state["messages"][-1]

    if last_message.tool_calls:
        return "continue"
    #     # Return the form name.
    #     return last_message.tool_calls[0]["name"]

    return END


class FormAgent(BaseAgent):
    name: str = "form_agent"

    def __init__(
        self,
        model: BaseChatModel,
        tools=[],  # Model to decide which forms to fill.
    ) -> None:
        self.workflow = StateGraph(
            FormInput,
            config_schema=ConfigSchema,
        )

        # Create agent to decide which forms to file or do something else..
        # agent = create_react_agent(
        #     model=model,
        #     tools=tools,
        #     name="form",
        # )
        model.bind_tools(tools)
        # _call_model = partial(
        #     call_model,
        #     model=model,
        # )

        def call_model(
            # model: BaseChatModel,
            state: FormInput,
        ):
            messages = state["messages"]
            response = model.invoke(messages)
            return {"messages": [response]}

        # Add nodes
        self.workflow.add_node("agent", call_model)

        # Add edges
        self.workflow.add_edge(START, "agent")
        # self.workflow.add_conditional_edges(
        #     "agent",
        #     select_form,
        #     [END, "agent"],
        # )  # noqa

        self.workflow = self.workflow.compile()

    def get(self):
        return self.workflow
