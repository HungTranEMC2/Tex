from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, CompiledStateGraph, StateGraph  # noqa
from langgraph.prebuilt import create_react_agent

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import FormInput
from tex.tools.call_agents import create_handoff_tool


def select_form(state: FormInput):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # Return the form name.
        return last_message.tool_calls[0]["name"]
    return END


class FormAgent(BaseAgent):
    name: str = "form_agent"

    def __init__(
        self,
        model: BaseChatModel,
        form_agents_as_tools=[],  # Model to decide which forms to fill.
    ) -> None:
        self.workflow = StateGraph(FormInput)

        # Create agent to decide which forms to file or do something else..
        agent = create_react_agent(
            model=model,
            tools=[create_handoff_tool(tool) for tool in form_agents_as_tools],
        )

        # Add nodes

        self.workflow.add_node("agent", agent)

        # Add edges
        self.workflow.add_edge(START, "agent")
        self.workflow.add_conditional_edges(
            "agent",
            select_form,
            [END, "agent"],
        )  # noqa

        self.workflow = self.workflow.compile()

    def get(self) -> CompiledStateGraph:
        return self.workflow
