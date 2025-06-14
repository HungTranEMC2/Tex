from functools import partial

from langgraph.graph import END, START, StateGraph  # noqa

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import ConfigSchema, FormInput

# from tex.tools.call_agents import create_handoff_tool
from tex.tools.call_model import call_model

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
    model: str = "gemini_chat"

    def __init__(
        self,
    ) -> None:
        self.workflow = StateGraph(
            FormInput,
            config_schema=ConfigSchema,
        )

        _call_model = partial(
            call_model,
            model_name=self.model,
        )  # noqa

        # Add nodes
        self.workflow.add_node("agent", _call_model)

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
