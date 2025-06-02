from functools import partial

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, CompiledStateGraph, StateGraph  # noqa

from tex.agents.base_agent import BaseAgent
from tex.agents.schemas import FormInput
from tex.data.utils import get_form_lines
from tex.tools import call_fill_model, retrieve_instructions  # noqa


def should_fill_form(state: FormInput):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        # Return the form name.
        return last_message.tool_calls[0]["name"]
    return END


class Form1040Agent(BaseAgent):
    name: str = "f1040"

    def __init__(
        self,
        model: BaseChatModel,
        year: int,
    ) -> None:
        self.workflow = StateGraph(FormInput)

        # Get lines in form 1040
        lines = get_form_lines(
            year=year,
            form_name=self.name,
        )

        # Add tools to model
        model.bind_tools([retrieve_instructions])

        # Add nodes and edges representing lines in form.
        prev_line = START
        for idx, line in enumerate(lines):
            self.workflow.add_node(
                line["name"],
                partial(
                    call_fill_model,
                    form_name=self.name,
                    line=line["context"],
                    model_with_tools=model,
                ),
            )
            self.workflow.add_edge(prev_line, line["name"])
            if idx > 0:
                prev_line = line["name"]

        self.workflow.add_edge(prev_line, END)
        self.workflow = self.workflow.compile()

    def get(self) -> CompiledStateGraph:
        return self.workflow
