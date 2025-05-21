from typing import Any, Callable, Dict, TypedDict, Union

from langgraph.graph import END, START, CompiledStateGraph, StateGraph
from langgraph.types import Command

from tex.workflows.base_workflow import BaseWorkflow


class Form1040Input(TypedDict):
    lines: Dict[
        str, Dict[str, Any]
    ]  # {line: {line_number: 0, context: "Line 1: Total amount on W-2 Form(s)", result: 100.0}}


class Form1040Workflow(BaseWorkflow):

    workflow: CompiledStateGraph

    @classmethod
    def build(
        cls,
        entry_point_key: str,
        finish_point_key: str,
        nodes: Dict[str, Dict[str, Union[str, Callable]]],
    ) -> None:
        workflow = StateGraph(Form1040Input)
        workflow.set_entry_point(START, entry_point_key)
        workflow.set_finish_point(END, finish_point_key)

        # Add nodes and edges in sequence order as specified in lines.
        for node_name, metadata in nodes.items():
            workflow.add_node(key=node_name, node=metadata["node"])
            workflow.add_edge(
                prev_key=metadata["prev_key"],
                end_key=metadata["key"],
            )

        Form1040Workflow.workflow = workflow.compile()

    @classmethod
    def get(cls) -> CompiledStateGraph:
        return Form1040Workflow.workflow
