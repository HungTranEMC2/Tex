# import asyncio

# from langgraph_sdk import get_client

# client = get_client(url="http://localhost:2024")


# async def main():
#     async for chunk in client.runs.stream(
#         None,  # Threadless run
#         "agent",  # Name of assistant. Defined in langgraph.json.
#         input={
#             "messages": [
#                 {
#                     "role": "human",
#                     "content": "What is LangGraph?",
#                 }
#             ],
#         },
#     ):
#         print(f"Receiving new event of type: {chunk.event}...")
#         print(chunk.data)
#         print("\n\n")


# asyncio.run(main())

from tex.agents.form import FormAgent
from tex.model import call_gemini_model
from tex.tools import create_handoff_tool

transfer_to_form1040 = create_handoff_tool(
    agent_name="form1040",
    description="Transfer to file form 1040.",
)
agent_obj = FoarmAgent(
    model=call_gemini_model(),
    tools=[transfer_to_form1040],
)
agent = agent_obj.get()
