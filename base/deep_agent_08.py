import os
import select
from unittest import result
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_deepseek import ChatDeepSeek
from langchain.tools import tool
from deepagents import create_deep_agent
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt, RunnableConfig
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

llm = ChatDeepSeek(model="deepseek-v4-flash")

@tool
def delete_database(table_name: str):
    """
    High-risk action tool: deletes the given table!
    :param table_name: Name of the table to delete
    :return: Result of the operation
    """
    print(f"Called the delete_database tool. Deleted table {table_name}!!")
    return f"Deleted table {table_name}!"

# Delete file tool
@tool
def delete_file(file_name: str):
    """
    High-risk action tool: deletes the given file!
    :param file_name: Name of the file to delete
    :return: Result of the operation
    """
    print(f"Called the delete_file tool. Deleted file {file_name}!!")
    return f"Deleted file {file_name}!"

# Query table data tool
@tool
def select_database(table_name: str):
    """
    Query action tool: queries data from the given table!
    :param table_name: Name of the table to query
    :return: Query result
    """
    print(f"Called the select_database tool. Queried data from table {table_name}!!")
    return f"Queried data from table {table_name}!"

# Create deepagent, set HITL interaction with high-risk action

checkpointer = InMemorySaver()
thread_config: RunnableConfig = {"configurable": {"thread_id": "erdaye"}}

main_agent = create_deep_agent(
    model = llm,
    tools = [delete_database, delete_file, select_database],
    checkpointer = checkpointer,
    system_prompt = "Answer with English, call specific tools for the functions needed",
    interrupt_on={
        "delete_database": True,
        "delete_file": True,
        "select_database": False,
    },
)

result_1 = main_agent.invoke({
    "messages": [
        HumanMessage(content="First query the product table data! Then delete the user table, and finally delete the zhaoweifeng.txt file")
    ]
}, config=thread_config)
# for chunk in main_agent.stream(
#     {
#         "messages": [
#             {
#                 "role": "user",
#                 "content": "First query the product table data! Then delete the user table, and finally delete the zhaoweifeng.txt file",
#             }
#         ]
#     },
#     config=thread_config,
# ):
#     if not isinstance(chunk, dict):
#         continue
#     if "__interrupt__" in chunk:
#         print(chunk["__interrupt__"])
#         continue
#     for state in chunk.values():
#         if not isinstance(state, dict):
#             continue
#         for msg in state.get("messages") or []:
#             if isinstance(msg, (AIMessage, HumanMessage, ToolMessage)):
#                 msg.pretty_print()

interrupt = result_1['__interrupt__']

if interrupt:
    action_requests = interrupt[0].value['action_requests']
    print(f"Tools Require Review: {len(action_requests)}, they are: {[action['name'] for action in action_requests]}")

    decisions = []
    for action in action_requests:
        action_name = action['name']
        action_args = action['args']

        if action_name == "delete_database":
            decisions.append({"type": "reject"})
        elif action_name == "delete_file":
            decisions.append({"type": "approve"})

    
    result_2 = main_agent.invoke(
        Command(
            resume={
                "decisions": decisions
            }
        ), 
        config=thread_config
    )

    for msg in result_2.get("messages") or []:
        if isinstance(msg, (AIMessage, HumanMessage, ToolMessage)):
            msg.pretty_print()