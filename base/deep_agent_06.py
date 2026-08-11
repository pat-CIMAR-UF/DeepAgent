# 实验6： 兼容langgraph子智能体格式
import os
from langchain_core.messages import HumanMessage, AIMessage
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain_deepseek import ChatDeepSeek

from langgraph.graph import add_messages, StateGraph, START, END
from typing import TypedDict, Annotated
_ = load_dotenv(find_dotenv())

llm = ChatDeepSeek(model="deepseek-v4-flash")

class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str

def processing_node(state: State) -> State:
    print(f"Called sub node from graph, original name: {state}")
    return {"messages": [AIMessage(content=f"Sub Node DONE the Job!!! from last message: {state['messages'][-1].content}")], "name": "processing_node"}

workflow = StateGraph(State)

workflow.add_node("worker", processing_node)

workflow.add_edge(START, "worker")
workflow.add_edge("worker", END)

app = workflow.compile()

sub_agent: CompiledSubAgent = {
    "name": "graph_agent",
    "description": (
        "Handles all business-logic requests. Call this for any user ask about "
        "processing business logic, workflows, or simple tasks."
    ),
    "runnable": app,
}

main_agent = create_deep_agent(
    model=llm,
    tools=[],
    subagents=[sub_agent],
    system_prompt=(
        "You are a commander. For every user request, immediately call the `task` "
        "tool with subagent_type='graph_agent' and pass the user's request as the "
        "description. Do not use ls, glob, read_file, or write_file. Do not ask "
        "clarifying questions. Always delegate first."
    ),
)

for chunk in main_agent.stream(
    {"messages": [HumanMessage(content="请处理一个简单的业务逻辑！")]}
):
    print(f"\n{chunk}\n")
