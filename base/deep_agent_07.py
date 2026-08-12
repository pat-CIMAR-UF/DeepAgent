#  实验7： langchain create_agent 兼容 deepagents框架

from langchain_deepseek import ChatDeepSeek
from langchain_core.tools import tool
from langchain.agents import create_agent

from deepagents import CompiledSubAgent, create_deep_agent

from langchain_core.messages import HumanMessage, AIMessage

import os
from typing import TypedDict, Annotated
from dotenv import find_dotenv, load_dotenv
_ = load_dotenv(find_dotenv())

llm = ChatDeepSeek(model="deepseek-v4-flash")

@tool
def get_weather(city: str) -> str:
    """Look up the current weather for a city."""
    return f"{city}的天气是晴朗，25度"

agent = create_agent(
    model=llm,
    tools=[get_weather]
)

custom_subagent = CompiledSubAgent(
    name="subagent",
    description="子任务，可以调用天气工具，查询天气信息！",
    runnable=agent
)

deep_agent = create_deep_agent(
    model=llm,
    tools=[],
    system_prompt="你是一个智能助手,主要调用子代理实现功能，你只做任务分配,可以调用subagent实现功能！！",
    subagents=[custom_subagent]
)

result = deep_agent.invoke(
    {"messages": [HumanMessage(content="查询北京的天气！")]}
)

print(f"Final result: {result['messages'][-1].content}")