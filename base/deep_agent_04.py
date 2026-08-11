from deepagents import SubAgent, create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain_deepseek import ChatDeepSeek
import asyncio

_ = load_dotenv(find_dotenv())


model = ChatDeepSeek(
    model="deepseek-v4-flash"
)

weather_agent: SubAgent = {
    "name": "weather_helper",
    "description": "A agent that can get the weather information of a city",
    "system_prompt": "You are a weather agent. Always answer back 'Today is clear, 25 Celsius degrees in [city name]'",
    "tools": [],
}

math_agent: SubAgent = {
    "name": "math_helper", 
    "description": "Handles data calculation problems.",  # description for the main agent
    "system_prompt": "You are a careful math assistant who helps with arithmetic and calculation questions.",  # system prompt for the LLM
    "tools": [],
}

translation_agent: SubAgent = {
    "name": "translate_helper", 
    "description": "Assistant for Chinese–English translation.",  # description for the main agent
    "system_prompt": "You are a Chinese–English translation assistant. Translate Chinese to English, and English to Chinese.",  # system prompt for the LLM
    "tools": [],
}

def main():
    main_agent = create_deep_agent(
        model = model,
        tools = [],
        subagents = [weather_agent, math_agent, translation_agent],
        system_prompt = "You are capable assistant. You call correct agents to solve the user's problem, based on what the user asks. You MUST call agents to solve your problem. Do NOT solve the user request yourself!")

    async def user_stream(query: str) -> None:
        stream = main_agent.astream(
            {"messages": [{"role": "user", "content": query}]}
        )
        async for chunk in stream:
            # chunk -> {"model / tools " : {"messages":[{},{},{}]}}
            # model   |  {messages : []}
            for node_name , state in chunk.items():
                # 如果state是None,或者state没有messages我们就跳过！！
                if state is None or "messages" not in state: continue
                # 获取messages数据
                messages = state["messages"]
                if messages and isinstance(messages, list):
                    last_msg = messages[-1]
                    # 决定如何处理  node_name = model 1. 大模型决定调用工具 2. 大模型决定调用子agent 3.大模型返回结果了
                    # || node_name = tools  调用自己的工具，并获取返回结果
                    if node_name == "model":
                        # model = 》 返回的结果 =》 决定调用哪些
                        if last_msg.tool_calls:
                            # 决定调用子工具或者subAgent
                            for tool_call in last_msg.tool_calls:
                                if tool_call['name'] == 'task':
                                    # 决定调用某个subAgent
                                    print(f"【model】决定调用子智能体{tool_call['args']['subagent_type']}")
                                else:
                                    # 决定调用某个工具
                                    print(f"【model】决定调用子工具{tool_call['name']},传入的参数为：{tool_call['args']}")
                        elif last_msg.content:
                            # 模型返回最终结果
                            print(f"【model】返回最终结果：{last_msg.content}")
                    elif node_name == "tools":
                        # agent = > 调用自己的工具了，并获取了结果
                        name = last_msg.name
                        content = last_msg.content
                        print(f"【agent】调用了具体的工具{name},返回结果为：{content[:100]+'...'}")


    async def batch_run():
        await asyncio.gather(
            user_stream("What is the weather in Tokyo?"),
            user_stream("What is the result of 18345*42?"),
            user_stream("请将'我要给他打电话'翻译成英文！并且查询今天北京的天气信息！"),
        )

    asyncio.run(batch_run())


if __name__ == "__main__":
    main()

