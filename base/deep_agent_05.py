# Experiment 5: nest agents via CompiledSubAgent (SubAgent dict has no "subagents" key)
from deepagents import CompiledSubAgent, SubAgent, create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain_deepseek import ChatDeepSeek

_ = load_dotenv(find_dotenv())

llm = ChatDeepSeek(model="deepseek-v4-flash")

# 1. Bottom layer: Coder (only agent allowed to write code)
coder_config: SubAgent = {
    "name": "Coder",
    "description": "高级Python工程师，他是唯一有权限编写具体代码的人。",
    "system_prompt": "你是一个高级Python工程师。你的职责是接收具体的编码任务并实现它。",
    "tools": [],
}

# 2. Middle layer: CTO as its own deep agent that can delegate to Coder
cto_agent = create_deep_agent(
    model=llm,
    name="CTO",
    system_prompt="""你是技术总监。
    注意：你没有编写代码的权限！
    你的职责是：
    1. 分析 CEO 的需求。
    2. 设计技术方案。
    3. 调用 'Coder' 子代理来完成具体的代码编写工作。
    """,
    tools=[],
    subagents=[coder_config],
)

cto_config: CompiledSubAgent = {
    "name": "CTO",
    "description": "技术总监，负责将战略需求转化为技术任务并分配给工程师。指挥Coder写代码的！",
    "runnable": cto_agent,
}

# 3. Top layer: CEO delegates technical work to CTO
ceo_agent = create_deep_agent(
    model=llm,
    name="CEO",
    system_prompt="""你是CEO，负责公司战略决策。
    注意：你严禁直接编写代码或操作文件！
    你必须将所有技术相关的开发任务委派给 'CTO' 处理。
    你的工作是验收 CTO 提交的结果。
    """,
    subagents=[cto_config],
)


def main():
    stream = ceo_agent.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "请编写一个简单的Python程序，实现一个斐波那契数列的计算。"
                        "要求用Python实现,直接提供代码的字符串即可！！"
                    ),
                }
            ]
        }
    )
    for chunk in stream:
        print(f"\n{chunk}\n")


if __name__ == "__main__":
    main()
