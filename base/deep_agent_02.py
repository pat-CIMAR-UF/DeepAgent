from deepagents import SubAgent, create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain_deepseek import ChatDeepSeek

from clock import get_current_time
from gemini import gemini_web_search

_ = load_dotenv(find_dotenv())


model = ChatDeepSeek(
    model="deepseek-v4-flash",
    reasoning_effort="high",
    extra_body={"thinking": {"type": "disabled"}},
)

research_subagent: SubAgent = {
    "name": "web-researcher",
    "model": "google_genai:gemini-3.6-flash",
    "tools": [gemini_web_search, get_current_time],
    "description": (
        "Answers questions that need current or external information by searching the "
        "web with Gemini. Delegate one self-contained research question at a time."
    ),
    "system_prompt": (
        "You are a web researcher. Use the `gemini_web_search` tool to answer the "
        "question you were given.\n\n"
        "- Call `get_current_time` first when the question involves recency, and put the "
        "real current date into your search queries instead of guessing it.\n"
        "- Break a broad question into several focused searches rather than one vague one.\n"
        "- Search again when the first result is incomplete or contradictory.\n"
        "- Report only what the search results support; say so when something is unknown.\n"
        "- Finish with a short answer followed by the source URLs you relied on."
    )
}

agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a research assistant. You have no web access yourself: delegate any "
        "question about current events, prices, companies, or anything else you cannot "
        "answer from memory to the `web-researcher` subagent via the `task` tool. "
        "Then synthesize its findings into a direct answer and keep the sources.\n\n"
        "Your training data is stale, so never assume you know today's date. Call "
        "`get_current_time` whenever the request mentions 'latest', 'recent', 'today', "
        "'current', or any other relative date, and pass the actual date into the "
        "research task you delegate so the subagent searches for the right day."
    ),
    tools=[get_current_time],
    subagents=[research_subagent],
)


def main():
    stream = agent.stream(
        {"messages": [
            {"role": "user", "content": "Report latest AI news"}
        ]}
    )

    for event in stream:
        '''
        4 types of events:
        1. Whether model is deciding to call tools
        2. Whether model is deciding to call subagents
        3. Whether a tool is called
        4. Whether the final answer
        '''
        for node_type, state in event.items():
            if not state or "messages" not in state:
                continue

            messages = state["messages"]
            last_message = messages[-1]

            if node_type == "model":
                if last_message.tool_calls:
                    tool_call = last_message.tool_calls[0]
                    if tool_call["name"] == "task":
                        print(f"---Calling Subagent: {tool_call['args']['subagent_type']}")
                    else:
                        print(
                            f"---Calling Tool: {tool_call['name']} "
                            f"with args: {tool_call['args']}"
                        )
                elif last_message.content:
                    print(f"============ Final Answer: ============\n{last_message.content}")

            elif node_type == "tools":
                content = last_message.content
                tool_result = content[:100] + "..." if len(content) > 100 else content
                print(f"---Tool Called: {last_message.name}\n------------Result: {tool_result}")




if __name__ == "__main__":
    main()
