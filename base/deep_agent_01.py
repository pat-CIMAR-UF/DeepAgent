from deepagents import SubAgent, create_deep_agent
from dotenv import find_dotenv, load_dotenv
from langchain_deepseek import ChatDeepSeek

from gemini import gemini_web_search

_ = load_dotenv(find_dotenv())


model = ChatDeepSeek(
    model="deepseek-v4-flash",
    reasoning_effort="high",
    extra_body={"thinking": {"type": "enabled"}},
)

research_subagent: SubAgent = {
    "name": "web-researcher",
    "description": (
        "Answers questions that need current or external information by searching the "
        "web with Gemini. Delegate one self-contained research question at a time."
    ),
    "system_prompt": (
        "You are a web researcher. Use the `gemini_web_search` tool to answer the "
        "question you were given.\n\n"
        "- Break a broad question into several focused searches rather than one vague one.\n"
        "- Search again when the first result is incomplete or contradictory.\n"
        "- Report only what the search results support; say so when something is unknown.\n"
        "- Finish with a short answer followed by the source URLs you relied on."
    ),
    "tools": [gemini_web_search],
}

agent = create_deep_agent(
    model=model,
    system_prompt=(
        "You are a research assistant. You have no web access yourself: delegate any "
        "question about current events, prices, companies, or anything else you cannot "
        "answer from memory to the `web-researcher` subagent via the `task` tool. "
        "Then synthesize its findings into a direct answer and keep the sources."
    ),
    subagents=[research_subagent],
)


def main():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Do a sentiment analysis of the recent $AAOI earning call"}]}
    )
    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
