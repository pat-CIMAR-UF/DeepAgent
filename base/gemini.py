"""Gemini-backed web search, exposed as a LangChain tool for subagent use."""

import os

from dotenv import find_dotenv, load_dotenv
from google import genai
from google.genai import types
from langchain_core.tools import tool

_ = load_dotenv(find_dotenv())

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@tool("gemini_web_search")
def gemini_web_search(query: str) -> str:
    """Search the web with Google and return a grounded answer with its sources.

    Use this for anything that depends on current or external information:
    news, prices, earnings, docs, people, products.

    Args:
        query: A self-contained natural language question. Include every detail
            needed to answer it, since no prior conversation is passed along.
    """
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=query,
        config=types.GenerateContentConfig(
            # THIS IS WHAT ENABLES WEB SEARCH:
            tools=[{"google_search": {}}],
        ),
    )

    parts = [response.text or "(no answer returned)"]

    candidates = response.candidates
    metadata = candidates[0].grounding_metadata if candidates else None
    if metadata:
        if metadata.web_search_queries:
            parts.append("Search queries used: " + ", ".join(metadata.web_search_queries))

        if metadata.grounding_chunks:
            sources = [
                f"  [{i + 1}] {chunk.web.title}: {chunk.web.uri}"
                for i, chunk in enumerate(metadata.grounding_chunks)
                if chunk.web
            ]
            if sources:
                parts.append("Sources:\n" + "\n".join(sources))

    return "\n\n".join(parts)


def main():
    print(gemini_web_search.invoke({"query": "What did SNDK report in its latest earnings?"}))


if __name__ == "__main__":
    main()
