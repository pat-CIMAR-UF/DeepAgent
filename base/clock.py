"""Current date and time, exposed as a LangChain tool so agents don't rely on memory."""

from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool


@tool("get_current_time")
def get_current_time(timezone: str = "") -> str:
    """Return the current date and time. Call this before any time-sensitive work.

    The model's own sense of "today" is frozen at its training cutoff, so use this
    whenever the request involves "latest", "recent", "today", "this week", "current",
    or any other relative date, and when a search result's freshness matters.

    Args:
        timezone: Optional IANA timezone name, e.g. "America/New_York" or "Asia/Shanghai".
            Leave empty for the machine's local timezone.
    """
    if timezone:
        try:
            now = datetime.now(ZoneInfo(timezone))
        except (ZoneInfoNotFoundError, ValueError):
            return f"Unknown timezone {timezone!r}. Use an IANA name like 'America/New_York'."
    else:
        now = datetime.now().astimezone()

    return (
        f"{now.strftime('%A, %d %B %Y, %H:%M:%S %Z (UTC%z)')}\n"
        f"ISO 8601: {now.isoformat()}\n"
        f"UTC: {now.astimezone(ZoneInfo('UTC')).strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )


def main():
    print(get_current_time.invoke({}))
    print()
    print(get_current_time.invoke({"timezone": "America/New_York"}))


if __name__ == "__main__":
    main()
