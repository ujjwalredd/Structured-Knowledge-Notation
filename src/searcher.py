"""
Web search module using DuckDuckGo (ddgs).

This module handles the first step of the pipeline: given a question,
search the internet and return the raw text results. Both JSON and SKN
extraction pipelines receive the same search results to ensure a fair
comparison.

No API key required.
"""

import logging
import time

from ddgs import DDGS

logger = logging.getLogger(__name__)


def search_web(
    query: str,
    num_results: int = 5,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Search the web using DuckDuckGo and return raw text results.

    Parameters
    ----------
    query : str
        The search query (typically the benchmark question).
    num_results : int
        Number of search results to fetch.
    max_retries : int
        Number of retry attempts on failure.
    base_delay : float
        Base delay in seconds for exponential backoff.

    Returns
    -------
    str
        Concatenated search result text (titles, snippets, URLs)
        as a single string.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Searching: %s (attempt %d)", query[:80], attempt)
            results = DDGS().text(query, max_results=num_results)
            break
        except Exception as exc:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Search failed (attempt %d/%d): %s. Retrying in %.1fs",
                attempt, max_retries, exc, wait,
            )
            if attempt == max_retries:
                logger.error("Search exhausted retries for: %s", query[:80])
                return ""
            time.sleep(wait)

    if not results:
        logger.warning("No search results for: %s", query[:80])
        return ""

    parts: list[str] = []
    for i, item in enumerate(results, 1):
        title = item.get("title", "")
        body = item.get("body", "")
        href = item.get("href", "")

        block = f"[Result {i}]\nTitle: {title}\nURL: {href}\nContent: {body}"
        parts.append(block)

    raw_text = "\n\n".join(parts)
    logger.info(
        "Search returned %d results (%d chars) for: %s",
        len(results), len(raw_text), query[:80],
    )
    return raw_text
