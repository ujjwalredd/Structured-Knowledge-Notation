"""
Shared LLM client using Ollama for local inference.

Replaces the Anthropic API dependency. All modules import from here
instead of maintaining their own API clients.

Requires: ollama running locally (default: http://localhost:11434)
Install: brew install ollama && ollama pull qwen2.5:7b
"""

import logging
import time

import ollama

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "qwen2.5:7b"


def call_llm(
    system: str,
    user_message: str,
    model: str = DEFAULT_MODEL,
    json_mode: bool = False,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    """Call a local Ollama model.

    Parameters
    ----------
    system : str
        System prompt.
    user_message : str
        User message.
    model : str
        Ollama model name (e.g. "qwen2.5:7b", "llama3.1:8b").
    json_mode : bool
        If True, force JSON output format via Ollama's format parameter.
    max_retries : int
        Retry attempts on transient errors.
    base_delay : float
        Base delay for exponential backoff.

    Returns
    -------
    str
        Model response text.
    """
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_message},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "options": {"temperature": 0, "num_predict": 4096},
    }
    if json_mode:
        kwargs["format"] = "json"

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(**kwargs)
            return response["message"]["content"].strip()
        except ollama.ResponseError as exc:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Ollama error (attempt %d/%d): %s. Retrying in %.1fs",
                attempt, max_retries, exc, wait,
            )
            if attempt == max_retries:
                raise
            time.sleep(wait)
        except Exception as exc:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Unexpected error (attempt %d/%d): %s. Retrying in %.1fs",
                attempt, max_retries, exc, wait,
            )
            if attempt == max_retries:
                raise
            time.sleep(wait)

    raise RuntimeError("Exhausted retries without a response")
