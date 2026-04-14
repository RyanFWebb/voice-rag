"""
llm.py — Ollama wrappers for text generation and embedding.

All HTTP calls go through here. Swap OLLAMA_BASE_URL in config.py
to point at a remote Ollama instance.
"""

import requests
from config import (
    OLLAMA_BASE_URL,
    OLLAMA_TIMEOUT,
    GEN_MODEL,
    EMBED_MODEL,
    EMBED_BATCH,
)


def check_ollama() -> list[str]:
    """
    Verify Ollama is running and return a list of available model names.
    Raises requests.ConnectionError if unreachable.
    """
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
    r.raise_for_status()
    return [m["name"] for m in r.json().get("models", [])]


def generate(prompt: str, model: str = GEN_MODEL) -> str:
    """Send a prompt to Ollama and return the response string."""
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    r.raise_for_status()
    return r.json()["response"]


def embed(texts: list[str] | str, model: str = EMBED_MODEL) -> list[list[float]]:
    """
    Embed one or more texts using Ollama.
    Automatically batches to EMBED_BATCH chunks per request.

    Returns a list of embedding vectors (one per input text).
    """
    if isinstance(texts, str):
        texts = [texts]

    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        r = requests.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": model, "input": batch},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
        all_embeddings.extend(r.json()["embeddings"])
    return all_embeddings
