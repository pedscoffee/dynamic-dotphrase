"""
Ollama LLM communication — generalised for any prompt, not just SOAP notes.
"""

import json
import requests

OLLAMA_BASE_URL = "http://localhost:11434"


def check_ollama() -> tuple[bool, list[str]]:
    """Return (is_running, list_of_available_models)."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m["name"] for m in resp.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []


def generate_with_prompt(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 180,
) -> str:
    """Stream an LLM response from the local Ollama instance."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
    }

    full_text = ""
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    full_text += delta
    except Exception as exc:
        return f"Error communicating with Ollama: {exc}"

    return full_text


def stream_with_prompt(
    system_prompt: str,
    user_message: str,
    model: str,
    timeout: int = 180,
):
    """Stream an LLM response from the local Ollama instance, yielding chunks."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        "stream": True,
    }

    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            stream=True,
            timeout=timeout,
        ) as resp:
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        yield delta
    except Exception as exc:
        yield f"\nError communicating with Ollama: {exc}"
