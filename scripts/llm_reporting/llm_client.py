import json
import os
from typing import Dict, Optional

try:
    from openai import OpenAI  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None


def create_client() -> "OpenAI":
    if OpenAI is None:
        raise RuntimeError("openai is not installed")
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_chat(
    system_prompt: str,
    user_prompt: str,
    model: str,
    temperature: float = 0.2,
    client: Optional["OpenAI"] = None,
) -> str:
    if not model:
        raise ValueError("model is required")
    client = client or create_client()
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("LLM returned empty response")
    return content


def ensure_json_response(text: str) -> Dict[str, object]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("Response does not contain JSON")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError("Failed to parse JSON from response") from exc
