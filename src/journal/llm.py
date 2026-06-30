import json
import re
from typing import Protocol

import ollama

from .config import LLM_MODEL


class LLM(Protocol):
    def complete_json(self, system: str, user: str) -> dict | None: ...


_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    m = _FENCE_RE.match(text)
    if m:
        text = m.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            return None
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return None
    if not isinstance(parsed, dict):
        return None
    return parsed


class OllamaLLM:
    def __init__(self, model: str = LLM_MODEL, client=None, max_retries: int = 3):
        self.model = model
        self.max_retries = max_retries
        self._client = client or ollama.Client()

    def complete_json(self, system: str, user: str) -> dict | None:
        for _ in range(self.max_retries):
            resp = self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                format="json",
            )
            content = resp["message"]["content"]
            parsed = _extract_json(content)
            if parsed is not None:
                return parsed
        return None
