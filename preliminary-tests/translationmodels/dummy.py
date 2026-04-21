from __future__ import annotations

import re

from .types import TranslationRequest

CEFR_PATTERN = re.compile(r"\b(A1|A2|B1|B2|C1|C2)\b")


def _preview(text: str, limit: int = 96) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


class DummyTranslator:
    provider_name = "dummy"

    def translate(self, request: TranslationRequest | None = None, **kwargs) -> str:
        request = request or TranslationRequest(**kwargs)
        match = CEFR_PATTERN.search(request.render_prompt())
        level = match.group(1) if match else "UNKNOWN"
        model_name = request.model or "dummy-model"
        return f"[dummy translation level={level} model={model_name}] {_preview(request.text)}"
