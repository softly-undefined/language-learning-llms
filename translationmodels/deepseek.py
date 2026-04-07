from __future__ import annotations

import re

from langchain_ollama import ChatOllama

from .types import TranslationRequest


class DeepSeekTranslator:
    provider_name = "deepseek"

    def translate(self, request: TranslationRequest | None = None, **kwargs) -> str | None:
        request = request or TranslationRequest(**kwargs)
        model_name = request.model or "deepseek-r1:7b"
        client = ChatOllama(
            model=model_name,
            temperature=request.temperature,
            model_kwargs={"num_predict": request.max_tokens},
        )
        response = client.invoke(request.render_prompt())
        return re.sub(r"<think>.*?</think>\s*", "", response.content, flags=re.DOTALL).strip()
