from __future__ import annotations

import os

import anthropic

from .types import TranslationRequest


class AnthropicTranslator:
    provider_name = "anthropic"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is missing. Set ANTHROPIC_API_KEY or pass api_key.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def translate(self, request: TranslationRequest | None = None, **kwargs) -> str | None:
        request = request or TranslationRequest(**kwargs)
        if not request.model:
            raise ValueError("Anthropic requests require a model name.")
        response = self.client.messages.create(
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system=request.system_prompt,
            messages=[{"role": "user", "content": [{"type": "text", "text": request.user_content}]}],
        )
        if not response.content:
            return None
        return response.content[0].text
