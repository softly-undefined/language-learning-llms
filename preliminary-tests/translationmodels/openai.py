from __future__ import annotations

import os

from openai import OpenAI

from .types import TranslationRequest


class OpenAITranslator:
    provider_name = "openai"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key.")
        self.client = OpenAI(api_key=self.api_key)

    def translate(self, request: TranslationRequest | None = None, **kwargs) -> str | None:
        request = request or TranslationRequest(**kwargs)
        if not request.model:
            raise ValueError("OpenAI requests require a model name.")
        response = self.client.responses.create(
            model=request.model,
            input=[
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_content},
            ],
            max_output_tokens=request.max_tokens,
        )
        return response.output_text
