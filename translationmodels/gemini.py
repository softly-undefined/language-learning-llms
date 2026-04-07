from __future__ import annotations

import os

import google.generativeai as genai

from .types import TranslationRequest


class GeminiTranslator:
    provider_name = "gemini"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is missing. Set GEMINI_API_KEY or pass api_key.")
        genai.configure(api_key=self.api_key)

    def translate(self, request: TranslationRequest | None = None, **kwargs) -> str | None:
        request = request or TranslationRequest(**kwargs)
        model_name = request.model
        if not model_name:
            raise ValueError("Gemini requests require a model name.")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            request.render_prompt(),
            generation_config=genai.types.GenerationConfig(
                temperature=request.temperature,
            ),
        )
        return getattr(response, "text", None)
