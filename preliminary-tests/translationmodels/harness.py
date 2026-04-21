from __future__ import annotations

from dataclasses import asdict
from importlib import import_module

from .types import TranslationRequest


class TranslationHarness:
    def __init__(self):
        self._providers = {
            "anthropic": ("anthropic", "AnthropicTranslator"),
            "claude": ("anthropic", "AnthropicTranslator"),
            "deepseek": ("deepseek", "DeepSeekTranslator"),
            "dummy": ("dummy", "DummyTranslator"),
            "gemini": ("gemini", "GeminiTranslator"),
            "google": ("gemini", "GeminiTranslator"),
            "gpt": ("openai", "OpenAITranslator"),
            "llama": ("llama", "LlamaTranslator"),
            "mock": ("dummy", "DummyTranslator"),
            "ollama-llama": ("llama", "LlamaTranslator"),
            "openai": ("openai", "OpenAITranslator"),
        }

    def available_providers(self) -> list[str]:
        return sorted(self._providers.keys())

    def _load_translator_class(self, provider: str):
        module_name, class_name = self._providers[provider]
        module = import_module(f".{module_name}", package=__package__)
        return getattr(module, class_name)

    def get_translator(self, provider: str):
        key = provider.lower()
        if key not in self._providers:
            raise ValueError(f"Unsupported provider '{provider}'. Available: {', '.join(self.available_providers())}")
        return self._load_translator_class(key)()

    def build_request(
        self,
        *,
        text: str,
        prompt: str,
        model: str | None = None,
        system_prompt: str = "You are a translation system.",
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> TranslationRequest:
        return TranslationRequest(
            text=text,
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def translate(
        self,
        *,
        provider: str,
        text: str,
        prompt: str,
        model: str | None = None,
        system_prompt: str = "You are a translation system.",
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str | None:
        translator = self.get_translator(provider)
        request = self.build_request(
            text=text,
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return translator.translate(request)

    def request_to_dict(self, request: TranslationRequest) -> dict:
        return asdict(request)
