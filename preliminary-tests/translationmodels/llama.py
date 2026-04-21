from __future__ import annotations

from langchain_ollama import ChatOllama

from .types import TranslationRequest


class LlamaTranslator:
    provider_name = "llama"

    def translate(self, request: TranslationRequest | None = None, **kwargs) -> str | None:
        request = request or TranslationRequest(**kwargs)
        model_name = request.model or "llama3.1"
        client = ChatOllama(
            model=model_name,
            temperature=request.temperature,
            model_kwargs={"num_predict": request.max_tokens},
        )
        response = client.invoke(request.render_prompt())
        return response.content
