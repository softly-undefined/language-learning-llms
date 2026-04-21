from importlib import import_module

from .harness import TranslationHarness
from .types import TranslationRequest

_PROVIDER_EXPORTS = {
    "AnthropicTranslator": ("anthropic", "AnthropicTranslator"),
    "DeepSeekTranslator": ("deepseek", "DeepSeekTranslator"),
    "DummyTranslator": ("dummy", "DummyTranslator"),
    "GeminiTranslator": ("gemini", "GeminiTranslator"),
    "LlamaTranslator": ("llama", "LlamaTranslator"),
    "OpenAITranslator": ("openai", "OpenAITranslator"),
}


def __getattr__(name: str):
    if name not in _PROVIDER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, class_name = _PROVIDER_EXPORTS[name]
    module = import_module(f".{module_name}", package=__name__)
    value = getattr(module, class_name)
    globals()[name] = value
    return value


__all__ = [
    "TranslationHarness",
    "TranslationRequest",
    *_PROVIDER_EXPORTS.keys(),
]
