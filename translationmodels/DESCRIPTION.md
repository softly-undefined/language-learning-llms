This directory contains a shared translation harness plus provider-specific API adapters.

Design:

- Prompt text is no longer hard-coded inside each provider file.
- A `TranslationRequest` carries `text`, `prompt`, `system_prompt`, `model`, `temperature`, and `max_tokens`.
- `TranslationHarness` accepts a provider name and request parameters, then dispatches to the correct provider adapter.
- Provider loading is lazy, so importing `translationmodels` does not require every provider SDK to be installed up front.
- Provider-specific files remain responsible only for API wiring and response extraction.
- A `dummy` provider is available for smoke tests that should exercise the harness without calling an external API.

Main entrypoints:

- `translationmodels/harness.py`
- `translationmodels/types.py`
- `translationmodels/openai.py`
- `translationmodels/anthropic.py`
- `translationmodels/gemini.py`
- `translationmodels/deepseek.py`
- `translationmodels/llama.py`

Example:

```python
from translationmodels import TranslationHarness

harness = TranslationHarness()

result = harness.translate(
    provider="openai",
    model="gpt-4.1-mini",
    text="學而時習之，不亦說乎？",
    system_prompt="You are a careful translation system.",
    prompt="Translate the following text to English. Output only the translation.\n\n{text}",
    temperature=0,
    max_tokens=300,
)
```

Smoke test:

```python
from translationmodels import TranslationHarness

harness = TranslationHarness()

result = harness.translate(
    provider="dummy",
    model="dummy-cefr-v1",
    text="Please translate this sentence.",
    system_prompt="You are a careful translation system.",
    prompt="Translate to Chinese as CEFR A2. Output only the translation.\n\n{text}",
)
```
