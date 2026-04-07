from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TranslationRequest:
    text: str
    prompt: str
    model: str | None = None
    system_prompt: str = "You are a translation system."
    temperature: float = 0.0
    max_tokens: int = 1000

    @property
    def user_content(self) -> str:
        return self.prompt.format(text=self.text)

    def render_prompt(self) -> str:
        return f"{self.system_prompt}\n\n{self.user_content}".strip()

