from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL_NAME = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"


@dataclass
class LoadedClassifier:
    tokenizer: AutoTokenizer
    model: AutoModelForSequenceClassification
    device: torch.device


def _resolve_device(device: int | None) -> torch.device:
    if device is not None:
        if device < 0:
            return torch.device("cpu")
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device}")
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _hub_cache_dir() -> Path:
    if "HUGGINGFACE_HUB_CACHE" in os.environ:
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"])
    if "HF_HOME" in os.environ:
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_model_cached(model_name: str) -> bool:
    repo_cache_dir = _hub_cache_dir() / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    return snapshots_dir.exists() and any(snapshots_dir.iterdir())


def _cached_model_path(model_name: str) -> str | None:
    repo_cache_dir = _hub_cache_dir() / f"models--{model_name.replace('/', '--')}"
    snapshots_dir = repo_cache_dir / "snapshots"
    if not snapshots_dir.exists():
        return None
    snapshot_dirs = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
    if not snapshot_dirs:
        return None
    return str(snapshot_dirs[-1])


def load_classifier(model_name: str = DEFAULT_MODEL_NAME, device: int | None = None):
    resolved_device = _resolve_device(device)
    local_only = _is_model_cached(model_name)
    model_source = _cached_model_path(model_name) or model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=local_only)
    model = AutoModelForSequenceClassification.from_pretrained(model_source, local_files_only=local_only)
    model.to(resolved_device)
    model.eval()
    return LoadedClassifier(tokenizer=tokenizer, model=model, device=resolved_device)


def _normalize_scores(raw_output) -> list[list[dict[str, float | str]]]:
    if not raw_output:
        return []
    if isinstance(raw_output[0], dict):
        raw_output = [raw_output]
    normalized: list[list[dict[str, float | str]]] = []
    for scores in raw_output:
        ordered = sorted(scores, key=lambda item: float(item["score"]), reverse=True)
        normalized.append(ordered)
    return normalized


def classify_texts(
    classifier,
    texts: str | Iterable[str],
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> list[list[dict[str, float | str]]]:
    if isinstance(texts, str):
        texts = [texts]
    text_list = list(texts)
    if not text_list:
        return []

    label_lookup = {
        int(index): label for index, label in classifier.model.config.id2label.items()
    }
    normalized: list[list[dict[str, float | str]]] = []

    with torch.no_grad():
        for start_index in range(0, len(text_list), batch_size):
            batch_texts = text_list[start_index : start_index + batch_size]
            encoded = classifier.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {
                key: value.to(classifier.device)
                for key, value in encoded.items()
            }
            logits = classifier.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1).cpu().tolist()
            raw_output = [
                [
                    {"label": label_lookup[index], "score": score}
                    for index, score in enumerate(row)
                ]
                for row in probabilities
            ]
            normalized.extend(_normalize_scores(raw_output))

    return normalized


def top_prediction(scores: list[dict[str, float | str]]) -> dict[str, float | str]:
    if not scores:
        raise ValueError("Expected at least one class score.")
    return max(scores, key=lambda item: float(item["score"]))
