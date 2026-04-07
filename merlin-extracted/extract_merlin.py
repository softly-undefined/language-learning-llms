#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MERLIN_ROOT = REPO_ROOT / "merlin-text-v1.2" / "meta_ltext_THs"
OUTPUT_DIR = Path(__file__).resolve().parent
LANGUAGE_DIRS = ("czech", "german", "italian")
SECTION_SEPARATOR = re.compile(r"\n\s*-{10,}\s*\n")
FIELD_PATTERN = re.compile(r"^(.*?):\s*(.*)$")


def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def normalize_value(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


def extract_labeled_content(section: str, label: str) -> str | None:
    stripped = section.strip()
    if not stripped.startswith(label):
        return None
    content = stripped[len(label) :].strip()
    return content or None


def parse_metadata(section: str) -> dict[str, str | None]:
    metadata: dict[str, str | None] = {}
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line or line in {"METADATA", "General:", "Rating:"}:
            continue
        match = FIELD_PATTERN.match(line)
        if not match:
            continue
        key = match.group(1).strip().lower().replace("/", "_").replace(" ", "_")
        value = normalize_value(match.group(2))
        metadata[key] = value
    return metadata


def parse_record(path: Path) -> dict[str, object]:
    raw_text = normalize_newlines(path.read_text(encoding="utf-8"))
    sections = [section.strip() for section in SECTION_SEPARATOR.split(raw_text.strip())]

    metadata = parse_metadata(sections[0]) if sections else {}
    learner_text = None
    target_hypothesis_1 = None
    target_hypothesis_2 = None

    for section in sections[1:]:
        if section.startswith("Learner text:"):
            learner_text = extract_labeled_content(section, "Learner text:")
        elif section.startswith("Target hypothesis 1:"):
            target_hypothesis_1 = extract_labeled_content(section, "Target hypothesis 1:")
        elif section.startswith("Target hypothesis 2:"):
            target_hypothesis_2 = extract_labeled_content(section, "Target hypothesis 2:")

    record = {
        "author_id": metadata.get("author_id"),
        "language": metadata.get("test_language"),
        "test_level": metadata.get("cefr_level_of_test"),
        "author_level": metadata.get("overall_cefr_rating"),
        "mother_tongue": metadata.get("mother_tongue"),
        "task": metadata.get("task"),
        "grammatical_accuracy": metadata.get("grammatical_accuracy"),
        "orthography": metadata.get("orthography"),
        "vocabulary_range": metadata.get("vocabulary_range"),
        "vocabulary_control": metadata.get("vocabulary_control"),
        "coherence_cohesion": metadata.get("coherence_cohesion"),
        "sociolinguistic_appropriateness": metadata.get("sociolinguistic_appropriateness"),
        "age": metadata.get("age"),
        "gender": metadata.get("gender"),
        "text": learner_text,
        "has_translation": False,
        "translation": None,
        "has_target_hypothesis_1": target_hypothesis_1 is not None,
        "target_hypothesis_1": target_hypothesis_1,
        "has_target_hypothesis_2": target_hypothesis_2 is not None,
        "target_hypothesis_2": target_hypothesis_2,
        "source_file": str(path.relative_to(REPO_ROOT)),
    }
    return record


def collect_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for language_dir in LANGUAGE_DIRS:
        for path in sorted((MERLIN_ROOT / language_dir).glob("*.txt")):
            records.append(parse_record(path))
    return records


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, object]]) -> None:
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_summary(path: Path, records: list[dict[str, object]]) -> None:
    language_counts = Counter(record["language"] for record in records)
    level_counts = Counter(record["author_level"] for record in records)
    summary = {
        "records": len(records),
        "languages": dict(sorted(language_counts.items())),
        "author_levels": dict(sorted(level_counts.items())),
        "with_target_hypothesis_1": sum(
            1 for record in records if record["has_target_hypothesis_1"]
        ),
        "with_target_hypothesis_2": sum(
            1 for record in records if record["has_target_hypothesis_2"]
        ),
        "with_translation": sum(1 for record in records if record["has_translation"]),
    }
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    records = collect_records()
    if not records:
        raise SystemExit("No MERLIN records found.")

    jsonl_path = OUTPUT_DIR / "merlin_interesting_data.jsonl"
    csv_path = OUTPUT_DIR / "merlin_interesting_data.csv"
    summary_path = OUTPUT_DIR / "merlin_interesting_data_summary.json"

    write_jsonl(jsonl_path, records)
    write_csv(csv_path, records)
    write_summary(summary_path, records)

    print(f"Wrote {len(records)} records")
    print(f"- {jsonl_path}")
    print(f"- {csv_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
