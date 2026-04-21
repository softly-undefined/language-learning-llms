# For each of the cleaned_ files in each directory under data/, 
# this file samples 100 segments from the cleaned json and saves them to a new json with "sampled_" prefix. 
# each of the sampled jsons are saved in the sample/ directory

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
SAMPLE_DIR = Path(__file__).resolve().parent / "sample"
SAMPLES_PER_LEVEL = 50
CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")


def iter_cleaned_json_files(data_dir: Path) -> list[Path]:
    return sorted(data_dir.rglob("cleaned_*.json"))


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return entries


def group_entries_by_level(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    grouped = {level: [] for level in CEFR_LEVELS}

    for entry in entries:
        level = entry.get("cefr_level")
        if level in grouped:
            grouped[level].append(entry)

    return grouped


def get_insufficient_levels(grouped_entries: dict[str, list[dict[str, Any]]]) -> dict[str, int]:
    return {
        level: len(grouped_entries[level])
        for level in CEFR_LEVELS
        if len(grouped_entries[level]) < SAMPLES_PER_LEVEL
    }


def sample_entries(grouped_entries: dict[str, list[dict[str, Any]]], seed_key: str) -> list[dict[str, Any]]:
    sampled_entries: list[dict[str, Any]] = []

    for level in CEFR_LEVELS:
        rng = random.Random(f"{seed_key}:{level}")
        level_entries = grouped_entries[level]
        selected_indices = sorted(rng.sample(range(len(level_entries)), SAMPLES_PER_LEVEL))
        sampled_entries.extend(level_entries[index] for index in selected_indices)

    return sampled_entries


def output_path_for(cleaned_path: Path) -> Path:
    return SAMPLE_DIR / f"sampled_{cleaned_path.name}"


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_files = iter_cleaned_json_files(DATA_DIR)
    if not cleaned_files:
        raise FileNotFoundError(f"No cleaned JSON files found under {DATA_DIR}")

    written_names: set[str] = set()
    created_count = 0
    skipped_count = 0

    for cleaned_path in cleaned_files:
        output_path = output_path_for(cleaned_path)
        if output_path.name in written_names:
            raise ValueError(f"Duplicate output filename detected: {output_path.name}")

        entries = load_entries(cleaned_path)
        grouped_entries = group_entries_by_level(entries)
        insufficient_levels = get_insufficient_levels(grouped_entries)

        if insufficient_levels:
            if output_path.exists():
                output_path.unlink()

            skipped_count += 1
            print(
                f"SKIP {cleaned_path.relative_to(DATA_DIR.parent)}: "
                f"cannot sample {SAMPLES_PER_LEVEL} per CEFR level "
                f"({', '.join(f'{level}={count}' for level, count in insufficient_levels.items())})"
            )
            continue

        sampled_entries = sample_entries(grouped_entries, cleaned_path.relative_to(DATA_DIR).as_posix())

        output_path.write_text(
            json.dumps(sampled_entries, ensure_ascii=False, indent=4) + "\n",
            encoding="utf-8",
        )
        written_names.add(output_path.name)
        created_count += 1

        print(
            f"{cleaned_path.relative_to(DATA_DIR.parent)} -> "
            f"{output_path.relative_to(DATA_DIR.parent)} "
            f"({len(sampled_entries)} entries)"
        )

    print(f"Created {created_count} sampled files; skipped {skipped_count}.")


if __name__ == "__main__":
    main()
