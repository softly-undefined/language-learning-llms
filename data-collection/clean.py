# This file cleans the data in data/
#
# The cleaning methods include:
# - Remove duplicates
# - Remove entries with missing values
# - Remove prefixes which don't add anything to the segment's "text" field
# ex. [4] 20वीं शताब्दी ... -> 20वीं शताब्दी ..
# - Remove segments which are too short (less than 15 characters)
#
# Each cleaned json saves to the same path as the original, but with "cleaned_" prefix.
# also, saves a md file with a description of what has changed.
#
# Note: Hindi contains some latin characters at times, keeping them for now.

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent / "data"
MIN_TEXT_LENGTH = 15
MAX_REPORT_EXAMPLES = 5
REQUIRED_FIELDS = (
    "title",
    "lang",
    "source_name",
    "format",
    "category",
    "cefr_level",
    "license",
    "text",
)


@dataclass
class FileStats:
    source_path: Path
    cleaned_path: Path
    report_path: Path
    original_count: int = 0
    cleaned_count: int = 0
    duplicates_removed: int = 0
    missing_removed: int = 0
    short_removed: int = 0
    prefix_edits: int = 0
    duplicate_examples: list[str] = field(default_factory=list)
    missing_examples: list[str] = field(default_factory=list)
    prefix_examples: list[tuple[str, str]] = field(default_factory=list)
    short_examples: list[str] = field(default_factory=list)


def iter_source_json_files(data_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in data_dir.rglob("*.json")
        if not path.name.startswith(("cleaned_", "sampled_"))
    )


def load_entries(path: Path) -> list[dict[str, Any]]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return entries


def missing_fields(entry: Any) -> list[str]:
    if not isinstance(entry, dict):
        return ["<non-object entry>"]

    missing: list[str] = []
    for field_name in REQUIRED_FIELDS:
        if field_name not in entry:
            missing.append(field_name)
            continue

        value = entry[field_name]
        if value is None:
            missing.append(field_name)
        elif isinstance(value, str) and not value.strip():
            missing.append(field_name)

    if "text" in entry and not isinstance(entry["text"], str):
        missing.append("text")

    return missing


def is_reference_marker(token: str) -> bool:
    token = token.strip()
    if not token:
        return False

    if len(token) <= 12 and any(char.isdigit() for char in token):
        return all(char.isdigit() or char in " .,:;/-" for char in token)

    if len(token) <= 4 and token.isalpha():
        return True

    if re.fullmatch(r"[IVXLCDMivxlcdm]{1,8}", token):
        return True

    return False


def is_bare_reference_marker(token: str) -> bool:
    token = token.strip()
    if not token or not any(char.isdigit() for char in token):
        return False

    return all(char.isdigit() or char in " .,:;/-" for char in token)


def strip_non_content_prefixes(text: str) -> tuple[str, list[str]]:
    working = text
    removed_prefixes: list[str] = []

    while True:
        stripped = working.lstrip()

        bracket_match = re.match(r"^(\[([^\[\]]{1,40})\]|\(([^\(\)]{1,40})\))\s*", stripped)
        if bracket_match:
            marker = bracket_match.group(2) or bracket_match.group(3) or ""
            if is_reference_marker(marker):
                removed_prefixes.append(bracket_match.group(1))
                working = stripped[bracket_match.end() :]
                continue

        bare_match = re.match(r"^([^\s\]\)\-:\.]{1,12})([\]\)\-:\.])\s+", stripped)
        if bare_match and is_bare_reference_marker(bare_match.group(1)):
            removed_prefixes.append(bare_match.group(1) + bare_match.group(2))
            working = stripped[bare_match.end() :]
            continue

        break

    return working.strip(), removed_prefixes


def truncate(text: str, limit: int = 120) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def append_example(examples: list[Any], value: Any) -> None:
    if len(examples) < MAX_REPORT_EXAMPLES:
        examples.append(value)


def clean_entries(entries: list[dict[str, Any]], source_path: Path) -> tuple[list[dict[str, Any]], FileStats]:
    cleaned_path = source_path.with_name(f"cleaned_{source_path.name}")
    report_path = source_path.with_name(f"cleaned_{source_path.stem}.md")
    stats = FileStats(source_path=source_path, cleaned_path=cleaned_path, report_path=report_path)

    seen_texts: set[str] = set()
    cleaned_entries: list[dict[str, Any]] = []

    for index, entry in enumerate(entries, start=1):
        stats.original_count += 1

        missing = missing_fields(entry)
        if missing:
            stats.missing_removed += 1
            append_example(stats.missing_examples, f"Entry {index}: missing {', '.join(sorted(set(missing)))}")
            continue

        cleaned_entry = dict(entry)
        original_text = cleaned_entry["text"]
        cleaned_text, removed_prefixes = strip_non_content_prefixes(original_text)

        if removed_prefixes and cleaned_text != original_text:
            stats.prefix_edits += 1
            append_example(
                stats.prefix_examples,
                (truncate(original_text), truncate(cleaned_text)),
            )

        cleaned_entry["text"] = cleaned_text

        if len(cleaned_text) < MIN_TEXT_LENGTH:
            stats.short_removed += 1
            append_example(stats.short_examples, truncate(cleaned_text))
            continue

        if cleaned_text in seen_texts:
            stats.duplicates_removed += 1
            append_example(stats.duplicate_examples, truncate(cleaned_text))
            continue

        seen_texts.add(cleaned_text)
        cleaned_entries.append(cleaned_entry)

    stats.cleaned_count = len(cleaned_entries)
    return cleaned_entries, stats


def render_report(stats: FileStats) -> str:
    lines = [
        f"# Cleaning Report for `{stats.source_path.name}`",
        "",
        f"- Source file: `{stats.source_path.name}`",
        f"- Cleaned file: `{stats.cleaned_path.name}`",
        f"- Original entries: {stats.original_count}",
        f"- Cleaned entries: {stats.cleaned_count}",
        "",
        "## Summary",
        "",
        f"- Duplicate segments removed: {stats.duplicates_removed}",
        f"- Entries with missing values removed: {stats.missing_removed}",
        f"- Prefix cleanups applied: {stats.prefix_edits}",
        f"- Segments removed for being shorter than {MIN_TEXT_LENGTH} characters: {stats.short_removed}",
    ]

    if stats.prefix_examples:
        lines.extend(["", "## Prefix Cleanup Examples", ""])
        for before, after in stats.prefix_examples:
            lines.append(f'- "{before}" -> "{after}"')

    if stats.duplicate_examples:
        lines.extend(["", "## Duplicate Examples Removed", ""])
        for text in stats.duplicate_examples:
            lines.append(f'- "{text}"')

    if stats.missing_examples:
        lines.extend(["", "## Missing Value Examples", ""])
        for example in stats.missing_examples:
            lines.append(f"- {example}")

    if stats.short_examples:
        lines.extend(["", "## Short Segment Examples Removed", ""])
        for text in stats.short_examples:
            lines.append(f'- "{text}"')

    lines.append("")
    return "\n".join(lines)


def write_outputs(cleaned_entries: list[dict[str, Any]], stats: FileStats) -> None:
    stats.cleaned_path.write_text(
        json.dumps(cleaned_entries, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    stats.report_path.write_text(render_report(stats), encoding="utf-8")


def main() -> None:
    json_files = iter_source_json_files(DATA_DIR)
    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {DATA_DIR}")

    totals = {
        "files": 0,
        "original": 0,
        "cleaned": 0,
        "duplicates": 0,
        "missing": 0,
        "prefix_edits": 0,
        "short": 0,
    }

    for path in json_files:
        entries = load_entries(path)
        cleaned_entries, stats = clean_entries(entries, path)
        write_outputs(cleaned_entries, stats)

        totals["files"] += 1
        totals["original"] += stats.original_count
        totals["cleaned"] += stats.cleaned_count
        totals["duplicates"] += stats.duplicates_removed
        totals["missing"] += stats.missing_removed
        totals["prefix_edits"] += stats.prefix_edits
        totals["short"] += stats.short_removed

        print(
            f"{path.relative_to(DATA_DIR.parent)}: "
            f"{stats.original_count} -> {stats.cleaned_count} "
            f"(duplicates={stats.duplicates_removed}, missing={stats.missing_removed}, "
            f"prefix_edits={stats.prefix_edits}, short={stats.short_removed})"
        )

    print(
        "Processed {files} files: {original} -> {cleaned} "
        "(duplicates={duplicates}, missing={missing}, prefix_edits={prefix_edits}, short={short})".format(
            **totals
        )
    )


if __name__ == "__main__":
    main()
