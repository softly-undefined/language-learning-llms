from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from translationmodels import TranslationHarness

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

CEFR_LEVELS = ("A1", "A2", "B1", "B2", "C1", "C2")
PROMPT_VERSION = "wmt_cefr_translation_v2_cefr_companion"
SYSTEM_PROMPT = (
    "You are a careful translation system. "
    "Produce translations that match the requested CEFR writing level in the target language."
)

# Source: Council of Europe, CEFR Companion Volume (2020), summary table p. 174 and
# self-assessment / written production descriptors pp. 177-178.
LEVEL_GUIDANCE = {
    "A1": '"familiar everyday expressions"; use very basic phrases, concrete needs, and highly simple sentence structure.',
    "A2": '"most immediate relevance"; use common everyday language, simple routine wording, and short linked sentences.',
    "B1": '"simple connected text"; use straightforward language on familiar topics, with brief reasons or explanations where useful.',
    "B2": '"clear, detailed text"; use more precise vocabulary, greater fluency, and some more complex sentence structure.',
    "C1": '"flexibly and effectively"; use well-structured, detailed language with strong control of register, cohesion, and nuance.',
    "C2": '"finer shades of meaning"; use highly precise, fluent, and stylistically polished language while staying faithful to the source.',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CEFR-targeted translation candidates for each row in a WMT CSV file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "wmt-data" / "en_zh_closest_500.csv",
        help="Path to the WMT CSV input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "en_zh_cefr_candidates.jsonl",
        help="Where to write JSONL candidate translations.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=SCRIPT_DIR / "outputs" / "en_zh_cefr_summary.json",
        help="Where to write the summary JSON file.",
    )
    parser.add_argument(
        "--provider",
        default="dummy",
        help="Translation provider name understood by TranslationHarness.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name. Required by some providers.",
    )
    parser.add_argument(
        "--source-field",
        default="english",
        help="CSV column containing the source text.",
    )
    parser.add_argument(
        "--reference-field",
        default=None,
        help="Optional CSV column containing a reference translation.",
    )
    parser.add_argument(
        "--target-language",
        required=True,
        help="Target language requested in the prompt, for example: German, Czech, or Italian.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        choices=CEFR_LEVELS,
        default=list(CEFR_LEVELS),
        help="One or more CEFR levels to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature forwarded to the translation provider.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=800,
        help="Maximum output tokens forwarded to the translation provider.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of WMT rows to process for a quick smoke test.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error record and continue when a generation request fails.",
    )
    return parser.parse_args()


def iter_csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        yield from csv.DictReader(handle)


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def maybe_tqdm(*, total: int | None, desc: str, unit: str, position: int):
    if tqdm is None:
        return None
    return tqdm(total=total, desc=desc, unit=unit, position=position, dynamic_ncols=True)


def build_prompt(level: str, target_language: str) -> str:
    guidance = LEVEL_GUIDANCE[level]
    return (
        "Translate the English source text into {target_language}.\n"
        "Write the result as if it were produced by a learner writing at CEFR {level}.\n"
        "{guidance}\n\n"
        "Requirements:\n"
        "- Output only the translation in {target_language}.\n"
        "- Keep the result plausible for CEFR {level}; do not sound like an expert translator unless the level is advanced.\n"
        "- Preserve the main meaning, but for lower levels prefer simpler wording over perfect coverage of every detail.\n"
        "- Do not explain your choices.\n"
        "- Do not include the CEFR label.\n\n"
        "English source:\n"
        "{{text}}"
    ).format(target_language=target_language, level=level, guidance=guidance)


def build_summary(
    *,
    args: argparse.Namespace,
    rows_seen: int,
    rows_processed: int,
    skipped_missing_source: int,
    records_written: int,
    success_counts: Counter,
    error_counts: Counter,
) -> dict:
    return {
        "input_path": str(args.input),
        "output_path": str(args.output),
        "summary_output_path": str(args.summary_output),
        "provider": args.provider,
        "model": args.model,
        "source_field": args.source_field,
        "reference_field": args.reference_field,
        "target_language": args.target_language,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "levels": list(args.levels),
        "prompt_version": PROMPT_VERSION,
        "limit": args.limit,
        "rows_seen": rows_seen,
        "rows_processed": rows_processed,
        "skipped_missing_source": skipped_missing_source,
        "records_written": records_written,
        "successful_generations_by_level": dict(success_counts),
        "errors_by_level": dict(error_counts),
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    harness = TranslationHarness()
    total_rows_available = count_csv_rows(args.input)
    total_rows_target = min(total_rows_available, args.limit) if args.limit is not None else total_rows_available
    total_requests_target = total_rows_target * len(args.levels)

    rows_seen = 0
    rows_processed = 0
    skipped_missing_source = 0
    records_written = 0
    success_counts: Counter = Counter()
    error_counts: Counter = Counter()
    rows_progress = maybe_tqdm(total=total_rows_target, desc="Rows", unit="row", position=0)
    requests_progress = maybe_tqdm(
        total=total_requests_target,
        desc="Generations",
        unit="req",
        position=1,
    )

    try:
        with args.output.open("w", encoding="utf-8") as output_handle:
            for row_index, row in enumerate(iter_csv_rows(args.input)):
                if args.limit is not None and rows_seen >= args.limit:
                    break

                rows_seen += 1
                source_text = (row.get(args.source_field) or "").strip()
                if rows_progress is not None:
                    rows_progress.set_postfix(
                        dataset=row.get("dataset") or "?",
                        seg_id=row.get("seg_id") or "?",
                        refresh=False,
                    )
                    rows_progress.update(1)
                if not source_text:
                    skipped_missing_source += 1
                    if requests_progress is not None:
                        requests_progress.update(len(args.levels))
                    continue

                rows_processed += 1
                reference_text = None
                if args.reference_field:
                    reference_text = (row.get(args.reference_field) or "").strip() or None

                for level in args.levels:
                    if requests_progress is not None:
                        requests_progress.set_postfix(
                            level=level,
                            ok=sum(success_counts.values()),
                            err=sum(error_counts.values()),
                            refresh=False,
                        )
                    record = {
                        "row_index": row_index,
                        "dataset": row.get("dataset"),
                        "doc_id": row.get("doc_id"),
                        "seg_id": row.get("seg_id"),
                        "requested_cefr": level,
                        "prompt_version": PROMPT_VERSION,
                        "provider": args.provider,
                        "model": args.model,
                        "source_field": args.source_field,
                        "reference_field": args.reference_field,
                        "target_language": args.target_language,
                        "temperature": args.temperature,
                        "max_tokens": args.max_tokens,
                        "source_text": source_text,
                        "reference_text": reference_text,
                    }
                    try:
                        record["candidate_translation"] = harness.translate(
                            provider=args.provider,
                            model=args.model,
                            text=source_text,
                            system_prompt=SYSTEM_PROMPT,
                            prompt=build_prompt(level, args.target_language),
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                        )
                        success_counts[level] += 1
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        record["error"] = f"{type(exc).__name__}: {exc}"
                        error_counts[level] += 1

                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    records_written += 1
                    if requests_progress is not None:
                        requests_progress.update(1)
    finally:
        if rows_progress is not None:
            rows_progress.close()
        if requests_progress is not None:
            requests_progress.close()

    summary = build_summary(
        args=args,
        rows_seen=rows_seen,
        rows_processed=rows_processed,
        skipped_missing_source=skipped_missing_source,
        records_written=records_written,
        success_counts=success_counts,
        error_counts=error_counts,
    )

    with args.summary_output.open("w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)
        summary_handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
