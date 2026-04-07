from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
REPO_ROOT = PROJECT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from universal_cefr_classifier import DEFAULT_MODEL_NAME, classify_texts, load_classifier, top_prediction

VALID_CEFR_LEVELS = {"A1", "A2", "B1", "B2", "C1", "C2"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the UniversalCEFR classifier on a JSONL dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "merlin-extracted" / "merlin_interesting_data.jsonl",
        help="Path to the JSONL input file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "merlin_interesting_data_cefr_predictions.jsonl",
        help="Where to write JSONL predictions.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=PROJECT_DIR / "outputs" / "merlin_interesting_data_cefr_summary.json",
        help="Where to write the summary JSON file.",
    )
    parser.add_argument(
        "--text-field",
        default="text",
        help="Field to classify from each JSONL record, for example: text or candidate_translation.",
    )
    parser.add_argument(
        "--label-field",
        default="author_level",
        help="Optional CEFR label field used for summary accuracy, for example: author_level or requested_cefr.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for classifier inference.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length passed to the classifier.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of rows to process for a quick smoke test.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name.",
    )
    return parser.parse_args()


def iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            yield json.loads(stripped)


def score_batch(records: list[dict], classifier, text_field: str, batch_size: int, max_length: int) -> list[dict]:
    texts = [record[text_field] for record in records]
    results = classify_texts(classifier, texts, batch_size=batch_size, max_length=max_length)
    scored_records: list[dict] = []
    for record, scores in zip(records, results):
        best = top_prediction(scores)
        scored_record = {
            key: value for key, value in record.items() if key != "_row_index"
        }
        scored_record.update(
            {
                "input_row_index": record["_row_index"],
                "text_field": text_field,
                "text_chars": len(record[text_field]),
                "predicted_cefr": best["label"],
                "predicted_score": best["score"],
                "score_distribution": {item["label"]: item["score"] for item in scores},
            }
        )
        scored_records.append(
            scored_record
        )
    return scored_records


def build_summary(
    *,
    input_path: Path,
    output_path: Path,
    summary_output_path: Path,
    text_field: str,
    label_field: str,
    requested_limit: int | None,
    total_rows_seen: int,
    skipped_missing_text: int,
    skipped_invalid_label: int,
    scored_rows: int,
    predicted_counts: Counter,
    label_counts: Counter,
    exact_matches: int,
) -> dict:
    comparable_rows = sum(label_counts.values())
    accuracy = None
    if comparable_rows:
        accuracy = exact_matches / comparable_rows
    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "summary_output_path": str(summary_output_path),
        "text_field": text_field,
        "label_field": label_field,
        "limit": requested_limit,
        "rows_seen": total_rows_seen,
        "rows_scored": scored_rows,
        "skipped_missing_text": skipped_missing_text,
        "skipped_invalid_label_for_accuracy": skipped_invalid_label,
        "predicted_distribution": dict(predicted_counts),
        "label_distribution_scored": dict(label_counts),
        "exact_match_accuracy": accuracy,
    }


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)

    classifier = load_classifier(args.model)

    batch: list[dict] = []
    predicted_counts: Counter = Counter()
    label_counts: Counter = Counter()
    exact_matches = 0
    skipped_missing_text = 0
    skipped_invalid_label = 0
    total_rows_seen = 0
    scored_rows = 0

    with args.output.open("w", encoding="utf-8") as output_handle:
        for row_index, record in enumerate(iter_jsonl(args.input)):
            if args.limit is not None and total_rows_seen >= args.limit:
                break

            total_rows_seen += 1
            record["_row_index"] = row_index

            text = record.get(args.text_field)
            if not isinstance(text, str) or not text.strip():
                skipped_missing_text += 1
                continue

            batch.append(record)
            if len(batch) < args.batch_size:
                continue

            for scored_record in score_batch(batch, classifier, args.text_field, args.batch_size, args.max_length):
                predicted_counts[scored_record["predicted_cefr"]] += 1
                expected_label = scored_record.get(args.label_field)
                if expected_label in VALID_CEFR_LEVELS:
                    label_counts[expected_label] += 1
                    if expected_label == scored_record["predicted_cefr"]:
                        exact_matches += 1
                else:
                    skipped_invalid_label += 1
                output_handle.write(json.dumps(scored_record, ensure_ascii=False) + "\n")
                scored_rows += 1
            batch.clear()

        if batch:
            for scored_record in score_batch(batch, classifier, args.text_field, args.batch_size, args.max_length):
                predicted_counts[scored_record["predicted_cefr"]] += 1
                expected_label = scored_record.get(args.label_field)
                if expected_label in VALID_CEFR_LEVELS:
                    label_counts[expected_label] += 1
                    if expected_label == scored_record["predicted_cefr"]:
                        exact_matches += 1
                else:
                    skipped_invalid_label += 1
                output_handle.write(json.dumps(scored_record, ensure_ascii=False) + "\n")
                scored_rows += 1

    summary = build_summary(
        input_path=args.input,
        output_path=args.output,
        summary_output_path=args.summary_output,
        text_field=args.text_field,
        label_field=args.label_field,
        requested_limit=args.limit,
        total_rows_seen=total_rows_seen,
        skipped_missing_text=skipped_missing_text,
        skipped_invalid_label=skipped_invalid_label,
        scored_rows=scored_rows,
        predicted_counts=predicted_counts,
        label_counts=label_counts,
        exact_matches=exact_matches,
    )

    with args.summary_output.open("w", encoding="utf-8") as summary_handle:
        json.dump(summary, summary_handle, ensure_ascii=False, indent=2)
        summary_handle.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
