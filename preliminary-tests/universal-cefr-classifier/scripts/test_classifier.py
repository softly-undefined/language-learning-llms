from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from universal_cefr_classifier import DEFAULT_MODEL_NAME, classify_texts, load_classifier, top_prediction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-example sanity check against the UniversalCEFR classifier."
    )
    parser.add_argument(
        "--text",
        default="Ich habe gestern mit meinen Freunden einen interessanten Film gesehen.",
        help="Text to classify.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help="Hugging Face model name.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length passed to the classifier.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    classifier = load_classifier(args.model)
    scores = classify_texts(classifier, args.text, batch_size=1, max_length=args.max_length)[0]
    prediction = top_prediction(scores)

    print("Input text:")
    print(args.text)
    print()
    print("Top prediction:")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    print()
    print("All class scores:")
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

