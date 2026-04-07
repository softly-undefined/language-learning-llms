# Universal CEFR Classifier

This folder contains a minimal runnable setup for testing the Hugging Face model:

`UniversalCEFR/xlm-roberta-base-cefr-all-classifier`

## Files

- `scripts/test_classifier.py`: sanity-check the model on one example sentence.
- `scripts/run_merlin_classifier.py`: run the classifier on JSONL input, including `merlin-extracted/merlin_interesting_data.jsonl`.
- `environment.yml`: conda environment definition.
- `requirements.txt`: pip alternative if you do not want conda.

## Setup

```bash
cd universal-cefr-classifier
conda env create -f environment.yml
conda activate universal-cefr-classifier
```

The first model run downloads the classifier from Hugging Face and caches it locally.

## Sanity Check

```bash
python scripts/test_classifier.py
```

Optional custom input:

```bash
python scripts/test_classifier.py --text "Ich lerne seit zwei Jahren Deutsch."
```

## Run On JSONL Data

Despite the historical filename, `run_merlin_classifier.py` now handles generic JSONL classification runs.

Default run on learner text:

```bash
python scripts/run_merlin_classifier.py
```

Quick smoke test:

```bash
python scripts/run_merlin_classifier.py --limit 20 --batch-size 4
```

Run on a different field, for example the corrected text:

```bash
python scripts/run_merlin_classifier.py --text-field target_hypothesis_1
```

Run on translation-level candidates and compare against `requested_cefr`:

```bash
python scripts/run_merlin_classifier.py \
  --input ../translation-levels/outputs/en_de_cefr_candidates_250.jsonl \
  --text-field candidate_translation \
  --label-field requested_cefr \
  --output outputs/en_de_cefr_candidates_250_classified.jsonl \
  --summary-output outputs/en_de_cefr_candidates_250_classified_summary.json
```

## Outputs

Outputs are written to:

- `outputs/merlin_interesting_data_cefr_predictions.jsonl`
- `outputs/merlin_interesting_data_cefr_summary.json`

Each JSONL row includes metadata, the top CEFR prediction, and the full score distribution.
