This section generates CEFR-targeted candidate translations for the WMT English source text.

Current script:

- `translation-levels/generate_candidates.py`

What it does:

- Reads `wmt-data/en_zh_closest_500.csv` by default.
- Uses `translationmodels.TranslationHarness` for every generation request.
- Builds one request per row per CEFR level: `A1`, `A2`, `B1`, `B2`, `C1`, `C2`.
- Writes long-form JSONL output, which is easier to feed into later scoring and CEFR-classification steps.

Defaults:

- Source field: `english`
- Reference field: optional
- Provider: `dummy`

Example smoke test:

```bash
python translation-levels/generate_candidates.py \
  --provider dummy \
  --model dummy-cefr-v1 \
  --target-language German \
  --limit 1 \
  --output translation-levels/outputs/smoke_dummy_candidates.jsonl \
  --summary-output translation-levels/outputs/smoke_dummy_summary.json
```

Example real run shape:

```bash
python translation-levels/generate_candidates.py \
  --provider openai \
  --model gpt-4.1-mini \
  --target-language German \
  --output translation-levels/outputs/en_zh_cefr_candidates.jsonl \
  --summary-output translation-levels/outputs/en_zh_cefr_summary.json
```

Output schema:

- One JSON object per source-row and CEFR-level pair.
- Includes WMT row metadata, the requested CEFR label, the English source text, the optional reference text, and the generated candidate translation.
