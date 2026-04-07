This folder should be where the classifier is located and tested.

The classifier is located here: UniversalCEFR/xlm-roberta-base-cefr-all-classifier

The following are instructions on how to use the classifier:

Here’s a clean, short Markdown guide you can drop into your repo 👇
# UniversalCEFR CEFR Classifier Usage

This project uses the Hugging Face model:

**UniversalCEFR/xlm-roberta-base-cefr-all-classifier**

to predict CEFR proficiency levels (A1–C2) from text.

---

## 🔧 Installation

```bash
pip install transformers torch
🚀 Basic Usage
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

model_name = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    top_k=None  # return all class scores
)

text = "Ich habe gestern mit meinen Freunden einen interessanten Film gesehen."

results = clf(text)

print(results)
print(model.config.id2label)
📊 Output Format
The model returns a list of CEFR levels with scores:
[
  {'label': 'B2', 'score': 0.72},
  {'label': 'C1', 'score': 0.18},
  ...
]
You can get the predicted level with:
pred = max(results[0], key=lambda x: x["score"])["label"]
🌍 Supported Languages
The model is multilingual (trained via UniversalCEFR), including:
German
Italian
Czech
English
other European languages
🧪 Recommended Use (for this project)
Run the classifier on:
MERLIN learner texts (text)
Corrected texts (text_target)
LLM-generated outputs
Then compare:
predicted CEFR distributions
differences across prompted proficiency levels
⚠️ Caveats
Model is trained on learner data, not LLM outputs
LLM text may be rated artificially high (C1/C2) due to fluency
Best used for relative comparison, not absolute scoring
💡 Tips
Truncate long texts (e.g., 512 tokens max)
Batch inputs for speed:
clf(list_of_texts, batch_size=8)
Store both:
top prediction
full score distribution
📚 Reference
Model: https://huggingface.co/UniversalCEFR/xlm-roberta-base-cefr-all-classifier

---

If you want next, I can give you:
- a **batch scoring script for MERLIN**
- or a **plot (CEFR distribution vs LLM levels)** that will immediately show results for your paper




