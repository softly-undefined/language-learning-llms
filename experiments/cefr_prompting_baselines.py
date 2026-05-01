#!/usr/bin/env python3

import os
import json
from dataclasses import asdict
from typing import List, Dict, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vllm import LLM, SamplingParams, EngineArgs


model_name = "unsloth/Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"

engine_args = EngineArgs(
    model=model_name,
    tokenizer=model_name,
    gpu_memory_utilization=0.2,
    max_model_len=16384,
)

llm = LLM(**asdict(engine_args))

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
    seed=42,
)


class CEFRClassifier:
    def __init__(
        self,
        model_name_or_path: str,
        device: str = "cuda:0",
        max_length: int = 512,
    ):
        self.device = torch.device(device)
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path
        ).to(self.device)

        self.model.eval()

        # Normalize id2label because HF configs sometimes load keys as strings.
        raw_id2label = getattr(self.model.config, "id2label", None)

        if raw_id2label:
            self.id2label = {int(k): v for k, v in raw_id2label.items()}
        else:
            self.id2label = {
                0: "A1",
                1: "A2",
                2: "B1",
                3: "B2",
                4: "C1",
                5: "C2",
            }

    @torch.inference_mode()
    def classify_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        results = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]

            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)

            pred_ids = torch.argmax(probs, dim=-1)

            for i, pred_id in enumerate(pred_ids):
                pred_id_int = int(pred_id.item())
                pred_label = self.id2label[pred_id_int]
                confidence = float(probs[i, pred_id_int].item())

                prob_dict = {
                    self.id2label[j]: float(probs[i, j].item())
                    for j in range(probs.shape[-1])
                }

                results.append({
                    "label": pred_label,
                    "confidence": confidence,
                    "probs": prob_dict,
                })

        return results

    def classify_one(self, text: str) -> Dict[str, Any]:
        return self.classify_batch([text], batch_size=1)[0]
    

cefr_classifier_name = "UniversalCEFR/xlm-roberta-base-cefr-all-classifier"
cefr_classifier = CEFRClassifier(
    model_name_or_path=cefr_classifier_name,
    device='cuda',
    max_length=512,
)


CEFR_DESCRIPTORS = {
    "A1": "very simple vocabulary, short sentences, direct wording, minimal grammar complexity",
    "A2": "simple everyday vocabulary, clear short sentences, basic connectors, limited grammar complexity",
    "B1": "clear intermediate English, familiar vocabulary, connected clauses, modest detail",
    "B2": "upper-intermediate English, broader vocabulary, fluent phrasing, varied sentence structures",
    "C1": "advanced English, precise vocabulary, nuanced phrasing, complex syntax, strong cohesion",
    "C2": "highly proficient English, sophisticated vocabulary, subtle nuance, polished near-native expression",
}


def apply_chat(messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate(llm, sampling_params, prompts: List[str]) -> List[str]:
    outputs = llm.generate(prompts, sampling_params)
    return [out.outputs[0].text.strip() for out in outputs]


def build_zero_shot_prompt(input_text: str, target_level: str) -> str:
    descriptor = CEFR_DESCRIPTORS[target_level]

    messages = [
        {"role": "system", "content": "Give the rewritten text ONLY."},
        {
            "role": "user",
            "content": (
                f"Rewrite the text below at CEFR {target_level} proficiency.\n"
                f"Target style: {descriptor}.\n\n"
                f"Text:\n{input_text}"
            ),
        },
    ]

    return apply_chat(messages)


def build_few_shot_prompt(
    input_text: str,
    target_level: str,
    few_shot_examples: List[Dict[str, str]],
) -> str:
    """
    few_shot_examples format:
    [
      {
        "source_level": "A1",
        "target_level": "B1",
        "source_text": "...",
        "target_text": "..."
      }
    ]
    """
    descriptor = CEFR_DESCRIPTORS[target_level]

    messages = [
        {"role": "system", "content": "Give the rewritten text ONLY."},
    ]

    for ex in few_shot_examples:
        ex_src = ex["source_text"]
        ex_tgt = ex["target_text"]
        ex_tgt_level = ex.get("target_level", target_level)

        messages.append({
            "role": "user",
            "content": (
                f"Rewrite the text below at CEFR {ex_tgt_level} proficiency.\n"
                f"Text:\n{ex_src}"
            ),
        })

        messages.append({
            "role": "assistant",
            "content": ex_tgt,
        })

    messages.append({
        "role": "user",
        "content": (
            f"Rewrite the text below at CEFR {target_level} proficiency.\n"
            f"Target style: {descriptor}.\n\n"
            f"Text:\n{input_text}"
        ),
    })

    return apply_chat(messages)


def load_cefr_samples(path: str) -> Dict[str, List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_few_shot_examples(
    samples_by_level: Dict[str, List[str]],
    src_level: str,
    tgt_level: str,
    n_shots: int = 3,
) -> List[Dict[str, str]]:
    """
    This assumes cefr_samples.json has unpaired level buckets:
      {
        "A1": ["...", "..."],
        "B1": ["...", "..."]
      }

    It pairs src and target examples by index only for demonstration.
    For stricter evaluation, use your aligned CSV pairs instead.
    """
    src_examples = samples_by_level[src_level]
    tgt_examples = samples_by_level[tgt_level]

    n = min(n_shots, len(src_examples), len(tgt_examples))

    examples = []

    for i in range(n):
        examples.append({
            "source_level": src_level,
            "target_level": tgt_level,
            "source_text": src_examples[i],
            "target_text": tgt_examples[i],
        })

    return examples


def run_baselines(
    transition: str = "B1_to_C2",
    samples_path: str = "data/cefr/cefr_samples.json",
    out_path: str = "outputs/cefr/baselines_A1_to_B1.jsonl",
    max_eval: int = 20,
    n_shots: int = 3,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    src_level, tgt_level = transition.split("_to_")

    samples_by_level = load_cefr_samples(samples_path)

    eval_inputs = samples_by_level[src_level][:max_eval]

    few_shot_examples = load_few_shot_examples(
        samples_by_level=samples_by_level,
        src_level=src_level,
        tgt_level=tgt_level,
        n_shots=n_shots,
    )

    zero_shot_prompts = []
    few_shot_prompts = []

    for input_text in eval_inputs:
        zero_shot_prompts.append(
            build_zero_shot_prompt(
                input_text=input_text,
                target_level=tgt_level,
            )
        )

        few_shot_prompts.append(
            build_few_shot_prompt(
                input_text=input_text,
                target_level=tgt_level,
                few_shot_examples=few_shot_examples,
            )
        )

    zero_shot_outputs = generate(llm, sampling_params, zero_shot_prompts)
    few_shot_outputs = generate(llm, sampling_params, few_shot_prompts)

    with open(out_path, "w", encoding="utf-8") as out_f:
        zero_results = {}
        few_results = {}
        for idx, input_text in enumerate(eval_inputs):
            output_cefr = cefr_classifier.classify_one(zero_shot_outputs[idx])
            output_pred = output_cefr["label"]
            if output_pred not in zero_results:
                zero_results[output_pred] = 0
            zero_results[output_pred] += 1
            print('ZERO OUTPUT:', zero_shot_outputs[idx])
            print('ZERO PRED:', output_pred)

            output_cefr = cefr_classifier.classify_one(few_shot_outputs[idx])
            output_pred = output_cefr["label"]
            if output_pred not in few_results:
                few_results[output_pred] = 0
            few_results[output_pred] += 1
            print('FEW OUTPUT:', few_shot_outputs[idx])
            print('FEW PRED:', output_pred)
            print('-----------')
            row = {
                "id": idx,
                "transition": transition,
                "source_cefr": src_level,
                "target_cefr": tgt_level,
                "input": input_text,

                "zero_shot_output": zero_shot_outputs[idx],
                "few_shot_output": few_shot_outputs[idx],

                "n_shots": n_shots,
                "few_shot_examples": few_shot_examples,
            }

            json.dump(row, out_f, ensure_ascii=False)
            out_f.write("\n")

    print(f"Wrote {out_path}")
    print(f"Transition: {transition}")
    print(f"Examples: {len(eval_inputs)}")
    print(f"Few-shot examples: {len(few_shot_examples)}")
    print('Zero results:', zero_results)
    print('Few results:', few_results)


if __name__ == "__main__":
    run_baselines(
        transition="A2_to_C1",
        samples_path="data/cefr/cefr_samples.json",
        out_path="outputs/cefr/baselines_B1_to_C1.jsonl",
        max_eval=20,
        n_shots=3,
    )
