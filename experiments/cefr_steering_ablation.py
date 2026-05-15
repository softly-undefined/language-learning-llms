#!/usr/bin/env python3
"""
CEFR Steering Effectiveness Ablation

Tests each layer individually by actually generating text with single-layer
steering, then measuring CEFR classifier accuracy. This directly measures
which layers influence generation behavior, not just representation quality.

Usage:
    python cefr_steering_ablation.py --transitions B2_to_C1 B1_to_C1 A1_to_C1 --num_samples 10
    python cefr_steering_ablation.py --transitions B2_to_C1 --num_samples 15 --alpha 0.4
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cefr_experiment_run import (
    CEFR_ORDER, CEFR_TO_ID, CEFR_DESCRIPTORS,
    CEFRClassifier,
    forward, get_rep_directions,
    get_direction, get_overshoot_direction, project_out_overshoot,
)
from control_block import WrappedReadingVecModel


def run_single_layer_steering(
    model, tokenizer, cefr_classifier, wrapped_model,
    layer_id, direction, alpha, layer_weight,
    test_inputs, target_level, sign=1,
):
    wrapped_model.unwrap()
    wrapped_model.wrap_block([layer_id], block_name="decoder_block")

    activation = {
        layer_id: torch.tensor(
            alpha * layer_weight * direction * sign
        ).to(model.device)
    }
    wrapped_model.set_controller([layer_id], activation, masks=1, normalize=False)

    predictions = []
    for inp in test_inputs:
        user_prompt = (
            f"Rewrite the text below at CEFR {target_level} proficiency. "
            f"{CEFR_DESCRIPTORS[target_level]}\n"
            f"Text: {inp}"
        )
        input_messages = [
            {"role": "system", "content": "Give an output ONLY. No explanation."},
            {"role": "user", "content": user_prompt},
        ]
        try:
            source_input_prompt = tokenizer.apply_chat_template(
                input_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            source_input_prompt = tokenizer.apply_chat_template(
                input_messages, tokenize=False, add_generation_prompt=True,
            )

        encoded_inputs = tokenizer([source_input_prompt], return_tensors='pt')

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = wrapped_model.generate(
                    **encoded_inputs.to(model.device),
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                ).detach().cpu()

                text = tokenizer.decode(outputs[0])
                text = text.replace(source_input_prompt, "")
                text = text.lstrip('<|begin_of_text|>').rstrip('<|eot_id|>').strip()
                if 'assistant<|end_header_id|>' in text:
                    text = text[text.find('assistant<|end_header_id|>'):].replace(
                        'assistant<|end_header_id|>', ''
                    ).strip()

        result = cefr_classifier.classify_one(text)
        predictions.append(result["label"])

    wrapped_model.reset()
    wrapped_model.unwrap()
    return predictions


def run_baseline(
    model, tokenizer, cefr_classifier,
    test_inputs, target_level,
):
    predictions = []
    for inp in test_inputs:
        user_prompt = (
            f"Rewrite the text below at CEFR {target_level} proficiency. "
            f"{CEFR_DESCRIPTORS[target_level]}\n"
            f"Text: {inp}"
        )
        input_messages = [
            {"role": "system", "content": "Give an output ONLY. No explanation."},
            {"role": "user", "content": user_prompt},
        ]
        try:
            source_input_prompt = tokenizer.apply_chat_template(
                input_messages, tokenize=False,
                add_generation_prompt=True, enable_thinking=False,
            )
        except TypeError:
            source_input_prompt = tokenizer.apply_chat_template(
                input_messages, tokenize=False, add_generation_prompt=True,
            )

        encoded_inputs = tokenizer([source_input_prompt], return_tensors='pt')

        with torch.no_grad():
            with torch.autocast(device_type="cuda", enabled=True):
                outputs = model.generate(
                    **encoded_inputs.to(model.device),
                    max_new_tokens=256,
                    do_sample=False,
                    use_cache=True,
                ).detach().cpu()

                text = tokenizer.decode(outputs[0])
                text = text.replace(source_input_prompt, "")
                text = text.lstrip('<|begin_of_text|>').rstrip('<|eot_id|>').strip()
                if 'assistant<|end_header_id|>' in text:
                    text = text[text.find('assistant<|end_header_id|>'):].replace(
                        'assistant<|end_header_id|>', ''
                    ).strip()

        result = cefr_classifier.classify_one(text)
        predictions.append(result["label"])

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='unsloth/Llama-3.2-3B-Instruct')
    parser.add_argument('--transitions', nargs='+', default=['B2_to_C1', 'B1_to_C1', 'A1_to_C1'])
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--num_samples', type=int, default=10)
    parser.add_argument('--cache_path', default='outputs/cefr/cross_cefr_similar_prompts.json')
    parser.add_argument('--layer_range', type=int, nargs=2, default=[-1, -27],
                        help='Layer range to test (inclusive), e.g. -1 -27')
    parser.add_argument('--out_path', default='outputs/cefr/steering_ablation_results.json')
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    model.eval()

    cefr_classifier = CEFRClassifier(
        model_name_or_path="UniversalCEFR/xlm-roberta-base-cefr-all-classifier",
        device='cuda',
        max_length=512,
    )

    with open(args.cache_path) as f:
        all_pairs = json.load(f)

    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))
    layer_start, layer_end = args.layer_range
    layers_to_test = list(range(layer_start, layer_end - 1, -1))

    all_results = {}

    for transition in args.transitions:
        src, tgt = transition.split('_to_')

        if transition not in all_pairs:
            print(f"Skipping {transition}: no pairs found")
            continue

        with open('data/cefr/cefr_samples.json', 'r') as f:
            test_inputs = json.load(f)[src][:args.num_samples]
        test_inputs = [t for t in test_inputs if len(t) >= 10]

        print(f"\n{'='*70}")
        print(f"Transition: {transition} | {len(test_inputs)} test samples | alpha={args.alpha}")
        print(f"Testing layers: {layers_to_test[0]} to {layers_to_test[-1]}")
        print(f"{'='*70}")

        # Compute directions
        cache_path = f'outputs/cefr/{transition}_cached_directions.pt'
        if os.path.exists(cache_path):
            cached = torch.load(cache_path, map_location='cpu')
            if isinstance(cached, tuple):
                directions, raw_norms = cached
            else:
                directions = cached
                raw_norms = {layer: float(cached[layer].norm().item()) for layer in cached}
        else:
            directions, raw_norms = get_direction(
                model, tokenizer, all_pairs[transition],
                -1, hidden_layers, use_pca=True,
            )
            torch.save((directions, raw_norms), cache_path)
            print(f"Saved directions for {transition}")

        # Overshoot projection
        overshoot_dirs, _ = get_overshoot_direction(
            model, tokenizer, transition, all_pairs, hidden_layers,
        )

        max_norm = max(raw_norms.get(l, 1.0) for l in layers_to_test) + 1e-8
        layer_weights = {l: raw_norms.get(l, 1.0) / max_norm for l in layers_to_test}

        # Baseline (no steering)
        print("\nRunning baseline (no steering)...")
        baseline_preds = run_baseline(model, tokenizer, cefr_classifier, test_inputs, tgt)
        baseline_acc = sum(1 for p in baseline_preds if p == tgt) / len(baseline_preds)
        baseline_dist = {}
        for p in baseline_preds:
            baseline_dist[p] = baseline_dist.get(p, 0) + 1
        print(f"  Baseline: {baseline_dist} | {tgt} accuracy: {baseline_acc:.0%}")

        wrapped_model = WrappedReadingVecModel(model, tokenizer)

        layer_results = {}
        print(f"\n{'Layer':>8} {'C1 Acc':>8} {'Distribution':<40} {'Time':>6}")
        print("-" * 70)

        for layer in layers_to_test:
            if layer not in directions:
                continue

            direction = directions[layer][0]

            if overshoot_dirs is not None and layer in overshoot_dirs:
                d = direction.float()
                ov = overshoot_dirs[layer][0].float() if overshoot_dirs[layer].dim() > 1 else overshoot_dirs[layer].float()
                ov_unit = ov / (ov.norm() + 1e-8)
                d_proj = d - (d @ ov_unit) * ov_unit
                if d_proj.norm() / (d.norm() + 1e-8) >= 0.1:
                    direction = d_proj / (d_proj.norm() + 1e-8)

            start = time.time()
            preds = run_single_layer_steering(
                model, tokenizer, cefr_classifier, wrapped_model,
                layer, direction, args.alpha, layer_weights[layer],
                test_inputs, tgt,
            )
            elapsed = time.time() - start

            acc = sum(1 for p in preds if p == tgt) / len(preds)
            dist = {}
            for p in preds:
                dist[p] = dist.get(p, 0) + 1

            layer_results[layer] = {
                'accuracy': acc,
                'distribution': dist,
                'predictions': preds,
                'weight': layer_weights[layer],
            }

            print(f"{layer:>8} {acc:>8.0%} {str(dist):<40} {elapsed:>5.1f}s")

        # Rank layers by target accuracy, break ties by fewer off-target predictions
        ranked = sorted(
            layer_results.items(),
            key=lambda x: (
                x[1]['accuracy'],
                -x[1]['distribution'].get('C2', 0),  # penalize overshoot
                -sum(v for k, v in x[1]['distribution'].items() if k != tgt),
            ),
            reverse=True,
        )

        print(f"\n--- Ranking for {transition} ---")
        print(f"{'Rank':>6} {'Layer':>8} {f'{tgt} Acc':>8} {'Distribution'}")
        print("-" * 60)
        for rank, (layer, metrics) in enumerate(ranked, 1):
            marker = " ***" if rank <= 5 else ""
            print(f"{rank:>6} {layer:>8} {metrics['accuracy']:>8.0%} {metrics['distribution']}{marker}")

        top5 = [layer for layer, _ in ranked[:5]]
        print(f"\nTop-5 steering layers for {transition}: {sorted(top5)}")

        # Also show top-10 combined for multi-layer steering
        top10 = [layer for layer, _ in ranked[:10]]
        print(f"Top-10 steering layers for {transition}: {sorted(top10)}")

        all_results[transition] = {
            'baseline': {'accuracy': baseline_acc, 'distribution': baseline_dist},
            'layers': {str(k): v for k, v in layer_results.items()},
            'ranking': [{'layer': layer, **metrics} for layer, metrics in ranked],
            'top5': sorted(top5),
            'top10': sorted(top10),
            'alpha': args.alpha,
            'num_samples': len(test_inputs),
        }

    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    with open(args.out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {args.out_path}")


if __name__ == '__main__':
    main()
