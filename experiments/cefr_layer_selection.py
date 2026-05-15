#!/usr/bin/env python3
"""
CEFR Layer Discriminability Analysis

Measures each layer's discriminative power for CEFR steering by computing
how well its direction vector separates held-out positive/negative pairs.

Outputs a ranking of layers by discriminability score, so you can steer
only the top-K most effective layers instead of all 10.

Usage:
    python cefr_layer_selection.py --transitions B2_to_C1 B1_to_C1 --top_k 5
"""

import argparse
import json
import os
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cefr_experiment_run import (
    CEFR_ORDER, CEFR_TO_ID,
    forward, get_rep_directions,
    masked_mean, get_hidden_states,
)


def compute_layer_discriminability(
    model, tokenizer, pairs, hidden_layers,
    n_folds=5, use_pca=True, prompt_only=True,
):
    """
    For each layer, measure how well its direction separates held-out pairs.

    Uses K-fold cross-validation:
      - Train direction on (K-1) folds
      - Evaluate on held-out fold: cosine similarity between direction and
        each held-out (pos - neg) difference vector
      - A good layer has high positive cosine on held-out pairs

    Returns dict: {layer_id: {'mean_cosine': float, 'accuracy': float, ...}}
    """
    n = len(pairs)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds

    all_hidden = forward(
        model, tokenizer, pairs,
        batch_size=8, hidden_layers=hidden_layers,
        prompt_only=prompt_only,
    )

    results = {}
    for layer in hidden_layers:
        H = all_hidden[layer]  # (n_pairs, hidden_dim)
        cosines = []
        correct = 0
        total = 0

        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

            H_train = H[train_idx]
            H_val = H[val_idx]

            # Train direction: mean or SVD
            if use_pca:
                H_np = H_train.cpu().detach().numpy().astype(np.float32)
                if H_np.ndim == 1:
                    H_np = H_np[np.newaxis, :]
                U, s, Vt = np.linalg.svd(H_np, full_matrices=False)
                direction = torch.from_numpy(Vt[0].copy()).float()
                mean_check = H_train.mean(dim=0).float()
                if (mean_check @ direction).item() < 0:
                    direction = -direction
            else:
                direction = H_train.mean(dim=0).float()

            direction = direction / (direction.norm() + 1e-8)

            # Evaluate: cosine of each held-out diff with direction
            for i in range(len(val_idx)):
                diff = H_val[i].float()
                cos = (diff @ direction).item() / (diff.norm().item() + 1e-8)
                cosines.append(cos)
                if cos > 0:
                    correct += 1
                total += 1

        results[layer] = {
            'mean_cosine': float(np.mean(cosines)),
            'std_cosine': float(np.std(cosines)),
            'accuracy': correct / max(total, 1),
            'median_cosine': float(np.median(cosines)),
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='unsloth/Llama-3.2-3B-Instruct')
    parser.add_argument('--transitions', nargs='+', default=['B2_to_C1', 'B1_to_C1', 'A1_to_C1'])
    parser.add_argument('--top_k', type=int, default=5, help='Number of top layers to recommend')
    parser.add_argument('--n_folds', type=int, default=5)
    parser.add_argument('--cache_path', default='outputs/cefr/cross_cefr_similar_prompts.json')
    parser.add_argument('--prompt_only', action='store_true', default=True)
    parser.add_argument('--use_pca', action='store_true', default=True)
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    model.eval()

    with open(args.cache_path) as f:
        all_pairs = json.load(f)

    hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1))

    for transition in args.transitions:
        if transition not in all_pairs:
            print(f"Skipping {transition}: no pairs found")
            continue

        pairs = all_pairs[transition]
        print(f"\n{'='*60}")
        print(f"Transition: {transition} ({len(pairs)} pairs)")
        print(f"{'='*60}")

        start = time.time()
        results = compute_layer_discriminability(
            model, tokenizer, pairs, hidden_layers,
            n_folds=args.n_folds, use_pca=args.use_pca,
            prompt_only=args.prompt_only,
        )
        print(f"Computed in {time.time()-start:.1f}s")

        # Sort by mean cosine (discriminability)
        ranked = sorted(results.items(), key=lambda x: x[1]['mean_cosine'], reverse=True)

        print(f"\n{'Layer':>8} {'Mean Cos':>10} {'Std':>8} {'Accuracy':>10} {'Median':>10}")
        print("-" * 50)
        for i, (layer, metrics) in enumerate(ranked):
            marker = " <-- TOP" if i < args.top_k else ""
            print(f"{layer:>8} {metrics['mean_cosine']:>10.4f} {metrics['std_cosine']:>8.4f} "
                  f"{metrics['accuracy']:>10.1%} {metrics['median_cosine']:>10.4f}{marker}")

        top_layers = [layer for layer, _ in ranked[:args.top_k]]
        print(f"\nRecommended layers for {transition}: {sorted(top_layers)}")
        print(f"Python: layer_ids = {sorted(top_layers)}")


if __name__ == '__main__':
    main()
