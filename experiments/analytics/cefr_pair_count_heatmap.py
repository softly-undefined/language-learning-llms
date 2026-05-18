#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def load_counts(json_path):
    with open(json_path) as f:
        data = json.load(f)
    return {k: len(v) for k, v in data.items()}


def build_matrix(counts):
    n = len(CEFR_ORDER)
    M = np.full((n, n), np.nan)
    for i, src in enumerate(CEFR_ORDER):
        for j, tgt in enumerate(CEFR_ORDER):
            if i == j:
                continue
            key = f"{src}_to_{tgt}"
            if key in counts:
                M[i, j] = counts[key]
    return M


def draw_heatmap(M, out_path, title, cmap_name='Blues', dpi=200,
                 cmap_lo=0.05, cmap_hi=0.65, log_scale=True):
    n = len(CEFR_ORDER)
    fig, ax = plt.subplots(figsize=(8.2, 6.8))

    base_cmap = plt.get_cmap(cmap_name)
    sampled = base_cmap(np.linspace(cmap_lo, cmap_hi, 256))
    cmap = mcolors.LinearSegmentedColormap.from_list(f'{cmap_name}_trunc', sampled)
    cmap.set_bad(color='#f0f0f0')

    masked = np.ma.array(M, mask=np.isnan(M))
    vmin = max(1, int(np.nanmin(M)))
    vmax = int(np.nanmax(M))
    if log_scale:
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    im = ax.imshow(masked, cmap=cmap, norm=norm, aspect='equal')

    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(CEFR_ORDER, fontsize=12, fontweight='bold')
    ax.set_yticklabels(CEFR_ORDER, fontsize=12, fontweight='bold')
    ax.set_xlabel('Target level', fontsize=13, labelpad=10)
    ax.set_ylabel('Source level', fontsize=13, labelpad=10)
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()

    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, '—', ha='center', va='center',
                        color='#888', fontsize=14)
                continue
            v = M[i, j]
            if np.isnan(v):
                continue
            rgba = cmap(norm(v))
            luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            txt_color = 'white' if luminance < 0.55 else '#1a1a1a'
            ax.text(j, i, f'{int(v)}',
                    ha='center', va='center',
                    color=txt_color, fontsize=12, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Contrastive pairs' + (' (log scale)' if log_scale else ''),
                   fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(title, fontsize=14, pad=18, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs_json', default='../outputs/cefr/cross_cefr_similar_prompts.json')
    parser.add_argument('--out_path', default='../outputs/cefr/all_pairs/contrastive_pair_count_heatmap.png')
    parser.add_argument('--title', default='Contrastive Pairs per CEFR Transition')
    parser.add_argument('--cmap', default='Blues')
    parser.add_argument('--cmap_lo', type=float, default=0.05)
    parser.add_argument('--cmap_hi', type=float, default=0.65)
    parser.add_argument('--linear', action='store_true', help='Use linear color scale (default: log10).')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)

    counts = load_counts(args.pairs_json)
    if not counts:
        raise SystemExit(f"No pairs loaded from {args.pairs_json}")
    M = build_matrix(counts)

    draw_heatmap(M, args.out_path, args.title,
                 cmap_name=args.cmap, cmap_lo=args.cmap_lo, cmap_hi=args.cmap_hi,
                 log_scale=not args.linear)
    print(f"Saved: {args.out_path}")

    vals = np.array([v for v in counts.values()])
    print(f"\nSummary:")
    print(f"  transitions covered: {len(vals)} / 30")
    print(f"  total pairs:         {vals.sum()}")
    print(f"  min / median / max:  {vals.min()} / {int(np.median(vals))} / {vals.max()}")


if __name__ == '__main__':
    main()
