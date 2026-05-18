#!/usr/bin/env python3
import argparse
import csv
import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


CEFR_ORDER = ["A1", "A2", "B1", "B2", "C1", "C2"]


def load_summary(csv_path):
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            try:
                rows[r['transition']] = {
                    'best_alpha': float(r['best_alpha']),
                    'dev_acc': float(r['dev_accuracy']),
                    'eval_acc': float(r['eval_accuracy']),
                    'n': int(r['eval_n']),
                }
            except (KeyError, ValueError):
                continue
    return rows


def build_matrix(rows, metric='eval_acc'):
    n = len(CEFR_ORDER)
    M = np.full((n, n), np.nan)
    for i, src in enumerate(CEFR_ORDER):
        for j, tgt in enumerate(CEFR_ORDER):
            if i == j:
                continue
            key = f"{src}_to_{tgt}"
            if key in rows:
                M[i, j] = rows[key][metric]
    return M


def draw_heatmap(M, alpha_M, out_path, title, cmap_name='RdYlGn',
                 annotate_alpha=True, dpi=200,
                 cmap_lo=0.0, cmap_hi=1.0):
    n = len(CEFR_ORDER)
    fig, ax = plt.subplots(figsize=(8.2, 6.8))

    base_cmap = plt.get_cmap(cmap_name)
    if cmap_lo > 0.0 or cmap_hi < 1.0:
        sampled = base_cmap(np.linspace(cmap_lo, cmap_hi, 256))
        cmap = mcolors.LinearSegmentedColormap.from_list(
            f'{cmap_name}_truncated', sampled,
        )
    else:
        cmap = base_cmap.copy()
    cmap.set_bad(color='#f0f0f0')  # diagonal / missing cells

    masked = np.ma.array(M, mask=np.isnan(M))
    im = ax.imshow(masked, cmap=cmap, vmin=0.0, vmax=1.0, aspect='equal')

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
            r, g, b, _ = cmap((v - 0.0) / 1.0)
            luminance = 0.299 * r + 0.587 * g + 0.114 * b
            txt_color = 'white' if luminance < 0.55 else '#1a1a1a'
            ax.text(j, i, f'{v:.0%}',
                    ha='center', va='center',
                    color=txt_color, fontsize=12, fontweight='bold')
            if annotate_alpha and not np.isnan(alpha_M[i, j]):
                ax.text(j, i + 0.32, fr'$\alpha$={alpha_M[i, j]:.1f}',
                        ha='center', va='center',
                        color=txt_color, fontsize=8, alpha=0.85)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Propotion hitting target', fontsize=11)
    cbar.ax.tick_params(labelsize=10)

    ax.set_title(title, fontsize=14, pad=18, fontweight='bold')
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary_csv', default='../outputs/cefr/all_pairs/all_pairs_summary.csv')
    parser.add_argument('--out_path', default='../outputs/cefr/all_pairs/steering_accuracy_heatmap.png')
    parser.add_argument('--title', default='CEFR Activation Steering - accuracy @ target-level')
    parser.add_argument('--cmap', default='Greens',
                        help='Matplotlib colormap. Try: Greens, YlGn, RdYlGn, YlGnBu, viridis.')
    parser.add_argument('--cmap_lo', type=float, default=0.05,
                        help='Lower end of colormap range (0-1). Higher = darker minimum.')
    parser.add_argument('--cmap_hi', type=float, default=0.55,
                        help='Upper end of colormap range (0-1). Lower = lighter maximum.')
    parser.add_argument('--no_alpha', dest='annotate_alpha', action='store_false', default=True,
                        help='Hide the alpha annotation in each cell.')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)

    rows = load_summary(args.summary_csv)
    if not rows:
        raise SystemExit(f"No rows loaded from {args.summary_csv}")
    acc_M = build_matrix(rows, 'eval_acc')
    alpha_M = build_matrix(rows, 'best_alpha')

    draw_heatmap(acc_M, alpha_M, args.out_path, args.title,
                 cmap_name=args.cmap, annotate_alpha=args.annotate_alpha,
                 cmap_lo=args.cmap_lo, cmap_hi=args.cmap_hi)
    print(f"Saved: {args.out_path}")

    n = len(CEFR_ORDER)
    upper = np.array([acc_M[i, j] for i in range(n) for j in range(n) if j > i and not np.isnan(acc_M[i, j])])
    lower = np.array([acc_M[i, j] for i in range(n) for j in range(n) if j < i and not np.isnan(acc_M[i, j])])
    overall = np.concatenate([upper, lower])
    print(f"\nSummary:")
    print(f"  overall mean acc:  {overall.mean():.2%}  (n={len(overall)})")
    print(f"  upshift mean acc:  {upper.mean():.2%}  (n={len(upper)}, src<tgt)")
    print(f"  downshift mean acc:{lower.mean():.2%}  (n={len(lower)}, src>tgt)")


if __name__ == '__main__':
    main()
