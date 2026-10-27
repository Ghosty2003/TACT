"""Prediction heatmap (fruit, afternoon) of the final from-scratch models of one encoding, as in plots.py.

    ARCH=onehot python -m tact.reward.heatmap_arch     # writes results/onehot_as_main/heatmap_pred_vs_truth.png + heatmap_stats.json

Uses the 350-row from-scratch checkpoints of replay.py (results/models for BertScorer, results/models_onehot for
OneHotScorer; both trained on the CPU with the same settings), mean of 5 seeds, same statistics as plots.py.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tact.core import plots
from tact.core.model import ARCH, MODELS, RESULTS_DIR, ROOT, FrozenBert, get_device

CKPT = {"bert": "models", "onehot": "models_onehot"}


def main():
    device = get_device()
    enc = FrozenBert(device)
    truth = plots.truth_grid()
    preds = []
    for seed in range(5):
        m = MODELS[ARCH](enc).to(device)
        m.load_state_dict(torch.load(os.path.join(RESULTS_DIR, CKPT[ARCH], f"scratch_seed{seed}_final.pt"), map_location=device))
        preds.append(plots.pred_grid(m))
    mean_pred = np.mean(preds, axis=0)
    name = MODELS[ARCH].__name__
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    plots.draw_heat(axes[0], truth, "Ground truth")
    im = plots.draw_heat(axes[1], mean_pred, f"{name} (from-scratch, 350 samples, mean of 5 seeds)")
    fig.colorbar(im, ax=axes, label="preference score", shrink=0.9, pad=0.02)
    fig.suptitle(f'"{plots.QUESTION}"  -  time = afternoon', x=0.01, ha="left", fontsize=11, color=plots.INK2)
    fig.savefig(os.path.join(RESULTS_DIR, "heatmap_pred_vs_truth.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)
    stats = {"truth": plots.heatmap_stats(truth), "mean_of_seeds": plots.heatmap_stats(mean_pred),
             "per_seed": {str(s): plots.heatmap_stats(p) for s, p in enumerate(preds)},
             "mean_abs_error_mean_pred": float(np.abs(mean_pred - truth).mean()),
             "pred_mean_grid": mean_pred.round(3).tolist(), "truth_grid": truth.round(3).tolist()}
    with open(os.path.join(RESULTS_DIR, "heatmap_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    sat = stats["mean_of_seeds"]
    print(f"{name}: Saturday/other = {sat['sat_over_other']:.3f} (truth {stats['truth']['sat_over_other']:.3f}); "
          f"mean Spearman(valence) {sat['mean_spearman_valence_vs_pred']:.3f}; monotonic days {sat['days_fully_monotonic']}/7")


if __name__ == "__main__":
    main()
