"""Where does the ~0.05 residual drift come from: label noise or optimisation?

    python -m tact.reward.replay --label score_clean ; python -m tact.reward.replay --model onehot --label score_clean
    python -m tact.reward.drift_cleanlabel

Compares the replay with the real noisy labels against the replay trained (and early-stopped) on score_clean.
Extra control: how much do models trained on the SAME 350 rows but with different seeds disagree on test.csv
(mean |pred_i - pred_j| over the 10 seed pairs)? That is the optimisation-only part of any drift.
Writes results/drift_cleanlabel.md.
"""
import itertools
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from tact.core.model import MODELS, RESULTS_DIR, FrozenBert, get_device, load_csv

ARCH_NAME = {"bert": "BertScorer", "onehot": "OneHotScorer"}
STRAT = {"scratch": "from-scratch", "warm": "warm-start"}
SUF = {"bert": "", "onehot": "_onehot"}


def pm(x, nd=4):
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def per_seed(df, strat):
    g = df[(df.strategy == strat) & (df["round"] >= 2)].groupby("seed")[["drift", "kendall_tau", "first_batch_mse"]].mean()
    return g


def seed_disagreement(arch, strat, label_tag, encoder, device, test_df):
    preds = []
    for seed in range(5):
        path = os.path.join(RESULTS_DIR, f"models{SUF[arch]}{label_tag}", f"{strat}_seed{seed}_final.pt")
        m = MODELS[arch](encoder).to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        preds.append(m.predict(test_df))
    return np.array([np.mean(np.abs(a - b)) for a, b in itertools.combinations(preds, 2)])


def main():
    device = get_device()
    encoder = FrozenBert(device)
    test_df = load_csv("test.csv")
    out = ["# Drift with noisy vs. noise-free training labels\n",
           "Replay as in `replay.py` (7 rounds × 50 rows, 5 seeds, both strategies), once with the real `score` and once with "
           "`score_clean` as the training label (this experiment only; validation labels for early stopping are also "
           "clean). Drift = mean |pred_r − pred_{r−1}| on test.csv, τ = Kendall τ between consecutive rounds; both averaged "
           "over rounds 2–7 per seed, then mean ± std over 5 seeds. `first-batch MSE` is measured against each run's own "
           "training labels, so it is not comparable between the two label types.\n",
           "| architecture | strategy | labels | drift | Kendall τ | first-batch MSE |", "|---|---|---|---|---|---|"]
    stats = {}
    for arch in ("bert", "onehot"):
        noisy = pd.read_csv(os.path.join(RESULTS_DIR, f"replay_metrics{SUF[arch]}.csv"))
        clean = pd.read_csv(os.path.join(RESULTS_DIR, f"replay_metrics{SUF[arch]}_cleanlabel.csv"))
        for s in ("scratch", "warm"):
            a, b = per_seed(noisy, s), per_seed(clean, s)
            stats[(arch, s)] = (a, b)
            for lab, g in (("noisy score", a), ("score_clean", b)):
                out.append(f"| {ARCH_NAME[arch]} | {STRAT[s]} | {lab} | {pm(g.drift)} | {pm(g.kendall_tau, 3)} | "
                           f"{pm(g.first_batch_mse)} |")
    out += ["\n## Change in drift when the label noise is removed\n",
            "| architecture | strategy | drift noisy | drift clean | relative change | Welch p |", "|---|---|---|---|---|---|"]
    for (arch, s), (a, b) in stats.items():
        rel = b.drift.mean() / a.drift.mean() - 1
        out.append(f"| {ARCH_NAME[arch]} | {STRAT[s]} | {a.drift.mean():.4f} | {b.drift.mean():.4f} | {rel:+.0%} | "
                   f"{ttest_ind(a.drift, b.drift, equal_var=False).pvalue:.3f} |")

    out += ["\n## Control: same data, different seeds (final round, 350 rows)\n",
            "Mean |pred_i − pred_j| on test.csv over the 10 seed pairs of the final models. This is disagreement caused only "
            "by initialisation / minibatch order / early stopping, with no new data between the two models.\n",
            "| architecture | strategy | labels | seed-to-seed disagreement |", "|---|---|---|---|"]
    for arch in ("bert", "onehot"):
        for s in ("scratch", "warm"):
            for lab, tag in (("noisy score", ""), ("score_clean", "_cleanlabel")):
                d = seed_disagreement(arch, s, tag, encoder, device, test_df)
                out.append(f"| {ARCH_NAME[arch]} | {STRAT[s]} | {lab} | {d.mean():.4f} ± {d.std(ddof=1):.4f} |")
    with open(os.path.join(RESULTS_DIR, "drift_cleanlabel.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
