"""Tables + figures for the follow-up experiments (needs replay.py for bert and onehot, and scaling.py).

    python -m tact.reward.plots_extra

Writes results/habit_subset_comparison.md, results/noise_diagnosis.png, results/scaling_interaction.png,
results/extra_tables.md. score_clean / habit_hit are used here for evaluation only.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tact.core.model import CONFIG, DATA_DIR, RESULTS_DIR, ROOT, load_eval_csv
from tact.core import plots  # shared style (rcParams) and colors

INK, INK2, GRID = plots.INK, plots.INK2, plots.GRID
STRAT = {"scratch": "from-scratch", "warm": "warm-start"}
MODEL_NAME = {"bert": "BertScorer", "onehot": "OneHotScorer"}


def r(name):
    return os.path.join(RESULTS_DIR, name)


def pm(x, nd=4):
    return f"{x.mean():.{nd}f} ± {x.std():.{nd}f}"


# ---------------------------------------------------------------- 1. habit subset
def habit_table():
    diag = {"bert": pd.read_csv(r("replay_diagnostics.csv")), "onehot": pd.read_csv(r("replay_diagnostics_onehot.csv"))}
    met = {"bert": pd.read_csv(r("replay_metrics.csv")), "onehot": pd.read_csv(r("replay_metrics_onehot.csv"))}
    n_habit = int(diag["bert"].habit_n.iloc[0])
    n_test = len(load_eval_csv("test.csv"))
    out = [f"# Habit-hit subset of test.csv: BertScorer vs OneHotScorer (final round, 350 training samples)\n",
           f"Habit-hit rows (`habit_hit` non-empty) in test.csv: **{n_habit} of {n_test}**. This is a very small subset, "
           "so every number in the habit columns is noisy; differences smaller than the ± are not meaningful. "
           "Mean ± std over 5 seeds. Pairwise accuracy within the subset uses only pairs with a label gap > 0.05 "
           "(habit scores are all low, so there are few such pairs).\n",
           "| model | strategy | overall MSE | habit MSE | habit Spearman | habit pairwise acc | non-habit MSE |",
           "|---|---|---|---|---|---|---|"]
    for m in ("bert", "onehot"):
        for s in ("scratch", "warm"):
            g = diag[m][(diag[m].strategy == s) & (diag[m]["round"] == 7)]
            out.append(f"| {MODEL_NAME[m]} | {STRAT[s]} | {pm(g.mse_score)} | {pm(g.habit_mse)} | "
                       f"{pm(g.habit_spearman, 3)} | {pm(g.habit_pairwise_acc, 3)} | {pm(g.nonhabit_mse)} |")
    out += ["\n## Overall test metrics, final round (for reference)\n",
            "| model | strategy | MSE | Spearman | pairwise acc |", "|---|---|---|---|---|"]
    for m in ("bert", "onehot"):
        for s in ("scratch", "warm"):
            g = met[m][(met[m].strategy == s) & (met[m]["round"] == 7)]
            out.append(f"| {MODEL_NAME[m]} | {STRAT[s]} | {pm(g.mse)} | {pm(g.spearman, 3)} | {pm(g.pairwise_acc, 3)} |")
    # stability of the baseline, same definition as before (rounds 2-7)
    out += ["\n## Stability of both models (mean over rounds 2-7, then mean ± std over 5 seeds)\n",
            "| model | strategy | drift | first-batch MSE | Kendall tau |", "|---|---|---|---|---|"]
    for m in ("bert", "onehot"):
        later = met[m][met[m]["round"] >= 2].groupby(["strategy", "seed"])[["drift", "first_batch_mse", "kendall_tau"]].mean()
        for s in ("scratch", "warm"):
            g = later.loc[s]
            out.append(f"| {MODEL_NAME[m]} | {STRAT[s]} | {pm(g.drift)} | {pm(g.first_batch_mse)} | {pm(g.kendall_tau, 3)} |")
    with open(r("habit_subset_comparison.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    return diag


# ---------------------------------------------------------------- 2. noise diagnosis
def noise_figure(diag):
    d = diag["bert"]
    floor = CONFIG["label_noise"] ** 2
    VIO, AQUA = "#6250d6", "#1baf7a"
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    for col, s in enumerate(("scratch", "warm")):
        for row, (kind, cols, ttl) in enumerate([("test", ("mse_score", "mse_clean"), "test.csv"),
                                                 ("train", ("train_mse_score", "train_mse_clean"), "training rows seen so far")]):
            ax = axes[row, col]
            for c, color, lab in zip(cols, (VIO, AQUA), ("vs. noisy label (score)", "vs. clean label (score_clean)")):
                g = d[d.strategy == s].groupby("n_train")[c].agg(["mean", "std"])
                ax.plot(g.index, g["mean"], color=color, lw=2, marker="o", ms=5, label=lab)
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15, lw=0)
            ax.axhline(floor, color=INK2, lw=1, ls="--")
            ax.text(52, floor, "label-noise variance σ² = 0.0025", ha="left", va="bottom", fontsize=8, color=INK2)
            ax.set_title(f"{STRAT[s]}: MSE on {ttl}", fontsize=10, loc="left", color=INK)
            ax.set_xticks(sorted(d.n_train.unique()))
            if col == 0:
                ax.set_ylabel("MSE")
            if row == 1:
                ax.set_xlabel("training samples")
    axes[0, 0].legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.88))
    fig.suptitle("Noise diagnosis (BertScorer, mean ± std over 5 seeds)", x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout()
    fig.savefig(r("noise_diagnosis.png"), dpi=170)
    plt.close(fig)

    # Share of the training-label noise the model has absorbed: slope of the fit error (pred - clean) on the
    # noise eps = score - score_clean over the training rows seen so far. With e = pred - clean:
    #   mean(e*eps) = (MSE_clean + mean(eps^2) - MSE_score) / 2 ;  absorbed = mean(e*eps) / mean(eps^2).
    # 0 = noise not fitted, 1 = noise fully memorised.
    tf = load_eval_csv("train.csv")
    eps2 = {n: float(((tf.score - tf.score_clean).iloc[:n] ** 2).mean()) for n in d.n_train.unique()}
    d = d.assign(absorbed=lambda x: [(a + eps2[n] - b) / 2 / eps2[n] for a, b, n in
                                     zip(x.train_mse_clean, x.train_mse_score, x.n_train)])
    out = ["# Noise diagnosis numbers (BertScorer, mean over 5 seeds)\n",
           f"Label-noise variance: nominal sigma^2 = {floor:.4f}; realised on test.csv = "
           f"{float(((lambda t: t.score - t.score_clean)(load_eval_csv('test.csv')) ** 2).mean()):.4f} "
           "(the test-set MSE difference below is this plus a finite-sample cross term over 75 rows).\n",
           "| strategy | n | test MSE vs score | test MSE vs score_clean | difference | train MSE vs score | train MSE vs score_clean | train noise absorbed |",
           "|---|---|---|---|---|---|---|---|"]
    t = d.groupby(["strategy", "n_train"])[["mse_score", "mse_clean", "train_mse_score", "train_mse_clean", "absorbed"]].mean()
    for (s, n), x in t.iterrows():
        out.append(f"| {STRAT[s]} | {n} | {x.mse_score:.4f} | {x.mse_clean:.4f} | {x.mse_score - x.mse_clean:.4f} | "
                   f"{x.train_mse_score:.4f} | {x.train_mse_clean:.4f} | {x.absorbed:.2f} |")
    return out


# ---------------------------------------------------------------- 3. scaling
def habit_counts():
    rows = []
    for name, ddir in [("data", DATA_DIR), ("data_1000", os.path.join(ROOT, "data_1000")),
                       ("data_2000", os.path.join(ROOT, "data_2000"))]:
        t = load_eval_csv("train.csv", ddir)
        h = t.habit_hit.notna()
        fs = h & (t.question.str.contains("fruit")) & (t.day == 5)
        rows.append((name, len(t), int(h.sum()), int(fs.sum())))
    return rows


def scaling_figure():
    df = pd.read_csv(r("scaling_metrics.csv"))
    truth = json.load(open(r("scaling_grids.json")))["truth_sat_over_other"]
    BLUE, ORANGE = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    g = df.groupby("n_train")["sat_over_other"].agg(["mean", "std"])
    ax.scatter(df.n_train, df.sat_over_other, color=BLUE, alpha=0.35, s=22, zorder=3)
    ax.plot(g.index, g["mean"], color=BLUE, lw=2, marker="o", ms=6, label="model (mean of 5 seeds; dots = seeds)", zorder=4)
    ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=BLUE, alpha=0.15, lw=0)
    ax.axhline(truth, color=INK2, ls="--", lw=1.2)
    ax.text(350, truth - 0.012, f"ground truth {truth:.2f}", ha="left", va="top", fontsize=9, color=INK2)
    ax.set_ylim(0, None)
    ax.set_ylabel("collapse ratio (Saturday mean / other days)")
    ax.set_title("Saturday-fruit collapse vs. training volume", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    for c, color, lab in (("habit_mse", ORANGE, "habit-hit subset"), ("nonhabit_mse", BLUE, "non-habit rows")):
        g = df.groupby("n_train")[c].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], color=color, lw=2, marker="o", ms=6, label=lab)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=color, alpha=0.15, lw=0)
    ax.set_ylabel("test MSE (each set's own test.csv)")
    ax.set_title("Test MSE: habit vs. non-habit rows", fontsize=10, loc="left", color=INK)
    ax.legend(frameon=False)
    for a in axes:
        a.set_xscale("log", base=2)
        a.set_xticks(sorted(df.n_train.unique()), [str(n) for n in sorted(df.n_train.unique())])
        a.minorticks_off()
        a.set_xlabel("training samples (from-scratch, final round)")
    fig.suptitle("Data-scale sweep: 500 / 1000 / 2000-sample datasets (train = 350 / 700 / 1400)", x=0.01, ha="left",
                 fontsize=11, color=INK2)
    fig.tight_layout()
    fig.savefig(r("scaling_interaction.png"), dpi=170)
    plt.close(fig)

    out = ["\n# Scaling sweep (from-scratch, final round, 5 seeds)\n",
           "| n_train | collapse ratio (per-seed mean ± std) | collapse ratio of the seed-mean grid | habit n (test) | habit MSE | non-habit MSE | overall MSE |",
           "|---|---|---|---|---|---|---|"]
    grids = json.load(open(r("scaling_grids.json")))["grids"]
    from tact.core.plots import heatmap_stats
    for n, x in df.groupby("n_train"):
        mg = heatmap_stats(np.mean(grids[str(n)], axis=0))["sat_over_other"]
        out.append(f"| {n} | {pm(x.sat_over_other, 3)} | {mg:.3f} | {int(x.habit_n.iloc[0])} | {pm(x.habit_mse)} | "
                   f"{pm(x.nonhabit_mse)} | {pm(x.mse)} |")
    out += [f"\nGround-truth collapse ratio: {truth:.3f}.\n",
            "Habit samples in each training set (all habit rules / Saturday-fruit rule only):\n",
            "| dataset | train rows | habit-hit rows | Saturday-fruit habit rows |", "|---|---|---|---|"]
    out += [f"| {n} | {a} | {b} | {c} |" for n, a, b, c in habit_counts()]
    return out


def main():
    diag = habit_table()
    out = noise_figure(diag) + scaling_figure()
    with open(r("extra_tables.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
