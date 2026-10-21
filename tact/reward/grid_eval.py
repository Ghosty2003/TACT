"""Full-grid ground-truth evaluation of the two architectures (final round, 350 samples, 5 seeds, both strategies).

    python -m tact.reward.grid_eval        # needs replay.py runs for bert and onehot (saved final checkpoints)

Grid = topic(12) x emotion(6) x day(7) x time(3) = 1512 cells; each topic uses its first question text.
Target = score_clean from the config.json formula (necessity * valence * availability). No label noise,
so this is a noise-free measure of what the model learned. Evaluation only: models are never trained on it.
Also exposes build_grid / grid_metrics, reused by hparam_fairness.py and noise_robustness.py.
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from tact.core.evaluate import metrics_from_pred
from tact.core.generate_dataset import QUESTIONS, availability
from tact.core.model import CONFIG, DATA_DIR, EMOTIONS, RESULTS_DIR, MODELS, FrozenBert, get_device
from tact.core.plots import is_habit, necessity

TOPICS = list(QUESTIONS)                      # 12 topics, config order
SHAPE = (len(TOPICS), len(EMOTIONS), 7, 3)    # topic, emotion, day, time
SUFFIX = {"bert": "", "onehot": "_onehot"}
ARCH_NAME = {"bert": "BertScorer", "onehot": "OneHotScorer"}
STRAT = {"scratch": "from-scratch", "warm": "warm-start"}


def build_grid():
    """Returns (DataFrame of model inputs, truth, habit mask), all flattened in C order over SHAPE."""
    rows, truth, habit = [], [], []
    for t in TOPICS:
        for e in EMOTIONS:
            for d in range(7):
                for m in range(3):
                    rows.append({"question": QUESTIONS[t][0], "emotion": e, "day": d, "time": m})
                    truth.append(necessity(t, d, m) * CONFIG["valence"][e] * availability(d, m))
                    habit.append(is_habit(t, d, m))
    return pd.DataFrame(rows), np.array(truth), np.array(habit)


def grid_metrics(pred, truth, habit):
    """MSE / Spearman / pairwise on all cells, plus the habit-cell subset and MSE on the non-habit cells."""
    a = metrics_from_pred(pred, truth)
    h = metrics_from_pred(pred[habit], truth[habit])
    return {"mse": a["mse"], "spearman": a["spearman"], "pairwise_acc": a["pairwise_acc"],
            "habit_mse": h["mse"], "habit_spearman": h["spearman"], "habit_pairwise_acc": h["pairwise_acc"],
            "nonhabit_mse": float(np.mean((pred[~habit] - truth[~habit]) ** 2))}


def n_pairs(truth, margin=0.05):
    d = np.abs(truth[:, None] - truth[None, :])
    return int((np.triu(d, 1) > margin).sum())


def pm(x, nd=4):
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def main():
    device = get_device()
    encoder = FrozenBert(device)
    grid, truth, habit = build_grid()
    # Truth check against the generated data (verification only): formula vs score_clean column (3-decimal rounding)
    ds = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))
    f = np.array([necessity(t, d, m) * CONFIG["valence"][e] * availability(d, m)
                  for t, e, d, m in zip(ds.topic, ds.emotion, ds.day, ds.time)])
    truth_err = float(np.abs(f - ds.score_clean.values).max())
    assert truth_err < 1e-3, truth_err

    rows = []
    for arch in ("bert", "onehot"):
        for strat in ("scratch", "warm"):
            for seed in range(5):
                path = os.path.join(RESULTS_DIR, f"models{SUFFIX[arch]}", f"{strat}_seed{seed}_final.pt")
                model = MODELS[arch](encoder).to(device)
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                rows.append({"arch": arch, "strategy": strat, "seed": seed,
                             **grid_metrics(model.predict(grid), truth, habit)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "grid_eval.csv"), index=False)

    out = ["# Full-grid ground-truth evaluation (final round, 350 training samples)\n",
           f"Grid: 12 topics × 6 emotions × 7 days × 3 times = {len(truth)} cells (first question text of each topic). "
           f"Target: the noise-free `score_clean` from the config.json formula (checked against the {len(ds)} rows of "
           f"dataset.csv: max |formula − score_clean| = {truth_err:.4f}, i.e. rounding only). "
           f"Habit cells (a habit rule fires): **{int(habit.sum())}**; other cells: {int((~habit).sum())}. "
           f"Pairs with label gap > 0.05: {n_pairs(truth):,} (all cells), {n_pairs(truth[habit]):,} (habit cells). "
           "Mean ± std (ddof=1) over 5 seeds.\n",
           "## All cells\n", "| architecture | strategy | MSE | Spearman | pairwise acc |", "|---|---|---|---|---|"]
    for arch in ("bert", "onehot"):
        for s in ("scratch", "warm"):
            g = df[(df.arch == arch) & (df.strategy == s)]
            out.append(f"| {ARCH_NAME[arch]} | {STRAT[s]} | {pm(g.mse)} | {pm(g.spearman, 3)} | {pm(g.pairwise_acc, 3)} |")
    out += ["\n## Habit cells vs. the rest\n",
            "| architecture | strategy | habit MSE | habit Spearman | habit pairwise acc | non-habit MSE |",
            "|---|---|---|---|---|---|"]
    for arch in ("bert", "onehot"):
        for s in ("scratch", "warm"):
            g = df[(df.arch == arch) & (df.strategy == s)]
            out.append(f"| {ARCH_NAME[arch]} | {STRAT[s]} | {pm(g.habit_mse)} | {pm(g.habit_spearman, 3)} | "
                       f"{pm(g.habit_pairwise_acc, 3)} | {pm(g.nonhabit_mse)} |")
    out += ["\n## BertScorer − OneHotScorer\n",
            "Mean difference over seeds and Welch t-test p-value (5 vs 5 seeds, a crude guide only). For the MSE rows a negative difference means BertScorer is better; for Spearman / pairwise rows a positive difference does.\n",
            "| strategy | metric | BertScorer | OneHotScorer | difference | Welch p |", "|---|---|---|---|---|---|"]
    for s in ("scratch", "warm"):
        b, o = df[(df.arch == "bert") & (df.strategy == s)], df[(df.arch == "onehot") & (df.strategy == s)]
        for col, nd in (("mse", 4), ("habit_mse", 4), ("nonhabit_mse", 4), ("spearman", 3), ("habit_spearman", 3),
                        ("pairwise_acc", 3), ("habit_pairwise_acc", 3)):
            p = ttest_ind(b[col], o[col], equal_var=False).pvalue
            out.append(f"| {STRAT[s]} | {col} | {b[col].mean():.{nd}f} | {o[col].mean():.{nd}f} | "
                       f"{b[col].mean() - o[col].mean():+.{nd}f} | {p:.3f} |")
    with open(os.path.join(RESULTS_DIR, "grid_eval.md"), "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
