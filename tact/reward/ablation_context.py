"""Context-input ablation (results/ablation_context.md).

    python -m tact.reward.ablation_context

Default BertScorer, trained from scratch on the 350 rows of train.csv (early stopping on val.csv, seeds as in the final
round of replay.py), once with the full context and once each with the emotion / time-of-day / day-of-week embedding replaced
by zeros (the embedding output is zeroed, the context MLP is unchanged). All four variants are trained in this script (same
feature cache, same seeds), so the full-context control is comparable with the ablated models; the checkpoints of
replay.py (trained on the CPU) are listed for reference only. Evaluation: the noise-free 1512-cell grid of grid_eval.py.
"""
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr, ttest_ind

from tact.reward.grid_eval import SHAPE, TOPICS, build_grid, grid_metrics, pm
from tact.core.model import CONFIG, EMOTIONS, RESULTS_DIR, BertScorer, FrozenBert, get_device, load_csv, make_tensors
from tact.reward.replay import HPARAMS, run_seed, train

VARIANTS = {"full": (), "no_emotion": ("emotion",), "no_time": ("time",), "no_day": ("day",)}
LABEL = {"full": "full context (control)", "no_emotion": "− emotion", "no_time": "− time of day", "no_day": "− day of week"}
VAL = np.array([CONFIG["valence"][e] for e in EMOTIONS])


def emotion_direction(pred):
    """Mean over the 252 (topic, day, time) groups of the Spearman between the 6 emotions' predictions and their valence
    (a constant prediction counts as 0)."""
    p = pred.reshape(SHAPE)
    rs = []
    for t in range(SHAPE[0]):
        for d in range(7):
            for m in range(3):
                x = p[t, :, d, m]
                rs.append(0.0 if np.ptp(x) == 0 else spearmanr(x, VAL).statistic)
    return float(np.mean(rs))


def main():
    dev = get_device()
    enc = FrozenBert(dev)
    train_df, val_df, test_df = load_csv("train.csv"), load_csv("val.csv"), load_csv("test.csv")
    tr, va = make_tensors(train_df, enc, dev), make_tensors(val_df, enc, dev)
    grid, truth, habit = build_grid()
    topic_idx = np.repeat(np.arange(SHAPE[0]), int(np.prod(SHAPE[1:])))
    rows = []
    for name, drop in VARIANTS.items():
        for seed in range(5):
            s = run_seed(seed, 7)
            torch.manual_seed(s)
            np.random.seed(s)
            model = BertScorer(enc, drop_ctx=drop).to(dev)
            train(model, tr, va, HPARAMS, gen_seed=s)
            pred = model.predict(grid)
            g = grid_metrics(pred, truth, habit)
            row = {"variant": name, "seed": seed, **g, "emotion_dir": emotion_direction(pred),
                   "test_mse": float(np.mean((model.predict(test_df) - test_df.score.values) ** 2))}
            for ti, t in enumerate(TOPICS):
                m = habit & (topic_idx == ti)
                if m.any():
                    row[f"habit_mse_{t}"] = float(np.mean((pred[m] - truth[m]) ** 2))
            rows.append(row)
            print(f"[ablation] {name} seed={seed} grid mse={g['mse']:.4f} habit mse={g['habit_mse']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation_context.csv"), index=False)

    ref = pd.read_csv(os.path.join(RESULTS_DIR, "grid_eval.csv"))
    ref = ref[(ref.arch == "bert") & (ref.strategy == "scratch")]
    full = df[df.variant == "full"]

    def p_vs_full(v, col):
        return ttest_ind(df[df.variant == v][col], full[col], equal_var=False).pvalue

    out = ["# Context-input ablation (`ablation_context.py`)\n",
           "Default BertScorer, from scratch on the 350 rows of train.csv (early stopping on val.csv), 5 seeds; the embedding of "
           "one context field is replaced by zeros. Evaluation on the 1512-cell noise-free grid (12 topics × 6 emotions × 7 days × "
           "3 times, `score_clean` formula); 60 of the cells are habit cells. Mean ± std over seeds; p = Welch test against the "
           "full-context control.\n",
           "## All cells\n", "| variant | MSE | Spearman | pairwise acc | emotion direction | MSE p vs full |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = df[df.variant == v]
        pv = "–" if v == "full" else f"{p_vs_full(v, 'mse'):.3f}"
        out.append(f"| {LABEL[v]} | {pm(g.mse, 4)} | {pm(g.spearman)} | {pm(g.pairwise_acc)} | {pm(g.emotion_dir)} | {pv} |")
    out += ["\nEmotion direction = Spearman between the predictions for the six emotions and their valence, within each of the 252 "
            "(topic, day, time) groups, averaged (a constant prediction counts as 0; the truth gives 1.0).\n",
            "## Habit cells and the rest\n",
            "| variant | habit MSE | habit Spearman | habit pairwise acc | non-habit MSE | habit MSE p vs full |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        g = df[df.variant == v]
        pv = "–" if v == "full" else f"{p_vs_full(v, 'habit_mse'):.3f}"
        out.append(f"| {LABEL[v]} | {pm(g.habit_mse, 4)} | {pm(g.habit_spearman)} | {pm(g.habit_pairwise_acc)} | "
                   f"{pm(g.nonhabit_mse, 4)} | {pv} |")
    hab_topics = [t for t in TOPICS if f"habit_mse_{t}" in df]
    rules = {"fruit": "fruit, Saturday", "bread": "bread, Sunday morning", "coffee": "coffee, weekday mornings",
             "snack": "snack, Friday evening"}
    out += ["\n## MSE on the habit cells of each habit rule\n",
            "| variant | " + " | ".join(rules.get(t, t) for t in hab_topics) + " |", "|---|" + "---|" * len(hab_topics)]
    for v in VARIANTS:
        g = df[df.variant == v]
        out.append(f"| {LABEL[v]} | " + " | ".join(pm(g[f"habit_mse_{t}"], 4) for t in hab_topics) + " |")
    out += ["\n## Reference: checkpoints of replay.py (trained on the CPU, from-scratch, final round)\n",
            f"All cells MSE {pm(ref.mse, 4)}, Spearman {pm(ref.spearman)}, pairwise {pm(ref.pairwise_acc)}; habit MSE "
            f"{pm(ref.habit_mse, 4)}. Differences from the control above come from training device and feature-cache order "
            "only (the training is sensitive to 1e-6 perturbations).\n"]
    with open(os.path.join(RESULTS_DIR, "ablation_context.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
