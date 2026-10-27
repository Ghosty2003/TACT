"""Figures + summary tables from results/replay_metrics.csv and the saved final scratch models.

    python -m tact.core.plots

Outputs (results/): performance_curves.png, stability_curves.png, heatmap_pred_vs_truth.png,
heatmap_pred_per_seed.png, summary_tables.md, heatmap_stats.json.
The ground-truth heatmap is the only place that uses the true generating formula (config.json).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr

from tact.core.generate_dataset import availability  # not stored in config.json, only its name in "formula"
from tact.core.model import CONFIG, EMOTIONS, MODELS, RESULTS_DIR, BertScorer, FrozenBert, get_device

METRICS_CSV = os.path.join(RESULTS_DIR, "replay_metrics.csv")
QUESTION, TIME_IDX = "Which fruit should I use for the salad?", 1
TOPIC = "fruit"
DAYS = CONFIG["days"]

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
COLORS = {"scratch": "#2a78d6", "warm": "#eb6834"}
LABELS = {"scratch": "from-scratch", "warm": "warm-start"}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ---------------------------------------------------------------- curves
def curve_figure(df, panels, xcol, xlabel, fname, title):
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 3.9))
    for ax, (col, ylabel, note) in zip(axes, panels):
        for strat in ("scratch", "warm"):
            g = df[df.strategy == strat].groupby(xcol)[col].agg(["mean", "std", "count"]).dropna(subset=["mean"])
            ax.plot(g.index, g["mean"], color=COLORS[strat], lw=2, marker="o", ms=5, label=LABELS[strat])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"],
                            color=COLORS[strat], alpha=0.15, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(note, fontsize=10, color=INK, loc="left")
        ax.set_xticks(sorted(df[xcol].unique()))
    axes[0].legend(frameon=False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=170)
    plt.close(fig)


# ---------------------------------------------------------------- ground truth / predictions
def is_habit(topic, day, time):
    """True if a habit rule of config.json resolves the ambiguity for (topic, day, time)."""
    for h in CONFIG["habits"]:
        if h["topic"] != topic:
            continue
        if h.get("weekday_only") and day >= 5:
            continue
        if h["day"] is not None and h["day"] != day:
            continue
        if h["time"] is not None and h["time"] != time:
            continue
        return True
    return False


def necessity(topic, day, time):
    return CONFIG["habit_necessity"] if is_habit(topic, day, time) else CONFIG["base_necessity"][topic]


def truth_grid():
    """(emotion x day) = necessity * valence(emotion) * availability(day, time)."""
    return np.array([[necessity(TOPIC, d, TIME_IDX) * CONFIG["valence"][e] * availability(d, TIME_IDX)
                      for d in range(7)] for e in EMOTIONS])


def pred_grid(model):
    rows = [{"question": QUESTION, "emotion": e, "day": d, "time": TIME_IDX} for e in EMOTIONS for d in range(7)]
    return model.predict(pd.DataFrame(rows)).reshape(len(EMOTIONS), 7)


def draw_heat(ax, grid, title):
    im = ax.imshow(grid, vmin=0, vmax=1, cmap=SEQ, aspect="auto")
    ax.set_xticks(range(7), DAYS)
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(grid.shape[0]):
        for j in range(7):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if grid[i, j] > 0.45 else INK)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    return im


def heatmap_stats(grid):
    """Saturday-fruit collapse and emotion monotonicity for an (emotion x day) grid."""
    val = np.array([CONFIG["valence"][e] for e in EMOTIONS])
    order = np.argsort(val)  # low -> high valence
    sat = grid[:, 5].mean()
    other = np.delete(grid, 5, axis=1).mean()
    per_day_rho = [spearmanr(val, grid[:, d]).statistic for d in range(7)]
    mono = [bool(np.all(np.diff(grid[order, d]) >= 0)) for d in range(7)]
    return {
        "sat_mean": float(sat), "other_days_mean": float(other), "sat_over_other": float(sat / other),
        "emotion_order_low_to_high": [EMOTIONS[i] for i in order],
        "mean_spearman_valence_vs_pred": float(np.mean(per_day_rho)),
        "days_fully_monotonic": int(sum(mono)), "monotonic_by_day": dict(zip(DAYS, mono)),
    }


def heatmaps():
    device = get_device()
    encoder = FrozenBert(device)
    truth = truth_grid()
    preds = []
    for seed in range(5):
        path = os.path.join(RESULTS_DIR, "models", f"scratch_seed{seed}_final.pt")
        m = MODELS["bert"](encoder).to(device)     # = BertScorer unless TACT_V2=1 (then the final architecture)
        m.load_state_dict(torch.load(path, map_location=device))
        preds.append(pred_grid(m))
    mean_pred = np.mean(preds, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
    draw_heat(axes[0], truth, "Ground truth")
    im = draw_heat(axes[1], mean_pred, "Prediction (from-scratch, 350 samples, mean of 5 seeds)")
    fig.colorbar(im, ax=axes, label="preference score", shrink=0.9, pad=0.02)
    fig.suptitle(f'"{QUESTION}"  -  time = afternoon', x=0.01, ha="left", fontsize=11, color=INK2)
    fig.savefig(os.path.join(RESULTS_DIR, "heatmap_pred_vs_truth.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    draw_heat(axes[0, 0], truth, "Ground truth")
    for k, (ax, p) in enumerate(zip(axes.flat[1:], preds)):
        im = draw_heat(ax, p, f"seed {k}")
    fig.colorbar(im, ax=axes, label="preference score", shrink=0.8, pad=0.02)
    fig.suptitle(f'"{QUESTION}"  -  time = afternoon, final from-scratch model per seed', x=0.01, ha="left",
                 fontsize=11, color=INK2)
    fig.savefig(os.path.join(RESULTS_DIR, "heatmap_pred_per_seed.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    stats = {"truth": heatmap_stats(truth), "mean_of_seeds": heatmap_stats(mean_pred),
             "per_seed": {str(s): heatmap_stats(p) for s, p in enumerate(preds)},
             "mean_abs_error_mean_pred": float(np.abs(mean_pred - truth).mean()),
             "pred_mean_grid": mean_pred.round(3).tolist(), "truth_grid": truth.round(3).tolist()}
    with open(os.path.join(RESULTS_DIR, "heatmap_stats.json"), "w") as f:
        json.dump(stats, f, indent=1)
    return stats


# ---------------------------------------------------------------- tables
def pm(g, col):
    return f"{g[col].mean():.4f} ± {g[col].std():.4f}"


def tables(df, stats):
    out = ["## Final round (350 samples), test.csv, mean ± std over 5 seeds\n",
           "| strategy | MSE | Spearman | pairwise acc |", "|---|---|---|---|"]
    for s in ("scratch", "warm"):
        g = df[(df.strategy == s) & (df["round"] == 7)]
        out.append(f"| {LABELS[s]} | {pm(g, 'mse')} | {pm(g, 'spearman')} | {pm(g, 'pairwise_acc')} |")
    out += ["\n## Stability, averaged over rounds 2-7 (mean ± std over 5 seeds of the per-seed round-average)\n",
            "| strategy | drift (MAE) | first-batch MSE | Kendall tau |", "|---|---|---|---|"]
    later = df[df["round"] >= 2].groupby(["strategy", "seed"])[["drift", "first_batch_mse", "kendall_tau"]].mean()
    for s in ("scratch", "warm"):
        g = later.loc[s]
        out.append(f"| {LABELS[s]} | {pm(g, 'drift')} | {pm(g, 'first_batch_mse')} | {pm(g, 'kendall_tau')} |")
    out += ["\n## Per-round means (5 seeds)\n"]
    cols = ["mse", "spearman", "pairwise_acc", "drift", "first_batch_mse", "kendall_tau"]
    t = df.groupby(["strategy", "round"])[cols].mean().round(4).reset_index()
    out.append("| strategy | round | n | " + " | ".join(cols) + " |")
    out.append("|" + "---|" * (len(cols) + 3))
    for _, r in t.iterrows():
        out.append(f"| {LABELS[r.strategy]} | {int(r['round'])} | {50 * int(r['round'])} | "
                   + " | ".join("-" if pd.isna(r[c]) else f"{r[c]:.4f}" for c in cols) + " |")
    out += ["\n## Heatmap statistics (fruit, afternoon; final from-scratch models)\n",
            "| | Sat mean | other-day mean | Sat / other | mean Spearman(valence, pred) over days | days fully monotonic (of 7) |",
            "|---|---|---|---|---|---|"]
    rows = [("truth", stats["truth"]), ("mean of 5 seeds", stats["mean_of_seeds"])]
    rows += [(f"seed {k}", v) for k, v in stats["per_seed"].items()]
    for name, v in rows:
        out.append(f"| {name} | {v['sat_mean']:.3f} | {v['other_days_mean']:.3f} | {v['sat_over_other']:.2f} | "
                   f"{v['mean_spearman_valence_vs_pred']:.3f} | {v['days_fully_monotonic']} |")
    out.append(f"\nEmotion order low -> high valence: {', '.join(stats['truth']['emotion_order_low_to_high'])}. "
               f"Mean |pred - truth| of the seed-mean grid: {stats['mean_abs_error_mean_pred']:.3f}.")
    with open(os.path.join(RESULTS_DIR, "summary_tables.md"), "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    df = pd.read_csv(METRICS_CSV)
    curve_figure(df, [("mse", "test MSE", "MSE (lower is better)"),
                      ("spearman", "Spearman", "Spearman (higher is better)"),
                      ("pairwise_acc", "pairwise accuracy", "Pairwise accuracy (higher is better)")],
                 "n_train", "training samples", "performance_curves.png",
                 "Performance on test.csv vs. training volume (mean ± std, 5 seeds)")
    curve_figure(df, [("drift", "mean |Δ prediction|", "Prediction drift vs. previous round"),
                      ("first_batch_mse", "MSE on first 50 samples", "Forgetting of the first batch"),
                      ("kendall_tau", "Kendall τ", "Ranking stability vs. previous round")],
                 "round", "retraining round", "stability_curves.png",
                 "Stability across incremental retraining rounds (mean ± std, 5 seeds)")
    stats = heatmaps()
    tables(df, stats)
    print("wrote figures + summary_tables.md in", RESULTS_DIR)


if __name__ == "__main__":
    main()
