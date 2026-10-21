"""Emotion-input noise robustness: does the Context MLP degrade more slowly than one-hot when the emotion
recognition is wrong?

    python -m tact.reward.noise_robustness              # each architecture's val-selected config from hparam_fairness.py
    python -m tact.reward.noise_robustness --defaults   # spec-default hyper-parameters (robustness check)

For each flip rate p in {0, 0.2, 0.4, 0.6} the emotion input of every training row and every validation row is,
independently with probability p, replaced by one of the 5 OTHER emotions (uniformly). The label `score` is
unchanged. The flip is drawn once per (seed, p) and shared by the two architectures, so the comparison is paired.
Models are trained from scratch on the 350 corrupted rows (seed, early stopping as in replay.py).

Evaluation is on the noise-free 1512-cell grid (truth = score_clean formula, see grid_eval.py):
  * MSE with the *test-time* emotion input flipped at rate p too (the setting asked for) - computed as the exact
    expectation over the flips, from the model's predictions for all 6 emotion inputs (no sampling noise);
  * MSE with a clean test-time emotion (trained on noisy emotion only), to separate train-time from test-time noise.
A Bayes-optimal reference (knows the truth function, sees only the flipped emotion) gives the irreducible floor.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel

from tact.reward.grid_eval import ARCH_NAME, SHAPE, build_grid, grid_metrics, pm
from tact.core.model import EMOTIONS, MODELS, RESULTS_DIR, FrozenBert, get_device, load_csv, make_tensors
from tact.reward.replay import HPARAMS, run_seed, train
from tact.core import plots  # shared plot style

PS = [0.0, 0.2, 0.4, 0.6]
E = len(EMOTIONS)


def flip_emotions(df, p, rng):
    """Replace each row's emotion, w.p. p, by a uniformly chosen different emotion."""
    emo = df["emotion"].map({e: i for i, e in enumerate(EMOTIONS)}).values.copy()
    flip = rng.random(len(df)) < p
    emo[flip] = (emo[flip] + rng.integers(1, E, flip.sum())) % E   # +1..+5 mod 6 = any other class
    return df.assign(emotion=[EMOTIONS[i] for i in emo])


def expected_mse(P, T, q, mask=None):
    """Exact expected grid MSE when the test emotion input is flipped at rate q.
    P[t, e', d, m]: prediction for input emotion e'; T[t, e, d, m]: truth for the true emotion e."""
    W = np.full((E, E), q / (E - 1))
    np.fill_diagonal(W, 1 - q)                                           # W[e, e'] = P(input e' | true e)
    D = (P[:, None, :, :, :] - T[:, :, None, :, :]) ** 2                 # [t, e, e', d, m]
    per_cell = np.einsum("tegdm,eg->tedm", D, W)                          # [t, e, d, m]
    if mask is not None:                                                 # mask over (t, d, m)
        per_cell = per_cell[np.broadcast_to(mask[:, None], per_cell.shape)]
    return float(per_cell.mean())


def bayes_floor(T, q, mask=None):
    """Irreducible MSE for a predictor that knows the truth but only sees the flipped emotion (uniform prior)."""
    W = np.full((E, E), q / (E - 1))
    np.fill_diagonal(W, 1 - q)
    O = np.einsum("tedm,eg->tgdm", T, W)       # O[t, g, d, m] = sum_e P(e | input g) * T[e]  (W symmetric)
    return expected_mse(O, T, q, mask)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--defaults", action="store_true", help="use the spec-default hyper-parameters instead of "
                                                           "results/hparam_selected.json")
    ap.add_argument("--plot-only", action="store_true", help="skip training; rebuild figure/tables from the saved csv")
    args = ap.parse_args()
    tag = "_defaults" if args.defaults else ""

    configs = {a: {"lr": HPARAMS["lr"], "kw": {}} for a in ("bert", "onehot")}
    if not args.defaults:
        with open(os.path.join(RESULTS_DIR, "hparam_selected.json")) as f:
            for a, c in json.load(f).items():
                kw = {k: tuple(v) if isinstance(v, list) else v for k, v in c["kwargs"].items()}
                configs[a] = {"lr": c["lr"], "kw": kw}

    grid, truth, habit = build_grid()
    T = truth.reshape(SHAPE)
    H = habit.reshape(SHAPE)[:, 0]             # habit mask over (t, d, m); independent of emotion

    rows = []
    if not args.plot_only:
        device = get_device()
        encoder = FrozenBert(device)
        train_df, val_df = load_csv("train.csv"), load_csv("val.csv")
    for p in ([] if args.plot_only else PS):
        for seed in HPARAMS["seeds"]:
            rng_tr = np.random.default_rng([seed, int(p * 100), 0])
            rng_va = np.random.default_rng([seed, int(p * 100), 1])
            tr_df, va_df = flip_emotions(train_df, p, rng_tr), flip_emotions(val_df, p, rng_va)
            tr, va = make_tensors(tr_df, encoder, device), make_tensors(va_df, encoder, device)
            for arch in ("bert", "onehot"):
                hp = {**HPARAMS, "lr": configs[arch]["lr"]}
                s = run_seed(seed, 7)
                torch.manual_seed(s)
                np.random.seed(s)
                model = MODELS[arch](encoder, **configs[arch]["kw"]).to(device)
                train(model, tr, va, hp, gen_seed=s)
                P = model.predict(grid).reshape(SHAPE)
                if p == 0:   # sanity: expectation formula reduces to the plain grid MSE
                    assert abs(expected_mse(P, T, 0.0) - grid_metrics(P.ravel(), truth, habit)["mse"]) < 1e-9
                rows.append({"arch": arch, "p": p, "seed": seed,
                             "mse_test_noisy": expected_mse(P, T, p), "mse_test_clean": expected_mse(P, T, 0.0),
                             "habit_mse_test_noisy": expected_mse(P, T, p, H),
                             "habit_mse_test_clean": expected_mse(P, T, 0.0, H)})
                print(f"p={p:.1f} seed={seed} {arch:6s} mse(noisy test)={rows[-1]['mse_test_noisy']:.4f} "
                      f"mse(clean test)={rows[-1]['mse_test_clean']:.4f}", flush=True)
    csv_path = os.path.join(RESULTS_DIR, f"noise_robustness{tag}.csv")
    if args.plot_only:
        df = pd.read_csv(csv_path)
    else:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    floor = {p: bayes_floor(T, p) for p in PS}

    # ------------------------------------------------------------ figure
    COL = {"bert": "#2a78d6", "onehot": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, col, ttl in ((axes[0], "mse_test_noisy", "train and test emotion flipped at rate p"),
                         (axes[1], "mse_test_clean", "train emotion flipped at rate p, clean test emotion")):
        for a in ("bert", "onehot"):
            g = df[df.arch == a].groupby("p")[col].agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=COL[a], lw=2, marker="o", ms=6, label=ARCH_NAME[a])
            ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COL[a], alpha=0.15, lw=0)
        if col == "mse_test_noisy":
            ax.plot(PS, [floor[p] for p in PS], color=plots.INK2, ls="--", lw=1.2, label="Bayes floor (knows the truth)")
        ax.set_xticks(PS)
        ax.set_xlabel("emotion flip rate p")
        ax.set_title(ttl, fontsize=10, loc="left", color=plots.INK)
    axes[0].set_ylabel("full-grid MSE vs. score_clean")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle("Emotion-noise robustness (from-scratch, 350 rows, mean ± std over 5 seeds"
                 + (", spec-default hyper-parameters" if args.defaults else ", val-selected hyper-parameters") + ")",
                 x=0.01, ha="left", fontsize=11, color=plots.INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"noise_robustness{tag}.png"), dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ tables
    out = [f"# Emotion-noise robustness{' (spec-default hyper-parameters)' if args.defaults else ' (val-selected hyper-parameters)'}\n",
           "Config: " + "; ".join(f"{ARCH_NAME[a]} lr={configs[a]['lr']:g} {configs[a]['kw'] or 'default size'}"
                                 for a in configs) + ". From-scratch, 350 corrupted rows, 5 seeds; the same flips are "
           "used for both architectures (paired by seed). Full-grid MSE against score_clean; test-time flips are "
           "handled by exact expectation.\n",
           "## Full-grid MSE (train and test emotion flipped at rate p)\n",
           "| p | BertScorer | OneHotScorer | Bayes floor | Bert − OneHot (paired mean) | seeds where Bert is better | paired t p |",
           "|---|---|---|---|---|---|---|"]

    def piv(col):
        return df.pivot(index=["p", "seed"], columns="arch", values=col)

    for col, title in (("mse_test_noisy", None), ("mse_test_clean", "## Same models, clean test emotion\n")):
        w = piv(col)
        if title:
            out += ["\n" + title, "| p | BertScorer | OneHotScorer | Bert − OneHot (paired mean) | seeds where Bert is better | paired t p |",
                    "|---|---|---|---|---|---|"]
        for p in PS:
            x = w.loc[p]
            diff = x["bert"] - x["onehot"]
            tp = ttest_rel(x["bert"], x["onehot"]).pvalue if diff.std() > 0 else float("nan")
            fl = f"{floor[p]:.4f} | " if col == "mse_test_noisy" else ""
            out.append(f"| {p:.1f} | {pm(x['bert'])} | {pm(x['onehot'])} | {fl}{diff.mean():+.4f} | "
                       f"{int((diff < 0).sum())}/5 | {tp:.3f} |")
    out += ["\n## Degradation relative to p = 0 (per seed, then mean ± std), train and test noisy\n",
            "Δ(p) = MSE(p) − MSE(0). A slower degradation of BertScorer would show as a smaller Δ.\n",
            "| p | Δ BertScorer | Δ OneHotScorer | Δ Bert − Δ OneHot (paired mean) | seeds where Bert degrades less | paired t p |",
            "|---|---|---|---|---|---|"]
    w = piv("mse_test_noisy")
    base = w.loc[0.0]
    for p in PS[1:]:
        d = w.loc[p] - base
        dd = d["bert"] - d["onehot"]
        out.append(f"| {p:.1f} | {pm(d['bert'])} | {pm(d['onehot'])} | {dd.mean():+.4f} | {int((dd < 0).sum())}/5 | "
                   f"{ttest_rel(d['bert'], d['onehot']).pvalue:.3f} |")
    out += ["\n## Habit cells only (60 cells), train and test noisy\n",
            "| p | BertScorer | OneHotScorer | Bert − OneHot (paired mean) |", "|---|---|---|---|"]
    wh = piv("habit_mse_test_noisy")
    for p in PS:
        x = wh.loc[p]
        out.append(f"| {p:.1f} | {pm(x['bert'])} | {pm(x['onehot'])} | {(x['bert'] - x['onehot']).mean():+.4f} |")
    with open(os.path.join(RESULTS_DIR, f"noise_robustness{tag}.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
