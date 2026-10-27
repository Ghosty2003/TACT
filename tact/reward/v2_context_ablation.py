"""v2: context-input ablation and emotion-label noise (results/v2/context_ablation.md, context_ablation_runs.csv, context_noise_runs.csv).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.reward.v2_context_ablation

Both experiments re-run on the pipeline of the main experiment (data_dpo, reward model trained on the ask + act rows of the first N = 350 scenarios, seed `seed * 1000 + 350`,
early stopping on the validation scenarios, threshold decision rule), 5 seeds. The full-context / noise-0 model is therefore the main model of the v2 paper. (The v1 reports
`ablation_context.md` / `noise_robustness.md` used the ask-only training file and `replay.py`'s trainer; their numbers are not comparable with these.)
 (1) Context ablation: the embedding of one context field (emotion / time of day / day of week) is replaced by zeros (BertScorer.drop_ctx, passed on by the bilinear head); the context
     encoder and the bilinear head are unchanged. Evaluation on the 1512-cell noise-free grid of grid_eval.py (MSE, Spearman, pairwise accuracy; habit cells; the cells of each habit rule).
 (2) Emotion-label noise: the emotion of every TRAINING and VALIDATION scenario (both of its rows) is, independently with probability p in {0, 0.1, 0.2, 0.4, 0.6}, replaced by one of the 5 other
     emotions (uniformly; the flip is drawn once per (seed, p)). The scores are unchanged. Evaluation with the CLEAN emotion: grid MSE, Spearman and the regret on the 75 test scenarios.
     Extra columns: the exact expected grid MSE when the test-time emotion input is flipped at rate p as well, and the Bayes floor of a predictor that knows the truth but sees the flipped emotion.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.grid_eval import SHAPE, TOPICS, build_grid, grid_metrics
from tact.core.model import EMOTIONS, INPUT_COLS, RESULTS_DIR, FrozenBert, get_device, make_tensors
from tact.reward.noise_robustness import bayes_floor, expected_mse

N = 350
VARIANTS = {"full": (), "no_emotion": ("emotion",), "no_time": ("time",), "no_day": ("day",)}
LABEL = {"full": "full context", "no_emotion": "− emotion", "no_time": "− time of day", "no_day": "− day of week"}
PS = [0.0, 0.1, 0.2, 0.4, 0.6]
RULES = {"snack": "snack, Friday evening", "bread": "bread, Sunday morning", "coffee": "coffee, weekday mornings", "fruit": "fruit, Saturday"}


def flip(long_df, p, rng):
    """Flip the emotion of whole scenarios (the ask and the act row of an id carry the same emotion)."""
    ids = long_df.id.unique()
    emo = {i: e for i, e in zip(long_df.id, long_df.emotion)}
    out = {}
    for i in ids:
        e = EMOTIONS.index(emo[i])
        out[i] = EMOTIONS[(e + int(rng.integers(1, len(EMOTIONS)))) % len(EMOTIONS)] if rng.random() < p else EMOTIONS[e]
    return long_df.assign(emotion=long_df.id.map(out))


def train(data, seed, tr=None, va=None):
    B.MODEL_KW = {}
    ours = B.Ours(data, seed)
    old_val = data.rm_val
    if tr is not None:
        data.prefix = lambda n: (None, tr, None)
        data.rm_val = va
    try:
        ours.fit_reward(N, seed * 1000 + N, warm=False)
    finally:
        if tr is not None:
            del data.prefix
            data.rm_val = old_val
    return ours


def main():
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    grid, truth, habit = build_grid()
    topic_idx = np.repeat(np.arange(SHAPE[0]), int(np.prod(SHAPE[1:])))
    ts, f_ts = data.tests["seen"], data.f_test["seen"]
    bl = B.baselines(ts)
    T = truth.reshape(SHAPE)
    H = habit.reshape(SHAPE)[:, 0]
    rows, nrows = [], []

    def metrics(ours):
        pred = ours.rm.predict(grid)
        gm = grid_metrics(pred, truth, habit)
        r = {**gm}
        for t in RULES:
            m = habit & (topic_idx == TOPICS.index(t))
            r[f"habit_mse_{t}"] = float(np.mean((pred[m] - truth[m]) ** 2))
        a, _ = ours.decide(f_ts)
        r["test_regret"] = B.evaluate(a, ts)["regret"]
        return pred, r

    # ---------------- (1) context ablation
    for name, drop in VARIANTS.items():
        for seed in B.SEEDS:
            B.MODEL_KW = {"drop_ctx": drop}
            ours = B.Ours(data, seed)
            ours.fit_reward(N, seed * 1000 + N, warm=False)
            B.MODEL_KW = {}
            _, r = metrics(ours)
            rows.append({"variant": name, "seed": seed, **r})
        x = pd.DataFrame(rows)
        x = x[x.variant == name]
        print(f"[ablation] {name}: grid mse {x.mse.mean():.4f}, habit mse {x.habit_mse.mean():.4f}, snack-Fri mse {x.habit_mse_snack.mean():.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "context_ablation_runs.csv"), index=False)

    # ---------------- (2) emotion-label noise
    _, _, _ = None, None, None
    for p in PS:
        for seed in B.SEEDS:
            rng = np.random.default_rng([seed, int(round(p * 100)), 5])
            tr_df, va_df = flip(data.train_long, p, rng), flip(data.val_long, p, rng)
            tr = make_tensors(tr_df[INPUT_COLS + data.loc_cols], data.enc, dev)
            va = make_tensors(va_df[INPUT_COLS + data.loc_cols], data.enc, dev)
            ours = train(data, seed, tr, va)
            pred, r = metrics(ours)
            P = pred.reshape(SHAPE)
            r["mse_test_flipped"] = expected_mse(P, T, p)
            r["bayes_floor"] = bayes_floor(T, p)
            nrows.append({"p": p, "seed": seed, **r})
        x = pd.DataFrame(nrows)
        x = x[x.p == p]
        print(f"[noise] p={p}: grid mse {x.mse.mean():.4f}, spearman {x.spearman.mean():.3f}, test regret {x.test_regret.mean():.4f}", flush=True)
    pd.DataFrame(nrows).to_csv(os.path.join(RESULTS_DIR, "context_noise_runs.csv"), index=False)
    report(pd.DataFrame(rows), pd.DataFrame(nrows), bl)


def ms(x, nd=4):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def report(A, Nz, bl):
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    out = ["# v2: context ablation and emotion-label noise (`v2_context_ablation.py`)\n",
           "Final architecture (frozen NLI-BERT mean pooling + Context encoder + rank-16 bilinear head), threshold decision rule, N = 350 scenarios of `data_dpo` (reward model on the ask and act rows), "
           "5 seeds, mean ± std. Grid = the 1512-cell noise-free grid (12 topics × 6 emotions × 7 days × 3 times, `score_clean` formula; 60 cells are habit cells). **Deviation from the v1 reports:** they used the "
           "ask-only training file and the trainer of `replay.py`; these runs use the pipeline of the main experiment, so the 'full context' row is the main v2 model (grid MSE 0.0066, Spearman 0.911 in "
           "`paper_numbers_v2.md` §1) and the numbers are not comparable with `ablation_context.md` / `noise_robustness.md` of v1.\n",
           "## 1. Context ablation (the embedding of one context field replaced by zeros)\n",
           "| variant | grid MSE | grid Spearman | habit-cell MSE (60 cells) | **snack Friday evening, MSE (6 cells)** | non-habit MSE | test regret (75 scenarios) | grid MSE vs full (paired, mean ± SE) |", "|---|---|---|---|---|---|---|---|"]
    full = A[A.variant == "full"].sort_values("seed")
    for v in VARIANTS:
        x = A[A.variant == v].sort_values("seed")
        d = x.mse.values - full.mse.values
        out.append(f"| {LABEL[v]} | {ms(x.mse)} | {ms(x.spearman, 3)} | {ms(x.habit_mse)} | **{ms(x.habit_mse_snack)}** | {ms(x.nonhabit_mse)} | {ms(x.test_regret)} | "
                   f"{'–' if v == 'full' else f'{d.mean():+.4f} ± {se(d):.4f}'} |")
    out += ["\nMSE on the habit cells of each habit rule (cells = emotions × the days/times of the rule: snack 6, bread 6, fruit-Saturday 18, coffee 30; 60 in all):\n",
            "| variant | " + " | ".join(RULES.values()) + " |", "|---|" + "---|" * len(RULES)]
    for v in VARIANTS:
        x = A[A.variant == v]
        out.append(f"| {LABEL[v]} | " + " | ".join(ms(x[f"habit_mse_{t}"]) for t in RULES) + " |")
    out += ["\nPairwise accuracy and habit-cell Spearman:\n", "| variant | grid pairwise accuracy | habit-cell Spearman |", "|---|---|---|"]
    for v in VARIANTS:
        x = A[A.variant == v]
        out.append(f"| {LABEL[v]} | {ms(x.pairwise_acc, 3)} | {ms(x.habit_spearman, 3)} |")
    out += ["\n## 2. Emotion-label noise (training and validation emotion flipped with probability p)\n",
            f"Test baselines on the 75 test scenarios: always-ask regret {bl['always_ask']['regret']:.3f}, never-ask {bl['never_ask']['regret']:.3f}. 'Grid MSE' / 'Spearman' / 'test regret' are evaluated with the CLEAN "
            "emotion input. 'MSE with flipped test emotion' = exact expected grid MSE when the emotion input at test time is flipped at rate p too; 'Bayes floor' = the irreducible MSE of that setting.\n",
            "| flip rate p | grid MSE | grid Spearman | **test regret** | habit-cell MSE | snack Friday evening MSE | MSE with flipped test emotion | Bayes floor (flipped test emotion) |", "|---|---|---|---|---|---|---|---|"]
    for p in PS:
        x = Nz[Nz.p == p]
        out.append(f"| {p:g} | {ms(x.mse)} | {ms(x.spearman, 3)} | **{ms(x.test_regret)}** | {ms(x.habit_mse)} | {ms(x.habit_mse_snack)} | {ms(x.mse_test_flipped)} | {x.bayes_floor.mean():.4f} |")
    out += ["\nPaired change against p = 0 (same seeds; mean ± SE):\n", "| flip rate p | grid MSE | test regret |", "|---|---|---|"]
    z = Nz[Nz.p == 0].sort_values("seed")
    for p in PS[1:]:
        x = Nz[Nz.p == p].sort_values("seed")
        d1, d2 = x.mse.values - z.mse.values, x.test_regret.values - z.test_regret.values
        out.append(f"| {p:g} | {d1.mean():+.4f} ± {se(d1):.4f} | {d2.mean():+.4f} ± {se(d2):.4f} |")
    with open(os.path.join(RESULTS_DIR, "context_ablation.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
