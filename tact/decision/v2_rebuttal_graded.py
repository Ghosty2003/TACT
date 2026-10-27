"""Rebuttal experiment 3: graded feedback (results/v2/rebuttal/graded_feedback.md, graded_feedback_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_rebuttal_graded

How coarse may the feedback be? The continuous labels of the main experiment are replaced by K-level labels (K = 3: {0, .5, 1}; K = 5: {0, .25, .5, .75, 1}), for the reward model's training AND validation rows
(ask and act rows alike); the evaluation truth (score_clean) is untouched. N = 350 on the 500-scenario data (data_dpo, 75 test scenarios) and N = 700 on the 1000-scenario data (data_cont_n1000, 150 test scenarios), 5 seeds,
seed `seed * 1000 + N`, final architecture, threshold rule, early stopping on the (quantised) validation labels.
Two ways of making graded labels, both reported:
  direct      PRIMARY. The user's own noisy continuous rating of the main experiment (clip(clean + N(0, 0.05), 0, 1)) is rounded to the nearest of the K levels. Nothing random is added; this is what a K-point
              rating scale gives. It is a deterministic, biased coarsening (values inside one bin are indistinguishable).
  stochastic  unbiased: the NOISE-FREE value x = clean * (K - 1) is rounded to floor(x) or floor(x) + 1 with probability equal to its fractional part (E[label] = clean). For K = 2 this is exactly the Bernoulli
              feedback of the binary-feedback experiment, so direct / stochastic with K = 3, 5 and the two end points form one scale. The draw is made once per dataset (rng seed (size, K, 17), file order).
Two losses, both reported: mse (the main experiment's loss, PRIMARY: then the only change against the continuous model is the coarseness of the labels) and bce (the loss of the binary experiment; the sigmoid output is the
mean label).
End points for comparison: continuous (re-run here with the same code as an integrity check against binary_feedback_runs.csv) and binary (Bernoulli, BCE: taken from binary_feedback_runs.csv, draw 0 of data_bin* plus the two
extra draws of binary_feedback_draws_runs.csv).
Metrics: test regret on the data's own test_seen split, normalised full-grid regret (1512 cells; divided by min(always-ask, never-ask) on the grid), learning degree L of the Saturday-fruit habit
(L = (1 - collapse ratio) / (1 - 0.15), the collapse ratio of the afternoon fruit heatmap, as in the binary / drift experiments), agreement with the oracle on the test split.
"""
import copy
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tact.decision import budget as B
from tact.drift.habit_drift import collapse_ratio, truth_grid
from tact.core.model import INPUT_COLS, RESULTS_DIR, ROOT, FrozenBert, get_device, make_tensors

OUT = os.path.join(RESULTS_DIR, "rebuttal")
PLAN = [(500, 350, "data_dpo"), (1000, 700, "data_cont_n1000")]
KS = [3, 5]
MODES = ["direct", "stochastic"]
LOSSES = ["mse", "bce"]
TRUE_RATIO = 0.15


def bce(m, b):
    return F.binary_cross_entropy(m(b["cls"], b["emotion"], b["day"], b["time"], b.get("location"), b.get("extra")), b["y"])


class OursLoss(B.Ours):
    loss = "mse"

    def fit_reward(self, n, rnd_seed, warm):
        d, dev = self.d, self.d.dev
        _, rm_tr, _ = d.prefix(n)
        self.rm = B.new_scorer(d.enc, dev, rnd_seed, sigmoid=True)
        h = self._hp("reward")
        fn = B.reward_loss if self.loss == "mse" else bce
        B.fit(self.rm, fn, rm_tr, d.rm_val if self.val else None, lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed)


def quantise(data, K, mode, size):
    """Copy of `data` whose training and validation labels (ask and act rows) are K-level."""
    d = copy.copy(data)
    rng = np.random.default_rng([size, K or 0, 17])
    for name in ("train_long", "val_long"):
        df = getattr(data, name).copy()
        if K is None:
            new = df.score.values
        elif mode == "direct":
            new = np.floor(df.score.values * (K - 1) + 0.5) / (K - 1)
        else:
            x = df.score_clean.values * (K - 1)
            lo = np.floor(x)
            new = (lo + (rng.random(len(x)) < (x - lo))) / (K - 1)
        df["score"] = new
        setattr(d, name, df)
    d.sc_train, d.sc_val = B.scenarios(d.train_long), B.scenarios(d.val_long)
    d.rm_val = make_tensors(d.val_long[INPUT_COLS + d.loc_cols], d.enc, d.dev)
    return d


def evaluate(ours, data, g, f_grid, grid_base):
    ts, f = data.tests["seen"], data.f_test["seen"]
    a, _ = ours.decide(f)
    ev = B.evaluate(a, ts)
    ga, _ = ours.decide(f_grid)
    best = np.maximum(g.c_ask.values, g.c_act.values)
    greg = float((best - np.where(ga, g.c_ask.values, g.c_act.values)).mean())
    with torch.no_grad():
        pa = B.reward_scores(ours.rm, f_grid)[0].cpu().numpy()
    sat = collapse_ratio(pa, g)
    return {"regret_seen": ev["regret"], "acc_seen": ev["acc"], "grid_norm": greg / grid_base, "sat_ratio": sat, "L": (1 - sat) / (1 - TRUE_RATIO),
            "ask_rate_seen": float(a.mean())}


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = get_device()
    enc = FrozenBert(dev)
    datas = {size: B.Data(enc, dev, data_dir=os.path.join(ROOT, d)) for size, _, d in PLAN}
    g = truth_grid(False)
    f_grid = B.feats(g, enc, dev)
    best = np.maximum(g.c_ask.values, g.c_act.values)
    grid_base = min(float((best - g.c_ask.values).mean()), float((best - g.c_act.values).mean()))
    rows = []
    for size, n, _ in PLAN:
        configs = [("continuous", None, "-", "mse")] + [(f"K={K}", K, mode, loss) for mode in MODES for K in KS for loss in LOSSES]
        for label, K, mode, loss in configs:
            data = quantise(datas[size], K, mode, size)
            for seed in B.SEEDS:
                ours = OursLoss(data, seed)
                ours.loss = loss
                ours.fit_reward(n, seed * 1000 + n, warm=False)
                r = {"size": size, "n": n, "seed": seed, "label": label, "K": K or 0, "mode": mode, "loss": loss, **evaluate(ours, data, g, f_grid, grid_base)}
                r["n_levels_used"] = int(data.sc_train.iloc[:n].s_ask.nunique())
                rows.append(r)
            x = pd.DataFrame(rows)
            x = x[(x["size"] == size) & (x.label == label) & (x["mode"] == mode) & (x["loss"] == loss)]
            print(f"[N={n} {label} {mode} {loss}] regret {x.regret_seen.mean():.4f} grid {x.grid_norm.mean():.3f} L {x.L.mean():.3f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "graded_feedback_runs.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
