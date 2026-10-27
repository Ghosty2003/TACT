"""v2 main results table (results/v2/main_table.csv, main_table.md): per N, 5 seeds, final architecture + threshold rule (TACT_V2=1).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_main_table

For each N in budget.NS (25 ... 350, including 75) and seed (the budget.py runs: `seed * 1000 + N`): decision quality of TACT (threshold rule) and of the
tabular agent on the 75 test scenarios (regret, decision agreement with the oracle = accuracy, clarification utility U = accuracy / (1 + ask rate)), and
reward-model quality: MSE against the observed (noisy) ask labels and Spearman / pairwise accuracy against the clean ask value on the test scenarios, and
MSE / Spearman / pairwise accuracy of the predicted ask score against the truth on the full 1512-cell grid (as grid_eval.py). Baselines (always-ask,
never-ask, oracle) are computed on the same test scenarios.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.core.evaluate import metrics_from_pred
from tact.core.grid_eval import build_grid, grid_metrics
from tact.core.model import RESULTS_DIR, FrozenBert, get_device


def util(ev):
    return ev["acc"] / (1.0 + (1.0 - ev["act_rate"]))


def main():
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    grid, truth, habit = build_grid()
    ts = data.tests["seen"]
    rows = []
    for n in B.NS:
        for seed in B.SEEDS:
            ours = B.Ours(data, seed)
            ours.fit_round(n, seed * 1000 + n, warm=False)
            a, _ = ours.decide(data.f_test["seen"])
            sc = data.sc_train.iloc[:n]
            tab = B.Tabular("topic")
            tab.update(sc, np.random.default_rng([seed, n]).permutation(n))
            c, _ = tab.decide(ts)
            with torch.no_grad():
                pa = B.reward_scores(ours.rm, data.f_test["seen"])[0].cpu().numpy()
            noisy = metrics_from_pred(pa, ts.s_ask.values)
            clean = metrics_from_pred(pa, ts.c_ask.values)
            gm = grid_metrics(ours.rm.predict(grid), truth, habit)
            for method, d in (("TACT", a), ("tabular", c)):
                ev = B.evaluate(d, ts)
                row = {"n": n, "seed": seed, "method": method, "regret": ev["regret"], "acc": ev["acc"], "habit_act": ev["habit_act"], "act_rate": ev["act_rate"],
                       "utility": util(ev)}
                if method == "TACT":
                    row.update({"test_mse_noisy": noisy["mse"], "test_mse_clean": clean["mse"], "test_spearman": clean["spearman"], "test_pairwise": clean["pairwise_acc"],
                                "grid_mse": gm["mse"], "grid_spearman": gm["spearman"], "grid_pairwise": gm["pairwise_acc"],
                                "habit_mse": gm["habit_mse"], "habit_spearman": gm["habit_spearman"]})
                rows.append(row)
        print(f"[main table] N={n} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "main_table.csv"), index=False)
    bl = B.baselines(ts)
    ms = lambda x, nd=3: f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"
    out = ["# v2 main results table (`v2_main_table.py`)\n",
           "Final architecture (frozen NLI-BERT mean pooling + Context encoder + rank-16 bilinear head), threshold decision rule; 75 test scenarios; mean ± std over 5 seeds. "
           "Decision agreement = accuracy against the oracle (ask iff the clean ask value ≥ the clean act value); clarification utility U = accuracy / (1 + ask rate). "
           "Reward-model columns: test = ask rows of the 75 test scenarios (MSE against the observed noisy labels; Spearman and pairwise accuracy against the clean value); "
           "grid = the 1512-cell grid against the truth.\n",
           "| N | TACT regret | TACT agreement | TACT utility | tabular regret | tabular agreement | tabular utility | test MSE (noisy) | test Spearman | test pairwise | grid MSE | grid Spearman | grid pairwise |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in B.NS:
        t, c = df[(df.n == n) & (df.method == "TACT")], df[(df.n == n) & (df.method == "tabular")]
        out.append(f"| {n} | {ms(t.regret)} | {ms(t.acc)} | {ms(t.utility)} | {ms(c.regret)} | {ms(c.acc)} | {ms(c.utility)} | {ms(t.test_mse_noisy, 4)} | {ms(t.test_spearman)} | "
                   f"{ms(t.test_pairwise)} | {ms(t.grid_mse, 4)} | {ms(t.grid_spearman)} | {ms(t.grid_pairwise)} |")
    out += ["\nBaselines on the same test scenarios:\n", "| | regret | agreement | utility |", "|---|---|---|---|"]
    for k, nm in (("always_ask", "always-ask"), ("never_ask", "never-ask"), ("oracle", "oracle")):
        out.append(f"| {nm} | {bl[k]['regret']:.4f} | {bl[k]['acc']:.3f} | {util(bl[k]):.3f} |")
    with open(os.path.join(RESULTS_DIR, "main_table.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
