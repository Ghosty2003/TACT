"""v2 extensibility check (location as a 4th variable), final architecture only (results/v2/extensibility_*.csv, extensibility.md).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.robustness.v2_extensibility     # after budget.py (re-uses its 3-variable curve)

The experiment of extensibility.py restricted to the final architecture (no one-hot comparison): `data_ext` = data_dpo + location (office: receptiveness x 0.3),
the model gets an 8-d location embedding, N = 25..350 with 5 seeds, decision regret on test_seen of TACT (threshold rule), the tabular agent with and without the
location key, the reward model on the 4536-cell grid (predicted office / non-office ratio, truth 0.3) and the same pipeline on the 3-variable data as control.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.robustness.extensibility import CTX4, DATA_EXT, LOC_KW, build_grid4, office_ratio
from tact.core.grid_eval import build_grid, grid_metrics
from tact.core.model import RESULTS_DIR, ROOT, FrozenBert, get_device


def main():
    dev = get_device()
    enc = FrozenBert(dev)
    d3, d4 = B.Data(enc, dev), B.Data(enc, dev, data_dir=DATA_EXT)
    bl3, bl4 = B.baselines(d3.tests["seen"]), B.baselines(d4.tests["seen"])
    base3, base4 = (min(b["always_ask"]["regret"], b["never_ask"]["regret"]) for b in (bl3, bl4))
    g3, t3, h3 = build_grid()
    g4, t4, h4, off4 = build_grid4()
    office_test = (d4.tests["seen"].location == "office").values
    p3 = sum(p.numel() for p in B.new_scorer(enc, dev, 0, True).parameters())
    B.MODEL_KW = LOC_KW["bert"]
    p4 = sum(p.numel() for p in B.new_scorer(enc, dev, 0, True).parameters())
    dec_rows, grid_rows = [], []
    for n in B.NS:
        for seed in B.SEEDS:
            s = seed * 1000 + n
            B.MODEL_KW = LOC_KW["bert"]
            ours = B.Ours(d4, seed)
            ours.fit_round(n, s, warm=False)
            a, b = ours.decide(d4.f_test["seen"])
            tr_sc = d4.sc_train.iloc[:n]
            order = np.random.default_rng([seed, n]).permutation(n)
            tabs = {"C_noloc": B.Tabular("topic"), "C_loc": B.Tabular("topic", ctx_cols=CTX4)}
            for t in tabs.values():
                t.update(tr_sc, order)
            dec = {"A": (a, {}), "B": (b, {})}
            for name, t in tabs.items():
                d, lv = t.decide(d4.tests["seen"])
                dec[name] = (d, {"lvl_exact": lv[0], "lvl_topic": lv[1], "lvl_global": lv[2]})
            for m, (d, extra) in dec.items():
                dec_rows.append({"arch": "bert", "n": n, "seed": seed, "method": m, **B.evaluate(d, d4.tests["seen"]),
                                 "act_office": float((~d[office_test]).mean()), "act_other": float((~d[~office_test]).mean()), **extra})
            if n == 350:
                pred = ours.rm.predict(g4)
                grid_rows.append({"arch": "bert", "variant": "4var", "seed": seed, **grid_metrics(pred, t4, h4),
                                  "mse_office": float(np.mean((pred[off4] - t4[off4]) ** 2)), "mse_other": float(np.mean((pred[~off4] - t4[~off4]) ** 2)),
                                  "office_ratio": office_ratio(pred, h4, off4), "pred_office": float(pred[~h4 & off4].mean()), "pred_other": float(pred[~h4 & ~off4].mean())})
                B.MODEL_KW = {}
                c3 = B.Ours(d3, seed)
                c3.fit_reward(350, s, warm=False)
                grid_rows.append({"arch": "bert", "variant": "3var", "seed": seed, **grid_metrics(c3.rm.predict(g3), t3, h3)})
        print(f"[ext] N={n} done", flush=True)
    B.MODEL_KW = {}
    D, G = pd.DataFrame(dec_rows), pd.DataFrame(grid_rows)
    D.to_csv(os.path.join(RESULTS_DIR, "extensibility_runs.csv"), index=False)
    G.to_csv(os.path.join(RESULTS_DIR, "extensibility_grid.csv"), index=False)
    br = pd.read_csv(os.path.join(RESULTS_DIR, "budget_runs.csv"))
    ms = lambda x, nd=3: f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"
    out = ["# v2 extensibility: location as a 4th context variable (`v2_extensibility.py`)\n",
           f"Final architecture, threshold rule, 5 seeds. Trainable parameters (reward model): {p3:,} with 3 variables, {p4:,} with the 8-d location embedding. "
           f"Test baselines min(always-, never-ask): {base3:.4f} (3 variables), {base4:.4f} (4 variables).\n",
           "## Reward model on the 4536-cell grid, N = 350\n", "| quantity | 4 variables | 3 variables (control, 1512 cells) |", "|---|---|---|"]
    g4_, g3_ = G[G.variant == "4var"], G[G.variant == "3var"]
    out += [f"| grid MSE | {ms(g4_.mse, 4)} | {ms(g3_.mse, 4)} |", f"| grid Spearman | {ms(g4_.spearman)} | {ms(g3_.spearman)} |",
            f"| predicted office / non-office ask score (truth 0.30) | {ms(g4_.office_ratio)} | – |"]
    out += ["\n## Decisions on the test scenarios (regret, mean ± std)\n", "| N | TACT (4 variables) | tabular without location | tabular with location | share of office scenarios in which TACT acts | tabular with location: exact-key share |",
            "|---|---|---|---|---|---|"]
    for n in B.NS:
        a, c0, c1 = (D[(D.n == n) & (D.method == m)] for m in ("A", "C_noloc", "C_loc"))
        out.append(f"| {n} | {ms(a.regret, 4)} | {ms(c0.regret, 4)} | {ms(c1.regret, 4)} | {ms(a.act_office, 2)} | {c1.lvl_exact.mean():.2f} |")
    c3 = br[(br.method == "C") & (br.test == "seen") & (br.n == 350)]
    c4 = D[(D.method == "C_loc") & (D.n == 350)]
    a3 = br[(br.method == "A") & (br.test == "seen") & (br.n == 350)].sort_values("seed").regret
    a4 = D[(D.method == "A") & (D.n == 350)].sort_values("seed").regret
    out += ["\n## 3 → 4 variables at N = 350 (normalised regret = regret / min(always-, never-ask) of the same test set)\n", "| quantity | 3 variables | 4 variables |", "|---|---|---|",
            f"| TACT normalised regret | {a3.mean() / base3:.3f} | {a4.mean() / base4:.3f} |",
            f"| tabular normalised regret (with location key) | {c3.regret.mean() / base3:.3f} | {c4.regret.mean() / base4:.3f} |",
            f"| tabular: share of test scenarios with an exact key | {c3.lvl_exact.mean():.3f} | {c4.lvl_exact.mean():.3f} |"]
    with open(os.path.join(RESULTS_DIR, "extensibility.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
