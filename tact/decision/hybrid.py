"""Hybrid decision rule: use the tabular topic-mean decision (method C) where the training data gives too little support,
otherwise use the learned policy A.   python -m tact.decision.hybrid   (after budget.py)   -> results/hybrid_*.{csv,md,png}

Support of a test scenario = number of training scenarios (among the first N) with the same topic whose
(emotion, day, time) differs in at most one coordinate (Hamming distance <= 1; includes exact matches).
Rule: ask = C's decision if support < k else A's decision; k = 0 means always A, k = infinity always C.
k is chosen per (N, seed) from {0, 1, 2, 3, 4, 6, 8, 12, inf} to minimise the OBSERVED (noisy-label) regret on the 75
validation scenarios (ties -> smaller k). No clean labels and no test data are used. A has early-stopped on the same
validation scenarios, so its validation regret is optimistic, which biases the selection towards A.
Models are re-trained exactly as in budget.py (same seeds), so A and C equal the budget.py results.
"""
import os

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.model import RESULTS_DIR

KS = [0, 1, 2, 3, 4, 6, 8, 12, 10 ** 9]


def support(train_sc, sc):
    emap = {e: i for i, e in enumerate(sorted(set(train_sc.emotion) | set(sc.emotion)))}
    tr = np.stack([train_sc.emotion.map(emap).values, train_sc.day.values, train_sc.time.values], 1)
    te = np.stack([sc.emotion.map(emap).values, sc.day.values, sc.time.values], 1)
    out = np.zeros(len(sc), int)
    for i in range(len(sc)):
        same_topic = train_sc.topic.values == sc.topic.values[i]
        out[i] = int((same_topic & ((tr != te[i]).sum(1) <= 1)).sum())
    return out


def observed_regret(ask, sc):
    s_ask, s_act = sc.s_ask.values, sc.s_act.values
    return float((np.maximum(s_ask, s_act) - np.where(ask, s_ask, s_act)).mean())


def main():
    dev = B.get_device()
    data = B.Data(B.FrozenBert(dev), dev)
    bl = {ts: B.baselines(sc) for ts, sc in data.tests.items()}
    rows = []
    for n in B.NS:
        for seed in B.SEEDS:
            ours = B.Ours(data, seed)
            ours.fit_round(n, seed * 1000 + n, warm=False)
            tr_sc = data.sc_train.iloc[:n]
            tab = B.Tabular("topic")
            tab.update(tr_sc, np.random.default_rng([seed, n]).permutation(n))
            a_val, _ = ours.decide(data.f_val)
            c_val, _ = tab.decide(data.sc_val)
            sup_val = support(tr_sc, data.sc_val)
            val_reg = [observed_regret(np.where(sup_val < k, c_val, a_val), data.sc_val) for k in KS]
            k = KS[int(np.argmin(val_reg))]            # first minimum = smallest k
            for ts in ("seen", "unseen"):
                sc, f = data.tests[ts], data.f_test[ts]
                a, _ = ours.decide(f)
                c, _ = tab.decide(sc)
                sup = support(tr_sc, sc)
                h = np.where(sup < k, c, a)
                for m, d in (("A", a), ("C", c), ("H", h)):
                    rows.append({"n": n, "seed": seed, "method": m, "test": ts, **B.evaluate(d, sc), "k": k if m == "H" else np.nan,
                                 "fallback_share": float((sup < k).mean()) if m == "H" else np.nan})
            print(f"[hybrid] N={n} seed={seed} k={k if k < 10 ** 8 else 'inf'}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "hybrid_runs.csv"), index=False)

    # integrity vs budget.py (A and C must reproduce)
    br = pd.read_csv(os.path.join(RESULTS_DIR, "budget_runs.csv"))
    chk = df[df.method.isin(["A", "C"])].merge(br, on=["n", "seed", "method", "test"], suffixes=("", "_ref"))
    max_diff = float((chk.regret - chk.regret_ref).abs().max())

    base = {ts: min(b["always_ask"]["regret"], b["never_ask"]["regret"]) for ts, b in bl.items()}
    mean = df.groupby(["test", "method", "n"]).regret.mean().unstack("n")
    sd = df.groupby(["test", "method", "n"]).regret.std().unstack("n")
    out = ["# Hybrid fallback rule (`hybrid.py`)\n",
           "Rule: ask = C's topic-mean decision if the scenario's support in the first N training scenarios is below k, "
           "else A's decision. Support = number of training scenarios with the same topic whose (emotion, day, time) differs "
           "in at most one coordinate. k ∈ {0, 1, 2, 3, 4, 6, 8, 12, ∞} is chosen per (N, seed) on the validation scenarios by "
           "observed (noisy-label) regret, ties to the smaller k; A has early-stopped on the same validation set, which "
           f"favours A in this selection. A and C reproduce budget.py (max abs regret difference {max_diff:.1e}).\n"]
    verdict = {}
    for ts in ("seen", "unseen"):
        out += [f"## test_{ts}: regret by N (mean ± std over 5 seeds)\n",
                "| N | A | C | H (hybrid) | min(A, C) | H ≤ min(A, C)? | H ≤ max(A, C)? | selected k (median, range) | share of test scenarios using C |",
                "|---|---|---|---|---|---|---|---|---|"]
        strict, weak = [], []
        for n in B.NS:
            a, c, h = mean.loc[(ts, "A"), n], mean.loc[(ts, "C"), n], mean.loc[(ts, "H"), n]
            hs = df[(df.method == "H") & (df.test == ts) & (df.n == n)]
            ks = hs.k.values
            kk = lambda x: "∞" if x >= 10 ** 8 else f"{x:.0f}"
            strict.append(h <= min(a, c) + 1e-12)
            weak.append(h <= max(a, c) + 1e-12)
            out.append(f"| {n} | {a:.3f} ± {sd.loc[(ts, 'A'), n]:.3f} | {c:.3f} ± {sd.loc[(ts, 'C'), n]:.3f} | "
                       f"**{h:.3f}** ± {sd.loc[(ts, 'H'), n]:.3f} | {min(a, c):.3f} | {'yes' if strict[-1] else 'no'} | "
                       f"{'yes' if weak[-1] else 'no'} | {kk(np.median(ks))} ({kk(ks.min())}–{kk(ks.max())}) | "
                       f"{hs.fallback_share.mean():.0%} |")
        verdict[ts] = (all(strict), all(weak), sum(strict))
        out.append(f"\nBaselines: always-ask {bl[ts]['always_ask']['regret']:.3f}, never-ask {bl[ts]['never_ask']['regret']:.3f}.\n")
    # paired H - min(A,C) across seeds
    out += ["## Paired difference H − better of (A, C), seed by seed\n", "| N | seen | unseen |", "|---|---|---|"]
    for n in B.NS:
        cells = []
        for ts in ("seen", "unseen"):
            p = df[(df.test == ts) & (df.n == n)].pivot(index="seed", columns="method", values="regret")
            d = p.H - p[["A", "C"]].min(axis=1)
            cells.append(f"{d.mean():+.3f} ± {d.std(ddof=1):.3f}")
        out.append(f"| {n} | {cells[0]} | {cells[1]} |")
    ok = verdict["seen"][0] and verdict["unseen"][0]
    out += ["\n## Criterion\n",
            "“Not worse than max(A, C)” is read as: at every N the hybrid's mean regret is no larger than that of the better of "
            "A and C (a weaker reading, no larger than the worse of the two, is shown as well).\n",
            f"- seen: strict criterion holds at {verdict['seen'][2]} of {len(B.NS)} N values; weak reading holds everywhere: {verdict['seen'][1]}.",
            f"- unseen: strict criterion holds at {verdict['unseen'][2]} of {len(B.NS)} N values; weak reading holds everywhere: {verdict['unseen'][1]}.",
            f"- Strict criterion met on both test sets at every N: **{'yes' if ok else 'no'}**" +
            (" -> the hybrid is added to the main figure (`budget_curves.png`)." if ok else
             " -> the main figure `budget_curves.png` is left unchanged; `hybrid_curves.png` shows the comparison.")]
    with open(os.path.join(RESULTS_DIR, "hybrid_summary.md"), "w") as f:
        f.write("\n".join(out) + "\n")

    # figures
    h_rows = df[df.method == "H"]
    B.curves_figure(pd.concat([br[br.method.isin(["A", "C"])], h_rows]), bl, "hybrid_curves.png", "n", B.NS,
                    "Hybrid (C's topic mean where support < k, else A) vs. A and C (mean ± std over 5 seeds)",
                    ["A", "C", "H"], "training scenarios N", True)
    if ok:
        B.curves_figure(pd.concat([br, h_rows]), bl, "budget_curves.png", "n", B.NS,
                        "Regret vs. number of training scenarios (mean ± std over 5 seeds; N counts scenarios)",
                        ["A", "B", "C", "Ct", "H"], "training scenarios N", True)
    print("\n".join(out))


if __name__ == "__main__":
    main()
