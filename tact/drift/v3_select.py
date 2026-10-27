"""v3, step 1: half-life selection for recency-weighted replay, VALIDATION SEEDS ONLY (results/v3/select_*.csv, results/v3/select.md).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.drift.v3_select

Appendix-C drift stream (15 rounds x 50, Saturday-fruit habit stops after interaction 150) and the NO-DRIFT stream, final architecture, threshold rule, continuous labels, on 5 NEW seeds 5-9 (the reporting seeds are 0-4;
seeds only change the initialisation / minibatch order, the data stream is the one of Appendix C). Methods: W = the current default (warm start on all data, no weights) and R<h> = recency-weighted replay, warm start on all data,
weights 0.5 ** (age in rounds / h), unweighted validation loss, h in {1, 2, 3, 4, 6}.
Selection rule, fixed before any run: among the half-lives with (i) no-drift test regret not worse than the default's on the same seeds (one-sided: mean <= default mean) and (ii) retention L >= 0.9 (mean of the three unchanged habits
at round 15, drift stream), choose the one that minimises the drifted-cell regret averaged over the post-drift stream (mean over the interactions 151..750 of the regret on the 18 Saturday-fruit cells, new truth, then over seeds).
Reporting-seed (0-4) results of any half-life are not used.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.drift import drift_trigger as D
from tact.drift import drift_targeted as T
from tact.core.model import ROOT, FrozenBert, get_device

OUT = os.path.join(ROOT, "results", "v3")
SEEDS = [5, 6, 7, 8, 9]
METHODS = ["W", "R1", "R2", "R3", "R4", "R6"]


def main():
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "select_final.csv")
    have = set(map(tuple, pd.read_csv(path)[["run", "method", "seed"]].values.tolist())) if os.path.exists(path) else set()
    dev = get_device()
    enc = FrozenBert(dev)
    B.Data(enc, dev, data_dir=os.path.join(ROOT, "data_dpo"))
    ctx = D.Ctx(enc, dev)
    streams = {}
    for run in ("drift", "ctrl"):
        for m in METHODS:
            for seed in SEEDS:
                if (run, m, seed) in have:
                    continue
                if run not in streams:
                    streams[run] = D.Stream(ctx, "cont", run)
                t0 = time.perf_counter()
                series, rounds, _, fin = T.Sim(ctx, streams[run], seed, m, None).run()
                meta = {"run": run, "method": m, "seed": seed}
                T.append(os.path.join(OUT, "select_series.csv"), pd.DataFrame([{**meta, "i": i, **v} for i, v in series.items()]))
                T.append(os.path.join(OUT, "select_rounds.csv"), pd.DataFrame([{**meta, **x} for x in rounds]))
                T.append(path, pd.DataFrame([{**meta, **fin}]))
                print(f"[{run} {m} seed={seed}] fs_new(end)={series[T.T_END]['fs_new']:.3f} test={fin['test_regret']:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    report()


def report():
    S, R, F = (pd.read_csv(os.path.join(OUT, f"select_{n}.csv")) for n in ("series", "rounds", "final"))
    rows = []
    for m in METHODS:
        sd = S[(S.run == "drift") & (S.method == m) & (S.i > T.DRIFT_I)]
        obj = sd.groupby("seed").fs_new.mean()
        Ld = R[(R.run == "drift") & (R.method == m) & (R["round"] == 15)].sort_values("seed")
        L3 = Ld[[f"L_{h}" for h in T.HAB3]].mean(axis=1).values
        fc = F[(F.run == "ctrl") & (F.method == m)].sort_values("seed").test_regret.values
        fd = F[(F.run == "drift") & (F.method == m)].sort_values("seed").test_regret.values
        gnew = S[(S.run == "drift") & (S.method == m) & (S.i == T.T_END)].sort_values("seed").grid_new.values
        rows.append(dict(method=m, obj=obj.values, L=L3, nd=fc, dr=fd, grid=gnew))
    w = rows[0]
    ms = lambda x, nd=4: f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"
    out = ["# v3 step 1: half-life selection on validation seeds 5–9 (`v3_select.py`)\n",
           "Appendix-C drift stream and no-drift stream, final architecture, threshold rule, 5 NEW seeds (5–9; the reporting seeds 0–4 are not used here). The data stream is the one of Appendix C (the seeds change initialisation and minibatch order only). "
           "W = current default (warm start on all data, unweighted); R<h> = recency-weighted replay with half-life h rounds (weights 0.5^(age/h), warm start, unweighted validation loss, early stopping as before).\n",
           "**Rule fixed in advance:** among half-lives with (i) no-drift test regret not worse than the default's on the same seeds (one-sided, mean ≤ default mean) and (ii) retention L ≥ 0.9, choose the minimiser of the drifted-cell regret averaged over the post-drift stream "
           "(interactions 151–750, 18 Saturday-fruit cells, new truth).\n",
           "**Disclosure.** Before this step I had already seen reporting-seed (0–4) results for half-lives 2 and 4 in the targeted-reset experiment (`results/drift_targeted.md`). They were not used to define the grid or the rule, "
           "and the choice below is computed from seeds 5–9 only; the reader should know that half-lives 2 and 4 were in my mind as candidates.\n",
           "## Validation table (seeds 5–9, mean ± std)\n",
           "| method | drifted-cell regret, mean over the post-drift stream | NO-DRIFT test regret | retention L (round 15) | drift: final test regret | drift: final grid regret (new truth) | (i) no-drift ≤ default | (ii) L ≥ 0.9 | feasible |", "|---|---|---|---|---|---|---|---|---|"]
    feas = {}
    for r in rows:
        i_ok = r["nd"].mean() <= w["nd"].mean() + 1e-12 if r["method"] != "W" else True
        ii_ok = r["L"].mean() >= 0.9
        feas[r["method"]] = i_ok and ii_ok and r["method"] != "W"
        name = "W (default, unweighted)" if r["method"] == "W" else f"R{r['method'][1:]} (half-life {r['method'][1:]} rounds)"
        out.append(f"| {name} | {ms(r['obj'])} | {ms(r['nd'])} | {np.mean(r['L']):.2f} ± {np.std(r['L'], ddof=1):.2f} | {ms(r['dr'])} | {ms(r['grid'])} | {'yes' if i_ok else 'no'} | {'yes' if ii_ok else 'no'} | {'yes' if feas[r['method']] else ('(baseline)' if r['method'] == 'W' else 'no')} |")
    cand = [r for r in rows if feas[r["method"]]]
    if not cand:
        out.append("\n**No half-life is feasible; the rule gives no choice.**")
        chosen = None
    else:
        best = min(cand, key=lambda r: r["obj"].mean())
        chosen = int(best["method"][1:])
        out.append(f"\n**Chosen half-life: {chosen} rounds** (feasible, lowest drifted-cell regret over the post-drift stream: {ms(best['obj'])}; the default's is {ms(w['obj'])}).")
        # sensitivity of the choice to the tolerance of condition (i)
        tol = [r for r in rows if r["method"] != "W" and r["nd"].mean() <= w["nd"].mean() + w["nd"].std(ddof=1) and r["L"].mean() >= 0.9]
        alt = min(tol, key=lambda r: r["obj"].mean())["method"]
        out.append(f"Sensitivity of the choice: with the looser reading of (i) (mean ≤ default mean + 1 std = {w['nd'].mean() + w['nd'].std(ddof=1):.4f}) the minimiser would be {alt} " + ("(the same)." if alt == best["method"] else "(DIFFERENT; the literal rule above was followed)."))
        # paired comparison to the default
        d = best["obj"] - w["obj"]
        out.append(f"Paired by seed against the default: Δ drifted-cell regret {d.mean():+.4f} ± SE {d.std(ddof=1) / np.sqrt(len(d)):.4f} ({int((d < 0).sum())}/5 seeds better).")
    with open(os.path.join(OUT, "select.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    json.dump({"half_life": chosen}, open(os.path.join(OUT, "chosen_half_life.json"), "w"))
    print("\n".join(out))


if __name__ == "__main__":
    main()
