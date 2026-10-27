"""Report for drift_trigger.py (results/drift_trigger.md, results/drift_trigger.png)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tact.core import plots
from tact.core.model import RESULTS_DIR

PER, DRIFT_I, T_END = 50, 150, 750
TARGET = 0.05
MAIN = "d0.15_k2"
SERIES = os.path.join(RESULTS_DIR, "drift_trigger_series.csv")
ROUNDS = os.path.join(RESULTS_DIR, "drift_trigger_rounds.csv")
EVENTS = os.path.join(RESULTS_DIR, "drift_trigger_events.csv")
HAB3 = ("bread", "coffee", "snack")
NAME = {"W": "warm start (paper default)", "S": "from scratch, last 3 rounds", "C": "tabular (online)", "Cb": "tabular (per round)",
        "N": "warm start + local relearning"}
COL = {"W": "#6b7280", "S": "#eb6834", "C": "#1baf7a", "Cb": "#8bc9a8", "N": "#2a78d6"}


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 else (f"{x.mean():.{nd}f}" if len(x) else "n/a")


def get(S, label, run, method, cfg="-"):
    return S[(S.label == label) & (S.run == run) & (S.method == method) & (S.cfg == cfg)]


def wide(df, col):
    """seed x i matrix of a series column."""
    return df.pivot(index="seed", columns="i", values=col)


def reach(df, col="fs_new", thr=TARGET):
    """per seed: (first interaction since the drift with regret < thr, first interaction from which it stays below thr); None if never."""
    w = wide(df, col)
    first, stable = {}, {}
    for seed, row in w.iterrows():
        r = row[row.index > DRIFT_I]
        below = r < thr
        first[seed] = int(r.index[below.values][0] - DRIFT_I) if below.any() else None
        st = None
        for i in r.index[::-1]:
            if r[i] < thr:
                st = int(i - DRIFT_I)
            else:
                break
        stable[seed] = st
    return first, stable


def censored(d, cap=T_END - DRIFT_I + 1):
    return np.array([cap if v is None else v for v in d.values()], float)


def summarize_reach(d):
    ok = [v for v in d.values() if v is not None]
    return f"{len(ok)} of {len(d)}" + (f" (mean {np.mean(ok):.0f}, median {np.median(ok):.0f}, range {min(ok)}–{max(ok)})" if ok else "")


def round_end(df, col, i):
    return df[df.i == i].sort_values("seed")[col].values


def retention(S, R, label, run, method, cfg="-"):
    df = get(S, label, run, method, cfg)
    end = df[df.i == T_END].sort_values("seed")
    reg = end[[f"h_{t}" for t in HAB3]].mean(axis=1).values
    worst = df.groupby("seed")[[f"h_{t}" for t in HAB3]].max().mean(axis=1).values
    rr = R[(R.label == label) & (R.run == run) & (R.method == method) & (R.cfg == cfg) & (R["round"] == 15)].sort_values("seed")
    L = rr[[f"L_{t}" for t in HAB3]].mean(axis=1).values
    return reg, worst, L, end, rr


def class_geometry():
    path = os.path.join(RESULTS_DIR, "drift_trigger_class_geometry.json")
    if os.path.exists(path):
        return json.load(open(path))
    import torch
    import torch.nn.functional as F
    from tact.core.generate_dataset import QUESTIONS
    from tact.core.model import FrozenBert
    enc = FrozenBert(torch.device("cpu"))
    topics = list(QUESTIONS)
    texts = [QUESTIONS[t][k] for t in topics for k in (0, 1)]
    v = F.normalize(enc.encode(texts), dim=1)
    cos = (v @ v.T).numpy()
    idx = [(i, j) for i in range(len(texts)) for j in range(i + 1, len(texts))]
    same = [cos[i, j] for i, j in idx if i // 2 == j // 2]
    diff = [cos[i, j] for i, j in idx if i // 2 != j // 2]
    g = {"same_n": len(same), "same_min": float(min(same)), "same_gt": float(np.mean(np.array(same) > 0.9)),
         "diff_n": len(diff), "diff_min": float(min(diff)), "diff_mean": float(np.mean(diff)), "diff_gt": float(np.mean(np.array(diff) > 0.9))}
    json.dump(g, open(path, "w"))
    return g


def block(S, R, E, label, h, cfgs):
    """Tables for one label type; returns (lines, numbers for the criterion)."""
    out = []
    methods = [("W", "-"), ("S", "-"), ("C", "-"), ("Cb", "-"), ("N", MAIN)]
    # ---- adaptation
    out += [f"{h} Adaptation speed: interactions after the drift until the Saturday-fruit cells' regret is below {TARGET}\n",
            f"Regret of the policy's decisions on the 18 Saturday-fruit grid cells against the NEW truth (oracle of the drifted world); the drift starts "
            f"at interaction 151. 'First' = first interaction (counted from the drift) at which the mean regret over those cells is < {TARGET}; 'stable' = from which "
            "it stays below until the end (interaction 750). Methods that retrain only at round ends can only change every 50 interactions; the tabular agent "
            "(online) and the override of the new method change after every interaction. A seed that never gets there counts as 601 in the means used by the criterion.\n",
            "| method | seeds reaching it (first) | first, mean over seeds (never = 601) | seeds with stable < 0.05 | stable, mean (never = 601) |", "|---|---|---|---|---|"]
    reach_res = {}
    for m, cfg in methods:
        df = get(S, label, "drift", m, cfg)
        if df.empty:
            continue
        f, st = reach(df)
        reach_res[m] = (f, st)
        out.append(f"| {NAME[m]} | {summarize_reach(f)} | {censored(f).mean():.0f} | {summarize_reach(st)} | {censored(st).mean():.0f} |")
    out += ["", "Regret on the 18 Saturday-fruit cells (new truth), mean ± std over 5 seeds, at selected numbers of interactions after the drift:\n",
            "| interactions after the drift | " + " | ".join(NAME[m] for m, c in methods if not get(S, label, 'drift', m, c).empty) + " |",
            "|---|" + "---|" * sum(not get(S, label, 'drift', m, c).empty for m, c in methods)]
    for k in (1, 2, 5, 10, 25, 49, 50, 100, 150, 300, 600):
        i = DRIFT_I + k
        cells = [ms(round_end(get(S, label, "drift", m, c), "fs_new", i)) for m, c in methods if not get(S, label, "drift", m, c).empty]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    out.append("")
    # ---- full grid regret over rounds
    out += [f"{h} Full-grid regret over the run (drift run, against the truth of the moment; mean ± std over 5 seeds)\n",
            "| interaction (round end) | " + " | ".join(NAME[m] for m, c in methods if not get(S, label, 'drift', m, c).empty) + " |",
            "|---|" + "---|" * sum(not get(S, label, 'drift', m, c).empty for m, c in methods)]
    for r in range(2, 16):
        i = PER * r
        col = "grid_old" if i <= DRIFT_I else "grid_new"
        out.append(f"| {i} (round {r}{', drift starts next' if i == DRIFT_I else ''}) | " + " | ".join(
            ms(round_end(get(S, label, "drift", m, c), col, i), 4) for m, c in methods if not get(S, label, "drift", m, c).empty) + " |")
    # time average after the drift
    avg = {}
    for m, c in methods:
        df = get(S, label, "drift", m, c)
        if df.empty:
            continue
        a = df[df.i > DRIFT_I].groupby("seed").grid_new.mean().values
        avg[m] = a
    out.append("| mean over all interactions after the drift | " + " | ".join(ms(avg[m], 4) for m, c in methods if m in avg) + " |")
    out.append("")
    # ---- retention
    out += [f"{h} Memory retention: the three unchanged habits (bread Sunday mornings, coffee weekday mornings, snack Friday evenings) at interaction 750 (round 15)\n",
            "Regret = mean over the cells of the three habits (decisions of the deployed policy; truth unchanged by the drift), L = learning degree of each habit "
            "(from the reward model's ask scores, 1 = fully learned; for the tabular agent from its Q values), mean of the three habits. 'Worst' = the largest regret "
            "on those cells at any interaction of the run (shows damage by overrides in between), mean over seeds.\n",
            "| method | habit regret at 750 | worst habit regret during the run | L at round 15 (mean of the 3 habits) |", "|---|---|---|---|"]
    ret = {}
    for m, c in methods:
        if get(S, label, "drift", m, c).empty:
            continue
        reg, worst, L, _, _ = retention(S, R, label, "drift", m, c)
        ret[m] = (reg, worst, L)
        out.append(f"| {NAME[m]} | {ms(reg)} | {worst.mean():.3f} | {ms(L, 2)} |")
    out.append("")
    # ---- false triggers
    ctrl = {m: get(S, label, "ctrl", m, c) for m, c in (("W", "-"), ("N", MAIN))}
    fa = {}
    if not ctrl["N"].empty:
        ev = E[(E.label == label) & (E.run == "ctrl") & (E.method == "N") & (E.cfg == MAIN)]
        cnt = ev.groupby("seed").size().reindex(range(5), fill_value=0).values
        dn, dw = ctrl["N"], ctrl["W"]
        tavg, endinc, worst = [], [], []
        for seed in range(5):
            a = dn[dn.seed == seed].set_index("i").grid_old
            b = dw[dw.seed == seed].set_index("i").grid_old
            d = (a - b).dropna()
            d = d[d.index > PER]
            tavg.append(d.mean())
            endinc.append(d[[i for i in d.index if i % PER == 0]].mean())
            worst.append(d.max())
        fa = {"count": cnt, "tavg": np.array(tavg), "end": np.array(endinc), "worst": np.array(worst)}
        out += [f"{h} False triggers: control run without any habit change (original labels throughout, main setting δ = 0.15, k = 2)\n",
                "| quantity | value |", "|---|---|",
                f"| triggers per run (all are false), mean ± std over 5 seeds | {ms(cnt, 1)} (per seed: {', '.join(str(int(x)) for x in cnt)}) |",
                f"| full-grid regret increase of the new method over warm start, mean over all interactions from 51 to 750 (paired by seed) | {ms(fa['tavg'], 4)} |",
                f"| … at the round ends only (retrained models, overrides cleared) | {ms(fa['end'], 4)} |",
                f"| … largest increase at any interaction, mean over seeds | {fa['worst'].mean():.4f} |",
                f"| samples deleted by the end of the control run, mean | {R[(R.label == label) & (R.run == 'ctrl') & (R.method == 'N') & (R.cfg == MAIN) & (R['round'] == 15)].n_deleted.mean():.0f} of 750 |",
                f"| warm-start full-grid regret in the control run, mean over interactions 51–750 | {ms([dw[(dw.seed == s) & (dw.i > PER)].grid_old.mean() for s in range(5)], 4)} |", ""]
    # ---- sensitivity
    sens = []
    if label == "cont":
        out += [f"{h} Sensitivity to δ and k (and a diagnostic with the topic field in the class definition)\n",
                "Same three quantities for each setting. 'Topic class' is NOT allowed in the method (the topic field is forbidden); it is run only to show whether failures come from the "
                "class definition. Reaching = seeds reaching the 0.05 target / mean interactions (never = 601); retention = regret on the 3 other habits at 750 and L at round 15; "
                "control = false triggers per run and the time-averaged full-grid regret increase over warm start.\n",
                "| setting | triggers per run, drift run (true / false) | reaching 0.05 | mean interactions (never = 601) | retention: regret / L | control: false triggers per run | control: regret increase, time-averaged / at round ends | "
                "(1) speed ≤ last-3-rounds | (2) retention ≥ warm start | (3) increase ≤ 0.005 (time-averaged) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for c in cfgs:
            dd, cc = get(S, label, "drift", "N", c), get(S, label, "ctrl", "N", c)
            if dd.empty:
                continue
            f, _ = reach(dd)
            ev = E[(E.label == label) & (E.run == "drift") & (E.method == "N") & (E.cfg == c)]
            ntrue = ev[ev.true_drift].groupby("seed").size().reindex(range(5), fill_value=0).mean()
            nfalse = ev[~ev.true_drift].groupby("seed").size().reindex(range(5), fill_value=0).mean()
            reg, _, L, _, _ = retention(S, R, label, "drift", "N", c)
            if not cc.empty:
                evc = E[(E.label == label) & (E.run == "ctrl") & (E.method == "N") & (E.cfg == c)].groupby("seed").size().reindex(range(5), fill_value=0)
                inc, inc_end = [], []
                for seed in range(5):
                    a = cc[cc.seed == seed].set_index("i").grid_old
                    b = ctrl["W"][ctrl["W"].seed == seed].set_index("i").grid_old
                    d = (a - b).dropna()[lambda x: x.index > PER]
                    inc.append(d.mean())
                    inc_end.append(d[[i for i in d.index if i % PER == 0]].mean())
                ctl = f"{evc.mean():.1f} | {np.mean(inc):+.4f} ± {np.std(inc, ddof=1):.4f} / {np.mean(inc_end):+.4f}"
                c3 = np.mean(inc) <= 0.005
            else:
                ctl, c3 = "– | –", None
            c1 = censored(f).mean() <= censored(reach_res["S"][0]).mean()
            dr, dL = reg - ret["W"][0], L - ret["W"][2]
            c2 = (dr.mean() <= dr.std(ddof=1) / np.sqrt(len(dr))) and (dL.mean() >= -dL.std(ddof=1) / np.sqrt(len(dL)))
            sens.append((c, c1, c2, c3))
            mark = lambda v: "–" if v is None else ("met" if v else "**not met**")
            out.append(f"| {c}{' (main)' if c == MAIN else ''} | {ntrue:.1f} / {nfalse:.1f} | {sum(v is not None for v in f.values())} of 5 | {censored(f).mean():.0f} | "
                       f"{ms(reg)} / {L.mean():.2f} | {ctl} | {mark(c1)} | {mark(c2)} | {mark(c3)} |")
        allok = [c for c, a1, a2, a3 in sens if a1 and a2 and a3 and "topicclass" not in c]
        out += ["", "Settings of the method (topic-class diagnostic excluded) meeting all three conditions: **" + (", ".join(allok) if allok else "none") + "**.", ""]
    # ---- events
    ev = E[(E.label == label) & (E.run == "drift") & (E.method == "N") & (E.cfg == MAIN)]
    if len(ev):
        per = ev.groupby("seed").agg(n=("i", "size"), n_true=("true_drift", "sum"), deleted=("deleted", "sum"), fs=("deleted_fruit_sat", "sum"),
                                     oh=("deleted_other_habit", "sum"), ot=("deleted_other_topics", "sum"))
        out += [f"{h} What the triggers did (drift run, main setting)\n",
                "| quantity per run (mean over 5 seeds) | value |", "|---|---|",
                f"| triggers | {per.n.mean():.1f} |",
                f"| … of which caused by the real drift (the triggering interaction is a post-drift fruit-Saturday scenario) | {per.n_true.mean():.1f} |",
                f"| samples deleted from the pool in total (some trigger deletions overlap with earlier ones only once) | {per.deleted.mean():.0f} of 750 |",
                f"| … fruit-Saturday samples | {per.fs.mean():.0f} |",
                f"| … samples of the three other habits' cells (bread / coffee / snack, their habit slots) | {per.oh.mean():.1f} |",
                f"| … samples of topics other than fruit | {per.ot.mean():.0f} |",
                f"| first trigger caused by the drift, interactions after the drift (mean of seeds that have one) | "
                f"{np.mean([e.i.min() - DRIFT_I for _, e in ev[ev.true_drift].groupby('seed')]) if ev.true_drift.any() else float('nan'):.1f} |", ""]
    return out, {"reach": reach_res, "ret": ret, "fa": fa, "avg": avg, "sens": sens}


def figure(S, fname):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [(axes[0, 0], "cont", "fs_new", "Saturday-fruit cells, new truth"), (axes[0, 1], "cont", "grid_new", "full grid, truth of the moment")]
    for ax, label, col, ttl in specs:
        for m, c in (("W", "-"), ("S", "-"), ("C", "-"), ("N", MAIN)):
            df = get(S, label, "drift", m, c)
            if df.empty:
                continue
            w = wide(df, col).copy()
            old = wide(df, col.replace("new", "old"))
            w.loc[:, [i for i in w.columns if i <= DRIFT_I]] = old.loc[:, [i for i in old.columns if i <= DRIFT_I]]   # truth of the moment
            mu, sd = w.mean(0), w.std(0, ddof=1)
            ax.plot(w.columns, mu, color=COL[m], lw=1.6, label=NAME[m])
            ax.fill_between(w.columns, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
        ax.axvline(DRIFT_I, color="#c44", ls="-.", lw=1)
        if col == "fs_new":
            ax.axhline(TARGET, color="k", ls="--", lw=1)
        ax.set_title(f"continuous labels: {ttl}", fontsize=10, loc="left")
        ax.set_xlabel("interaction")
        ax.set_ylabel("regret")
        ax.set_xlim(50, T_END)
        if col == "fs_new":
            ax.set_xlim(100, 450)
    axes[0, 0].legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    dn, dw = get(S, "cont", "ctrl", "N", MAIN), get(S, "cont", "ctrl", "W")
    if not dn.empty and not dw.empty:
        d = wide(dn, "grid_old") - wide(dw, "grid_old")
        mu, sd = d.mean(0), d.std(0, ddof=1)
        ax.plot(d.columns, mu, color=COL["N"], lw=1.4)
        ax.fill_between(d.columns, mu - sd, mu + sd, color=COL["N"], alpha=0.15, lw=0)
        ax.axhline(0.005, color="k", ls="--", lw=1)
        ax.axhline(0, color="gray", lw=0.8)
        ax.text(T_END, 0.0055, "criterion 0.005", ha="right", fontsize=8)
    ax.set_title("control without drift: full-grid regret of the new method − warm start", fontsize=10, loc="left")
    ax.set_xlabel("interaction")
    ax.set_ylabel("regret increase")
    ax = axes[1, 1]
    for m, c in (("W", "-"), ("S", "-"), ("C", "-"), ("N", MAIN)):
        df = get(S, "bin", "drift", m, c)
        if df.empty:
            continue
        w = wide(df, "fs_new").copy()
        old = wide(df, "fs_old")
        w.loc[:, [i for i in w.columns if i <= DRIFT_I]] = old.loc[:, [i for i in old.columns if i <= DRIFT_I]]
        mu, sd = w.mean(0), w.std(0, ddof=1)
        ax.plot(w.columns, mu, color=COL[m], lw=1.6, label=NAME[m])
        ax.fill_between(w.columns, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
    ax.axvline(DRIFT_I, color="#c44", ls="-.", lw=1)
    ax.axhline(TARGET, color="k", ls="--", lw=1)
    ax.set_xlim(100, 450)
    ax.set_title("0/1 labels: Saturday-fruit cells, new truth", fontsize=10, loc="left")
    ax.set_xlabel("interaction")
    ax.set_ylabel("regret")
    fig.suptitle("Opposition-triggered local relearning (mean ± std over 5 seeds; red line = drift starts)", x=0.01, ha="left", fontsize=11, color=plots.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=170)
    plt.close(fig)


def criteria(label, nums):
    """The three pre-registered conditions for the main setting."""
    lines = []
    r = nums["reach"]
    if "N" in r and "S" in r:
        tn, ts = censored(r["N"][0]), censored(r["S"][0])
        c1 = tn.mean() <= ts.mean()
        lines.append(f"1. Adaptation speed not slower than retraining from scratch on the last 3 rounds: mean interactions to < {TARGET}: new method {tn.mean():.0f} "
                     f"(seeds reaching: {sum(v is not None for v in r['N'][0].values())} of 5), last-3-rounds retrain {ts.mean():.0f} "
                     f"(reaching: {sum(v is not None for v in r['S'][0].values())} of 5); warm start {censored(r['W'][0]).mean():.0f} "
                     f"({sum(v is not None for v in r['W'][0].values())} of 5; 601 = never within the 600 interactions) → **{'met' if c1 else 'not met'}**.")
    else:
        c1 = None
    rn, rw = nums["ret"].get("N"), nums["ret"].get("W")
    if rn is not None and rw is not None:
        d_reg, d_L = rn[0] - rw[0], rn[2] - rw[2]
        se_reg, se_L = d_reg.std(ddof=1) / np.sqrt(len(d_reg)), d_L.std(ddof=1) / np.sqrt(len(d_L))
        c2 = (d_reg.mean() <= se_reg) and (d_L.mean() >= -se_L)
        lines.append(f"2. Retention of the three other habits not worse than warm start (rule used: the new method's mean regret not above warm start's by more than the "
                     f"paired standard error, and its L not below by more than the paired SE): regret at round 15 {rn[0].mean():.3f} vs {rw[0].mean():.3f} "
                     f"(paired difference {d_reg.mean():+.3f} ± SE {se_reg:.3f}); L {rn[2].mean():.2f} vs {rw[2].mean():.2f} (paired difference {d_L.mean():+.2f} ± SE {se_L:.2f}) → "
                     f"**{'met' if c2 else 'not met'}**.")
    else:
        c2 = None
    fa = nums["fa"]
    if fa:
        c3 = fa["tavg"].mean() <= 0.005
        lines.append(f"3. In the no-drift control the full-grid regret increase caused by false triggers ≤ 0.005: {fa['tavg'].mean():+.4f} ± {fa['tavg'].std(ddof=1):.4f} "
                     f"(time-averaged over interactions 51–750, raw regret; at round ends only {fa['end'].mean():+.4f}); false triggers per run {fa['count'].mean():.1f} → "
                     f"**{'met' if c3 else 'not met'}**.")
    else:
        c3 = None
    ok = None if None in (c1, c2, c3) else bool(c1 and c2 and c3)
    return lines, ok


def write_report():
    if not os.path.exists(SERIES):
        return
    S = pd.read_csv(SERIES)
    R = pd.read_csv(ROUNDS)
    E = pd.read_csv(EVENTS) if os.path.exists(EVENTS) else pd.DataFrame(columns=["label", "run", "method", "cfg", "seed", "i", "true_drift"])
    g = class_geometry()
    cfgs = [MAIN, "d0.10_k2", "d0.20_k2", "d0.15_k1", "d0.15_k3", "d0.15_k2_topicclass"]
    out = ["# Experiment B: opposition-triggered local relearning (`drift_trigger.py`, `drift_trigger_report.py`)\n"]
    summary_pos = len(out)
    cont, nums_c = block(S, R, E, "cont", "###", cfgs) if not get(S, "cont", "drift", "N", MAIN).empty else ([], None)
    lines_c, ok_c = criteria("cont", nums_c) if nums_c else ([], None)
    binb, nums_b = block(S, R, E, "bin", "###", [MAIN]) if not get(S, "bin", "drift", "N", MAIN).empty else ([], None)
    lines_b, ok_b = criteria("bin", nums_b) if nums_b else ([], None)
    out += ["## Pre-registered criterion (main setting δ = 0.15, k = 2, continuous labels)\n"] + lines_c
    if ok_c is not None:
        out.append(f"\n**All three conditions {'hold' if ok_c else 'do not all hold'}.**\n")
    if lines_b:
        out += ["### The same three conditions with 0/1 labels (optional repetition; opposition = executed action's predicted acceptance probability > 0.8 but label 0, or < 0.2 but label 1; k = 2)\n"] + lines_b
        if ok_b is not None:
            out.append(f"\n**With 0/1 labels all three conditions {'hold' if ok_b else 'do not all hold'}.**\n")
    out += ["## Findings (numbers from the tables below)\n",
            "1. **Fast local adaptation works at the moment of the trigger.** With the specified setting the Saturday-fruit cells' regret is below 0.05 after a mean of 32 interactions "
            "(median 25, range 11–75; 5 of 5 seeds), against 80 (median 50) for retraining from scratch on the last 3 rounds, 426 / 450 for the tabular agent (online / per round) "
            "and 561 for warm start (only 2 of 5 seeds get there within 600 interactions, both at 500). Condition (1) as worded is met.",
            "2. **The gain is transient.** The override is cleared at every retrain and the retrained model has not absorbed the change: regret is 0.012 ± 0.028 at 49 interactions "
            "after the drift and 0.168 ± 0.061 at 50 (the round end); it needs a mean of 300 interactions to stay below 0.05 (last-3-rounds retrain: 120).",
            "3. **It rests on very few true triggers.** Per drift run 40.4 triggers, of which 1.8 are caused by the real drift and 38.6 are false; in the no-drift control 71.8 ± 4.6 "
            "false triggers per run and 377 of 750 samples deleted (223 in the drift run).",
            "4. **Retention of the other habits fails (condition 2).** L of the three unchanged habits at round 15 is 0.38 ± 0.06 against 0.78 ± 0.07 for warm start; 39.8 samples "
            "of their habit cells are deleted per drift run; regret on their cells 0.310 vs 0.277. The class (CLS cosine > 0.9, same day and time) is the whole (day, time) slot "
            "of all topics, so a trigger on Saturday also deletes and overrides the other topics of that slot.",
            "5. **Condition (3) holds only under the metric I chose.** The time-averaged full-grid regret increase in the control is +0.0019 ± 0.0023 (≤ 0.005), but measured at the "
            "round ends (retrained models) it is +0.0076 ± 0.0032. Warm start's inertia hides the deletion in the decisions; the false-trigger count and the deleted half of the data are "
            "the more informative signals.",
            "6. **No setting of δ and k meets all three conditions** (sensitivity table; δ ∈ {0.10, 0.15, 0.20} × k = 2 and k ∈ {1, 2, 3} at δ = 0.15). k = 3 reduces false triggers the "
            "most (11.8 per drift run) and gives the best retention (L 0.60, regret 0.230), but adapts more slowly than the last-3-rounds retrain (100 vs 80 interactions) and its "
            "retention L is still below warm start's 0.78.",
            "7. **Diagnostic, not part of the method:** with the topic field in the class definition false triggers fall to 16.0 per drift run (44.2 in the control) but retention "
            "L is still only 0.55. The coarse CLS class is part of the problem, but exact classes would not rescue the mechanism at δ = 0.15, k = 2.",
            "8. **With 0/1 labels the mechanism is almost inert.** 1.4 triggers per drift run (0.2 caused by the drift; the first one 306 interactions after the drift), "
            "so it is no faster than warm start (390 vs 390 mean interactions, 3 of 5 seeds reaching vs 4 of 5; last-3-rounds retrain: 70). Conditions (2) and (3) hold trivially "
            "because almost nothing triggers; condition (1) fails. Retraining from scratch on the last three rounds adapts fastest but forgets the other habits entirely "
            "(regret 0.85, L 0.02); the tabular agent never gets below 0.05 on the changed cells.\n",
            "## Setup\n",
            "- **Stream.** 15 rounds × 50 interactions, scenarios of `data_drift15_src` (`generate_dataset.py --n 1100 --seed 42 --with-act`, first 750 training scenarios, 165 validation "
            "scenarios for early stopping). Dense-evidence variant of `habit_drift.py`: 8 of every 50 training scenarios (13 of the 165 validation scenarios) are replaced by random "
            "fruit-Saturday scenarios (same random generator and seed 777 as the dense variant). Rounds 1–3 carry the original labels; from round 4 (interaction 151) the Saturday-fruit "
            "rule no longer holds (necessity 0.8 instead of 0.1: ask value 0.8 × receptiveness, act value 0.2) for all later fruit-Saturday scenarios; the validation scenarios follow. "
            "`ctrl` runs use the same scenarios with the original labels throughout. The 0/1 data are Bernoulli draws of the clean values of the same streams (one draw, `numpy` seed 4321).",
            "- **Methods.** All decide with the DPO policy A, deployed from the last round-end retrain: W = warm start on all data (reward model continues, DPO continues; paper default); "
            "S = from scratch on the last three rounds; C = tabular (constant lr 0.5; 0/1: acceptance frequency) updated online after every interaction; Cb = tabular updated once per round "
            "(the protocol of `habit_drift.py`); N = W plus the opposition-triggered mechanism. Seeds 0–4 and the per-round seeds of `habit_drift.py` (`seed·1000 + 900 + round`), so N is "
            "identical to W until its first trigger. Hyper-parameters: `budget.HP`, untouched. Only δ and k were varied (below).",
            f"- **Class definition, as specified, is degenerate with frozen BERT CLS.** Cosine of the question CLS vectors > 0.9 and same day and time. In this question set the cosine is "
            f"> 0.9 for {g['same_gt']:.0%} of the {g['same_n']} same-topic phrasing pairs and for **{g['diff_gt']:.0%} of the {g['diff_n']} pairs from different topics** (cross-topic "
            f"mean {g['diff_mean']:.3f}, minimum {g['diff_min']:.3f}). The class is therefore the (day, time of day) slot of all topics together: the CLS cosine cannot separate topics. "
            "I kept the specified threshold (only δ and k may be tuned). A topic-class variant is run as a labelled diagnostic only.",
            "- **Interpretations I had to make.** (a) The executed action is the policy's decision (ask when the class is overridden); the residual uses the label of that action. "
            "(b) Monitoring starts in round 2 (the first model exists after round 1). (c) Oppositions are counted per class around the newest opposition: pairwise relation (same day, "
            "same time, cosine > 0.9); the group that reaches k triggers. (d) 'Samples older than the trigger time' = samples of the class that arrived before the *first* opposition of the "
            "triggering group (the opposing observations themselves are kept; deleting them would remove the evidence of the new rule). (e) After a trigger the class is overridden until the "
            "next retrain and its further oppositions are not counted (otherwise a second trigger could delete the first group's new-rule samples). (f) At every retrain the override and the "
            "pending oppositions are cleared; deletions are permanent. (g) The override applies on the grid to the cells with the same day and time as the triggering interaction and a question "
            "cosine > 0.9 (i.e., all topics in that slot). (h) The evaluation after every interaction applies the current overrides to the decisions of the deployed policy; the retrained "
            "model replaces them at the round end.",
            "- The truth used for regret changes with the stream: the old rule up to interaction 150 and the new rule afterwards (the two grids differ only in the 18 Saturday-fruit cells). "
            "Grid regret is the raw mean over 1512 cells (baseline min(always-, never-ask) ≈ 0.05).\n"]
    out += ["## Continuous labels\n"] + cont
    if binb:
        out += ["## 0/1 labels (optional repetition, main setting only)\n"] + binb
    figure(S, "drift_trigger.png")
    with open(os.path.join(RESULTS_DIR, "drift_trigger.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
