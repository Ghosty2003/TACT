"""results/v3_paper_numbers.md: the numbers of Sec. 5.4, Appendix B (habit drift), Appendix D (partial feedback) and the novel-topic statement under the NEW default (recency-weighted warm start, half-life 2); reporting seeds 0-4.

    TACT_V2=1 python -m tact.reporting.v3_paper_numbers        # after v3_semionline.py, v3_drift.py, v3_partial_cold.py, v3_paper_ood.py
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tact.drift import habit_drift as H
from tact.core.model import ROOT

V2, V3 = os.path.join(ROOT, "results", "v2"), os.path.join(ROOT, "results", "v3")
HH = json.load(open(os.path.join(V3, "chosen_half_life.json")))["half_life"]
NEWM = f"R{HH}"
PER, DRIFT_I, T_END, TARGET, CAP = 50, 150, 750, 0.05, 601
HAB3 = ("bread", "coffee", "snack")
rd2 = lambda f: pd.read_csv(os.path.join(V2, f))
rd3 = lambda f: pd.read_csv(os.path.join(V3, f))
ms = lambda x, nd=4: (lambda a: f"{a.mean():.{nd}f} ± {a.std(ddof=1):.{nd}f}")(np.asarray(x, float))


def sub(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    return df.sort_values("seed")


def pw(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return d, p


out = [f"# Paper numbers under the new default (`results/v3_paper_numbers.md`)\n",
       f"New default = recency-weighted warm start, half-life {HH} rounds (weights 0.5^(age in rounds / {HH}), warm start, unweighted validation loss, early stopping as before). All numbers on the reporting seeds 0–4; "
       "mean ± std over the 5 seeds (sample std, ddof = 1) unless a column says otherwise. Old = the unweighted warm start of the paper (`results/v2`). No git commit, no push.\n"]

# ================================================================= 1
rp, rp_old = rd3("semionline_replay.csv"), rd2("replay_metrics.csv")
bu, bu_old = rd3("semionline_budget.csv"), rd2("dpo_semionline.csv")
one = sub(rd2("budget_runs.csv"), method="A", test="seen", n=350)
new = lambda st: rp[(rp.weighting == "recency") & (rp.strategy == st)]
r7 = lambda df, st: sub(df[(df.strategy == st) & (df["round"] == 7)], seed=slice(None)) if False else df[(df.strategy == st) & (df["round"] == 7)].sort_values("seed")
md = lambda df, st: df[(df.strategy == st) & (df["round"] >= 2)].groupby("seed").drift.mean().sort_index()
mw7, ms7 = r7(new("warm"), "warm"), r7(new("scratch"), "scratch")
dw, ds = md(new("warm"), "warm"), md(new("scratch"), "scratch")
w_loop = sub(bu, variant="warm_recency", test="seen", round=7)
s_loop = sub(bu, variant="scratch_recency", test="seen", round=7)
a_old = sub(bu_old, method="A", test="seen", round=7)
out += ["## 1. Sec. 5.4 semi-online stability (7 rounds × 50 interactions)\n",
        "| quantity | **new default (recency-weighted)** | old (unweighted warm start, paper) |", "|---|---|---|",
        f"| round-7 test MSE, warm start | **{ms(mw7.mse)}** | {ms(r7(rp_old, 'warm').mse)} |",
        f"| round-7 test MSE, from scratch (also recency-weighted) | **{ms(ms7.mse)}** | {ms(r7(rp_old, 'scratch').mse)} |",
        f"| mean prediction drift (rounds 2–7), warm start | **{ms(dw.values)}** | {ms(md(rp_old, 'warm').values)} |",
        f"| mean prediction drift (rounds 2–7), from scratch | **{ms(ds.values)}** | {ms(md(rp_old, 'scratch').values)} |",
        f"| seeds with warm drift < scratch drift (per-seed mean over rounds 2–7) | **{int((dw.values < ds.values).sum())} of 5** | {int((md(rp_old, 'warm').values < md(rp_old, 'scratch').values).sum())} of 5 |",
        f"| decision regret after 350 interactions, semi-online loop (warm start, 75 test scenarios) | **{ms(w_loop.regret)}** | {ms(a_old.regret)} |",
        f"| decision regret after 350 interactions, one-shot offline (N = 350, `budget_runs.csv`) | {ms(one.regret)} | {ms(one.regret)} |",
        f"| decision regret after 350 interactions, from-scratch retraining with the recency weights (reference) | {ms(s_loop.regret)} | – |"]
# per-round drift counts
per_round = []
for r in range(2, 8):
    a, b = new("warm")[new("warm")["round"] == r].sort_values("seed").drift.values, new("scratch")[new("scratch")["round"] == r].sort_values("seed").drift.values
    per_round.append(int((a < b).sum()))
d, p = pw(dw.values, ds.values)
dl, pl = pw(w_loop.regret.values, one.regret.values)
out += [f"\nWarm − scratch drift (paired): {d.mean():+.4f} ± SE {d.std(ddof=1) / np.sqrt(5):.4f}, Wilcoxon p = {p:.3f} (the smallest possible with 5 seeds is 0.0625). Seeds with warm drift < scratch drift, round by round (rounds 2…7): "
        f"{', '.join(f'{k} of 5' for k in per_round)}. Loop vs one-shot regret (paired): {dl.mean():+.4f} ± SE {dl.std(ddof=1) / np.sqrt(5):.4f}, loop lower in {int((dl < 0).sum())} of 5 seeds (p = {pl:.3f}). "
        "The MSE / drift rows come from the `replay.py` protocol (ask-only data), the regret rows from the `budget.semi_online` protocol (`data_dpo`, ask + act rows); the old column of the MSE / drift rows is `results/v2/replay_metrics.csv`, "
        "which an unweighted rerun of the v3 code reproduces only up to numerical noise beyond round 1 (see `results/v3_recency.md` §2).\n"]

# ================================================================= 2
sd, rd_, fd = rd2("drift_targeted_series.csv"), rd2("drift_targeted_rounds.csv"), rd2("drift_targeted_final.csv")
sn, rn, fn = rd3("drift_series.csv"), rd3("drift_rounds.csv"), rd3("drift_final.csv")
ts_ = rd2("drift_trigger_series.csv")
ts_ = ts_[(ts_.label == "cont") & (ts_.run == "drift") & (ts_.method == "C") & (ts_.cfg == "-")]
tr_ = rd2("drift_trigger_rounds.csv")
tr_ = tr_[(tr_.label == "cont") & (tr_.run == "drift") & (tr_.method == "C")]
g_old, g_new = H.truth_grid(False), H.truth_grid(True)
ratio_old, ratio_new = H.collapse_ratio(g_old.c_ask.values, g_old), H.collapse_ratio(g_new.c_ask.values, g_new)
mid = (ratio_old + ratio_new) / 2
ENT = [(f"**new default: recency-weighted warm start (h = {HH})**", sn, rn, "run", NEWM), ("old default: warm start, unweighted", sd, rd_, "cfg", "W"),
       ("windowed: from scratch, last 3 rounds", sd, rd_, "cfg", "S"), ("tabular, constant learning rate 0.5 (online)", ts_, tr_, None, "C")]


def pick(S, m):
    if "cfg" in S:
        S = S[(S.run == "drift") & (S.method == m) & (S.cfg == "-")] if m != "C" else S
    elif "run" in S:
        S = S[(S.run == "drift") & (S.method == m)]
    return S


def reach(S):
    first, stays = {}, {}
    for seed, g in S.groupby("seed"):
        g = g[g.i > DRIFT_I].sort_values("i")
        below = (g.fs_new < TARGET).values
        first[seed] = int(g.i.values[below][0] - DRIFT_I) if below.any() else None
        st = None
        for i, b in zip(g.i.values[::-1], below[::-1]):
            if b:
                st = int(i - DRIFT_I)
            else:
                break
        stays[seed] = st
    return first, stays


cens = lambda d: np.array([CAP if v is None else v for v in d.values()], float)
rows = []
for lab, S, R, _, m in ENT:
    Sd = S[(S.run == "drift") & (S.method == m) & ((S.cfg == "-") if "cfg" in S else True)] if m != "C" else S
    Rd = R[(R.run == "drift") & (R.method == m) & ((R.cfg == "-") if "cfg" in R else True)] if m != "C" else R
    first, stays = reach(Sd)
    L15 = Rd[Rd["round"] == 15].sort_values("seed")
    L3 = L15[[f"L_{h}" for h in HAB3]].mean(axis=1).values
    c3, c15 = Rd[Rd["round"] == 3].sort_values("seed").collapse.values, L15.collapse.values
    end = Sd[Sd.i == T_END].sort_values("seed")
    rows.append(dict(lab=lab, first=first, stays=stays, L3=L3, Lh={h: L15[f"L_{h}"].values for h in HAB3}, c3=c3, c15=c15, grid_end=end.grid_new.values, fs_end=end.fs_new.values))
out += ["## 2. Habit drift (paper Appendix B): the Saturday-fruit habit stops after interaction 150; 15 rounds × 50 interactions; 5 reporting seeds\n",
        "Adaptation = interactions after the drift until the regret on the 18 drifted cells (new truth) is below 0.05 — *first* crossing and *sustained* (from which it stays below until the end of the stream); mean over seeds, a seed that never gets there counts as 601, "
        "and the number of seeds that reach it is given. Retention L = learning degree of the three habits that do not change (bread / Sunday morning, coffee / weekday morning, snack / Friday evening) at round 15, mean of the three. "
        f"Full-grid regret at the end of the stream = 1512-cell grid under the new truth after interaction 750. Score ratio = predicted ask score of the Saturday-fruit cells over the other days of the afternoon fruit heatmap (collapse ratio; truth {ratio_old:.2f} before the drift, {ratio_new:.2f} after it). "
        f"**Expired habit unlearned** is counted in two ways: (a) the regret on the 18 drifted cells is < {TARGET} after interaction 750, (b) the predicted ratio at round 15 is above the midpoint {mid:.2f} between the old and the new truth.\n",
        "| method | adaptation: first (interactions) | seeds reaching | adaptation: sustained | retention L (3 other habits) | full-grid regret, end of stream | pre-drift score ratio (round 3) | score ratio at round 15 | seeds unlearned (a) regret < 0.05 | seeds unlearned (b) ratio > midpoint |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['lab']} | {cens(r['first']).mean():.0f} | {sum(v is not None for v in r['first'].values())} of 5 | {cens(r['stays']).mean():.0f} | {ms(r['L3'], 2)} | {ms(r['grid_end'])} | {ms(r['c3'], 2)} | {ms(r['c15'], 2)} | "
               f"{int((r['fs_end'] < TARGET).sum())} of 5 | {int((r['c15'] > mid).sum())} of 5 |")
out += ["\nRetention per habit (L at round 15):\n", "| method | bread, Sunday mornings | coffee, weekday mornings | snack, Friday evenings |", "|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['lab']} | " + " | ".join(ms(r["Lh"][h], 2) for h in HAB3) + " |")
out += ["\nPer seed (new default): first / sustained adaptation, score ratio at rounds 3 and 15, final regret on the drifted cells: " +
        "; ".join(f"seed {s}: {rows[0]['first'][s]} / {rows[0]['stays'][s]}, {rows[0]['c3'][s]:.2f} → {rows[0]['c15'][s]:.2f}, {rows[0]['fs_end'][s]:.3f}" for s in range(5)) + ".",
        f"\nCheck of the quoted old pre-drift ratio (0.26 ± 0.05): the old default's collapse ratio at round 3 is {ms(rows[1]['c3'], 2)}; the new default's is {ms(rows[0]['c3'], 2)}."]

# ================================================================= 3
pf, pf_old = rd3("partial_runs.csv"), rd2("partial_feedback_runs.csv")
SET = ["full", "reveal", "bandit", "eps"]
NAME = {"full": "full feedback", "reveal": "ask-reveals", "bandit": "strict bandit", "eps": "strict bandit + ε-greedy (ε = 0.1)"}
gp = lambda df, st, s, r, col: df[(df.stream == st) & (df.setting == s) & (df["round"] == r)].sort_values("seed")[col].values
out += ["\n## 3. Partial feedback (paper Appendix D), semi-online loop with the recency-weighted default\n",
        "Cold start (round 1 asked), 5 seeds, mean ± std. Stream `dpo7` = the paper's semi-online stream (7 rounds × 50 scenarios of `data_dpo`; 75 test scenarios). Regret = test regret after round 7; **Δ = paired difference to full feedback, mean ± std over the seeds** "
        "(not SE); L = Saturday-fruit learning degree. Old columns = `results/v2/partial_feedback_runs.csv`.\n",
        "| setting | **test regret** | **paired Δ to full** | **L** | old: test regret | old: paired Δ | old: L |", "|---|---|---|---|---|---|---|"]
for s in SET:
    x, xo = gp(pf, "dpo7", s, 7, "test_regret"), gp(pf_old, "dpo7", s, 7, "test_regret")
    dn, do_ = x - gp(pf, "dpo7", "full", 7, "test_regret"), xo - gp(pf_old, "dpo7", "full", 7, "test_regret")
    out.append(f"| {NAME[s]} | **{ms(x)}** | {'–' if s == 'full' else f'**{dn.mean():+.4f} ± {dn.std(ddof=1):.4f}**'} | **{ms(gp(pf, 'dpo7', s, 7, 'L_sat'), 2)}** | {ms(xo)} | "
               f"{'–' if s == 'full' else f'{do_.mean():+.4f} ± {do_.std(ddof=1):.4f}'} | {ms(gp(pf_old, 'dpo7', s, 7, 'L_sat'), 2)} |")
cum = lambda df, st, s, col: df[(df.stream == st) & (df.setting == s)].groupby("seed")[col].sum().sort_index().values
out += ["\n**Labels obtained** (cumulative over the 7 rounds of `dpo7`; full feedback = 350 ask + 350 act labels):\n", "| setting | ask labels | act labels | act-label share (act / all labels) | interactions asked during deployment (share) |", "|---|---|---|---|---|"]
for s in SET:
    a, b = cum(pf, "dpo7", s, "ask_labels"), cum(pf, "dpo7", s, "act_labels")
    asked = pf[(pf.stream == "dpo7") & (pf.setting == s)].groupby("seed").deploy_ask_rate.mean().sort_index().values
    out.append(f"| {NAME[s]} | {ms(a, 1)} | {ms(b, 1)} | {ms(b / (a + b), 3)} | {ms(asked, 3)} |")
a_r = cum(pf, "dpo7", "reveal", "ask_labels")
out.append(f"\nUnder ask-reveals the number of ask labels is {ms(a_r, 1)} of the 350 interactions (round 1 contributes 50, because every cold-start interaction is asked); under strict bandit the act-label share is {ms(cum(pf, 'dpo7', 'bandit', 'act_labels') / (cum(pf, 'dpo7', 'bandit', 'ask_labels') + cum(pf, 'dpo7', 'bandit', 'act_labels')), 3)}.")
out += ["\n**15-round stream** (`drift15` = `data_drift15/cont_ctrl`, no drift, 165 test scenarios), test regret after round 15 (round 7 also given):\n", "| setting | round 7: test regret | round 15: test regret | round 15: paired Δ to full (mean ± std) | round 15: L | old round 15: test regret |", "|---|---|---|---|---|---|"]
for s in SET:
    x15, x7 = gp(pf, "drift15", s, 15, "test_regret"), gp(pf, "drift15", s, 7, "test_regret")
    d15 = x15 - gp(pf, "drift15", "full", 15, "test_regret")
    out.append(f"| {NAME[s]} | {ms(x7)} | {ms(x15)} | {'–' if s == 'full' else f'{d15.mean():+.4f} ± {d15.std(ddof=1):.4f}'} | {ms(gp(pf, 'drift15', s, 15, 'L_sat'), 2)} | {ms(gp_ := gp(pf_old, 'drift15', s, 15, 'test_regret'))} |")
m15 = {s: gp(pf, "drift15", s, 15, "test_regret").mean() for s in SET}
order = sorted(("full", "reveal", "bandit"), key=lambda s: m15[s])
out.append(f"\nOrdering after round 15 (best → worst, mean regret): {' < '.join(f'{NAME[s]} ({m15[s]:.4f})' for s in order)}"
           f"; for reference old: " + " < ".join(f"{NAME[s]} ({gp(pf_old, 'drift15', s, 15, 'test_regret').mean():.4f})" for s in sorted(('full', 'reveal', 'bandit'), key=lambda s: gp(pf_old, 'drift15', s, 15, 'test_regret').mean())) + ".")
out.append("Per seed, round 15: " + "; ".join(f"seed {k}: full {gp(pf, 'drift15', 'full', 15, 'test_regret')[k]:.4f}, reveal {gp(pf, 'drift15', 'reveal', 15, 'test_regret')[k]:.4f}, bandit {gp(pf, 'drift15', 'bandit', 'test_regret'.replace('test_regret', 'test_regret') if False else 15, 'test_regret')[k]:.4f}" for k in range(5)) + ".")

# ================================================================= 4
op, ob = rd3("ood_priority_v3.csv"), rd3("ood_base_v3.csv")
bl = json.load(open(os.path.join(V3, "ood_baselines_v3.json")))
oo = rd2("ood_priority.csv")
out += ["\n## 4. Novel-topic regret with the new default (`v3_paper_ood.py`, rerun of `v2_ood_priority.py`)\n",
        "**What the old 0.066 vs 0.033 is.** It is not a number of `ood_priority.md`'s retraining table: it is the regret of the N = 350 model on the 30 held-out novel-topic queries of the random half of `ood_eval.py` ('half' scheme, score left unchanged; always-ask on the same 30 queries). "
        "Both are reproduced below (the 'old' rows reproduce `ood_downstream.csv` / `ood_priority.csv` exactly: max |difference| of the regret = "
        f"{np.abs(op[op.variant == 'old'].sort_values(['seed', 'B', 'strategy']).regret_new.values - oo.sort_values(['seed', 'B', 'strategy']).regret_new.values).max():.1e}). "
        f"Under the new default the base model is the recency-weighted semi-online model after 7 rounds (seeds `seed·1000 + 700 + round`), and the one warm-start update of the priority experiment is round 8 of the same loop "
        f"(350 old scenarios weighted 0.5^((8 − round)/{HH}), the B newly labelled scenarios weight 1). Constant policies on the 60 evaluation queries: always-ask {bl['always-ask']:.3f}, always-act {bl['always-act']:.3f}.\n",
        "### 4a. Base model on novel topics (no update), score unchanged\n", "| queries | model | new default | old | always-ask (same queries) | seeds where the new-default model is worse than always-ask |", "|---|---|---|---|---|---|"]
for scheme, nm in (("half", "30 held-out queries (random half; the paper's 0.066 vs 0.033)"), ("loto", "all 60 queries")):
    n_, o_ = sub(ob[(ob.variant == "new") & (ob.scheme == scheme)], seed=slice(None)) if False else ob[(ob.variant == "new") & (ob.scheme == scheme)].sort_values("seed"), ob[(ob.variant == "old") & (ob.scheme == scheme)].sort_values("seed")
    out.append(f"| {nm} | regret | **{ms(n_.model)}** | {ms(o_.model)} | {ms(n_.always_ask)} | {int((n_.model.values > n_.always_ask.values).sum())} of 5 (old: {int((o_.model.values > o_.always_ask.values).sum())} of 5) |")
out += ["\n### 4b. Regret on the 60 held-out novel-topic queries of the priority experiment: no update vs after B labelled scenarios (round 8 update)\n",
        f"Always-ask = {bl['always-ask']:.3f}. 'worse' = regret above always-ask.\n", "| B | strategy | **new default: regret on novel topics** | old: regret on novel topics | new: seeds worse than always-ask | new: regret on known topics | old: regret on known topics |", "|---|---|---|---|---|---|---|"]
for b_, st in [(0, "none")] + [(b, s) for b in (10, 20, 40) for s in ("priority", "random")]:
    n_ = op[(op.variant == "new") & (op.B == b_) & (op.strategy == st)].sort_values("seed")
    o_ = op[(op.variant == "old") & (op.B == b_) & (op.strategy == st)].sort_values("seed")
    out.append(f"| {b_} | {st if b_ else 'none (no update)'} | **{ms(n_.regret_new, 3)}** | {ms(o_.regret_new, 3)} | {int((n_.regret_new.values > bl['always-ask']).sum())} of 5 | {ms(n_.regret_known, 3)} | {ms(o_.regret_known, 3)} |")
nn = op[(op.variant == "new") & (op.B == 0)].sort_values("seed")
nh = ob[(ob.variant == "new") & (ob.scheme == "half")]
out += [f"\n**Statement “worse than always-ask until feedback accumulates”.** New default, no update: {ms(nh.model)} vs always-ask {ms(nh.always_ask)} on the 30 held-out queries (worse in {int((nh.model.values > nh.always_ask.values).sum())} of 5 seeds; old {ms(ob[(ob.variant == 'old') & (ob.scheme == 'half')].model)}), "
        f"and {ms(nn.regret_new, 3)} vs always-ask {bl['always-ask']:.3f} on the 60 evaluation queries of the priority experiment (worse in {int((nn.regret_new.values > bl['always-ask']).sum())} of 5 seeds; old {ms(op[(op.variant == 'old') & (op.B == 0)].regret_new, 3)}, "
        "i.e. the gap to always-ask is smaller under the new default on this query set and about the size of the seed noise). After the update with B = 40 labelled scenarios the new-default regret is "
        f"{ms(op[(op.variant == 'new') & (op.B == 40) & (op.strategy == 'priority')].regret_new, 3)} (priority) / {ms(op[(op.variant == 'new') & (op.B == 40) & (op.strategy == 'random')].regret_new, 3)} (random), below always-ask. "
        "So the qualitative statement still holds, with the caveat that on the 60-query set the no-update gap is within seed noise; the safest wording is the 30-query one (0.054 ± 0.027 vs 0.033), which shows the same direction as before but a smaller gap than the old 0.066.\n"]

# ================================================================= 5
out += ["## 5. Figures copied into `figures_final/` (PNG copies in `figures_final_png/`)\n",
        "| file in `figures_final/` | content | source |", "|---|---|---|",
        "| `fig_semionline.pdf` | **new default** (both warm start and from scratch recency-weighted) | `figures_v3/fig_semionline.pdf` |",
        "| `fig_semionline_v2.pdf` | old (unweighted) | the previous `figures_final/fig_semionline.pdf` |",
        "| `fig_drift.pdf` | **new default**: regret on the drifted cells and full-grid regret, old default / new default / windowed / tabular | `figures_v3/fig_drift.pdf` (84 mm, 7 pt, final-figure style) |",
        "| `fig_drift_v2.pdf` | old drift figure (older style: not 84 mm, fonts < 7 pt) | `figures_v2/fig_drift.pdf` |",
        "\n`figures_final/` previously had no drift figure (it was not in the original file list); `fig_drift.pdf` is new there, and `fig_drift_v2.pdf` is the old figure of `figures_v2/`. `MISSING.txt` is unchanged."]
with open(os.path.join(ROOT, "results", "v3_paper_numbers.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
