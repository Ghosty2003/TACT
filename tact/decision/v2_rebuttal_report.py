"""Rebuttal report: assembles results/v2/rebuttal/rebuttal.md from the csv files of the five experiments (+ the paired tests of experiment 5).

    TACT_V2=1 python -m tact.decision.v2_rebuttal_report

Inputs (all in results/v2): rebuttal/cost_receptivity_runs.csv, rebuttal/sensitivity_info.csv, rebuttal/sensitive_share_by_topic.csv, rebuttal/graded_feedback_runs.csv, rebuttal/tabular_tuning_runs.csv,
budget_runs.csv, binary_feedback_runs.csv, binary_feedback_draws_runs.csv, replay_metrics.csv, dpo_semionline.csv, ablation_threshold_vs_dpo.csv, multiuser_matrix_A.csv, plus rebuttal/verdicts.md (hand-written).
Paired statistics everywhere: per-seed differences (same seeds, same data), mean ± standard error (std / sqrt(n)), number of seeds in which the first method is better, and the two-sided exact Wilcoxon signed-rank
p value (scipy). With 5 pairs the smallest possible two-sided p is 0.0625, so no 5-seed comparison can reach p < 0.05: the p values are reference only, the direction count is the more informative number.
"""
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from tact.core.model import RESULTS_DIR

R = RESULTS_DIR
RB = os.path.join(R, "rebuttal")
rd = lambda f: pd.read_csv(os.path.join(R, f))
rb = lambda f: pd.read_csv(os.path.join(RB, f))
TRUE_RATIO = 0.15


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f}"


def paired(a, b, nd=4, better="lower"):
    """a, b: per-seed values (same order). Reports mean(a - b) ± SE, how often a is better than b, Wilcoxon p."""
    d = np.asarray(a, float) - np.asarray(b, float)
    k = int((d < 0).sum() if better == "lower" else (d > 0).sum())
    ties = int((d == 0).sum())
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return f"{d.mean():+.{nd}f} ± {d.std(ddof=1) / np.sqrt(len(d)):.{nd}f}", f"{k}/{len(d)}" + (f" ({ties} ties)" if ties else ""), f"{p:.3f}"


def first_ahead(a, b, ns):
    """First N from which a < b for every larger N (mean curves), else None."""
    ok = [n for n in ns if a[n] < b[n]]
    for n in ns:
        if all(m in ok for m in ns if m >= n):
            return n
    return None


# ------------------------------------------------------------------ data
cr = rb("cost_receptivity_runs.csv")
info = pd.read_csv(os.path.join(RB, "sensitivity_info.csv"), header=None, index_col=0)[1]
share = rb("sensitive_share_by_topic.csv")
gr = rb("graded_feedback_runs.csv")
tt = rb("tabular_tuning_runs.csv")
bud = rd("budget_runs.csv")
bud = bud[bud.test == "seen"]
BA = lambda n: bud[(bud.method == "A") & (bud.n == n)].sort_values("seed")
test_seen = pd.read_csv(os.path.join(os.path.dirname(R), "..", "data_dpo", "test_seen.csv"))
ask_, act_ = test_seen[test_seen.cand == "ask"].score_clean.values, test_seen[test_seen.cand == "act"].score_clean.values
best_ = np.maximum(ask_, act_)
TEST_BASE = {"always-ask": float((best_ - ask_).mean()), "never-ask": float((best_ - act_).mean())}
TEST_ACC = {"always-ask": float((ask_ >= act_).mean()), "never-ask": float((ask_ < act_).mean())}
GRID_BASE = {"always-ask": float(info["grid_always"]), "never-ask": float(info["grid_never"])}


def sel(df, **kw):
    m = np.ones(len(df), bool)
    for k, v in kw.items():
        m &= (df[k] == v).values
    return df[m].sort_values("seed")


out = ["# Rebuttal experiments (`results/v2/rebuttal/`)\n",
       "Final v2 architecture throughout (frozen NLI-BERT mean pooling, Context encoder + rank-16 bilinear head, threshold rule), 5 seeds (0–4), `data_dpo` unless stated, no existing result file modified. All numbers are mean ± std over "
       "the 5 seeds unless stated; paired differences are per seed, mean ± standard error, with the number of seeds in which the first method wins and the exact two-sided Wilcoxon signed-rank p. **With 5 seeds the smallest "
       "attainable two-sided Wilcoxon p is 0.0625, so no 5-seed test can reach 0.05; read the direction count, not the p value.** Scripts: `v2_rebuttal_cost.py` (1, 2), `v2_rebuttal_graded.py` (3), `v2_rebuttal_tabular.py` (4), "
       "`v2_rebuttal_report.py` (this file, experiment 5). Integrity checks: the full-TACT rows reproduce `budget_runs.csv` (method A, test_seen) exactly (max |difference| = 0), the −emotion rows reproduce "
       "`context_ablation_runs.csv`, the default tabular reproduces `budget_runs.csv` (method C), and the continuous rows of experiment 3 reproduce `binary_feedback_runs.csv` (regret, grid, L).\n"]
if os.path.exists(os.path.join(RB, "verdicts.md")):
    out.append(open(os.path.join(RB, "verdicts.md")).read() + "\n")

# ================================================================== 1
out += ["\n---\n## 1. Learned constant-cost baselines (reviewer item 3)\n",
        "Variants (all trained on the first N training scenarios, same seeds): **q_const** = r̂(q) (the TACT network trained on the ask rows only, all context embeddings zeroed, so it sees only the question text) against one scalar "
        "c = mean observed act label; **q_bucket** = same with c(e) per emotion (6 scalars). Extras (not asked for): **q_const_opt** = c chosen to minimise the training regret; **tact_ask_const** = full TACT's context-aware ask "
        "score against the scalar c (the act side made constant). Regret = on the 75 test scenarios (test_seen), clean values; 'grid' = the 1512-cell noise-free grid, regret divided by min(always-ask, never-ask) on the grid "
        f"({min(GRID_BASE.values()):.4f}). Always-ask / never-ask on the test scenarios: regret {TEST_BASE['always-ask']:.3f} / {TEST_BASE['never-ask']:.3f}, agreement {TEST_ACC['always-ask']:.2f} / {TEST_ACC['never-ask']:.2f}; "
        f"grid regret {GRID_BASE['always-ask']:.4f} / {GRID_BASE['never-ask']:.4f}.\n"]
NAMES = {"tact": "**full TACT**", "q_const": "constant cost (q_const)", "q_bucket": "emotion-bucketed cost (q_bucket)", "tact_noemo": "TACT − emotion (ablation)",
         "q_const_opt": "extra: constant cost, c fitted on training regret", "tact_ask_const": "extra: TACT ask side vs constant c"}
for n in (150, 350):
    out += [f"\n### N = {n}\n",
            "| method | test regret | test agreement with oracle | test_unseen regret | grid regret (normalised) | paired Δ test regret vs TACT (method − TACT) | seeds TACT better | Wilcoxon p |", "|---|---|---|---|---|---|---|---|"]
    base = sel(cr, n=n, method="tact", **{"eval": "seen"}, subset="all")
    for m in ("tact", "q_const", "q_bucket", "tact_noemo", "q_const_opt", "tact_ask_const"):
        s = sel(cr, n=n, method=m, **{"eval": "seen"}, subset="all")
        u = sel(cr, n=n, method=m, **{"eval": "unseen"}, subset="all")
        gg = sel(cr, n=n, method=m, **{"eval": "grid"}, subset="all")
        if m == "tact":
            pd_, k_, p_ = "–", "–", "–"
        else:
            pd_, _, p_ = paired(s.regret.values, base.regret.values)
            k_ = f"{int((s.regret.values - base.regret.values > 0).sum())}/5"      # seeds in which TACT has the lower regret
        out.append(f"| {NAMES[m]} | {ms(s.regret)} | {ms(s.acc, 2)} | {ms(u.regret)} | {ms(gg.regret / min(GRID_BASE.values()), 2)} | {pd_} | {k_} | {p_} |")
    out.append(f"| always-ask / never-ask | {TEST_BASE['always-ask']:.3f} / {TEST_BASE['never-ask']:.3f} | {TEST_ACC['always-ask']:.2f} / {TEST_ACC['never-ask']:.2f} | – | "
               f"{GRID_BASE['always-ask'] / min(GRID_BASE.values()):.2f} / {GRID_BASE['never-ask'] / min(GRID_BASE.values()):.2f} | – | – | – |")

# ================================================================== 2
out += ["\n---\n## 2. What receptivity (emotion) contributes to the decision (reviewer item 2)\n",
        "### 2a. How often the optimal decision depends on the emotion (noise-free truth)\n",
        f"A (topic, day, time) triple is **emotion-sensitive** if the oracle decision (ask iff clean ask ≥ clean act) is not the same for all six emotions. Of the {int(info['n_triples'])} triples "
        f"(12 topics × 7 days × 3 times) **{int(info['n_sens_triples'])} ({info['n_sens_triples'] / info['n_triples']:.1%})** are emotion-sensitive; they hold {int(info['n_sens_cells'])} of the {int(info['n_cells'])} grid "
        f"cells ({info['n_sens_cells'] / info['n_cells']:.1%}) and **{int(info['n_sens_test'])} of the {int(info['n_test'])} test scenarios ({info['n_sens_test'] / info['n_test']:.1%})**. "
        f"The oracle acts in {info['grid_oracle_act_rate']:.1%} of the grid cells. Upper bound on what emotion can add: an oracle that knows the truth except the emotion (it picks, per triple, the decision with the higher "
        f"emotion-averaged value) has grid regret **{info['blind_oracle_grid_regret']:.4f}** (always-ask {GRID_BASE['always-ask']:.4f}, never-ask {GRID_BASE['never-ask']:.4f}, perfect 0): knowing the emotion can remove at most "
        f"{info['blind_oracle_grid_regret']:.4f} of the {min(GRID_BASE.values()):.4f} regret of the best constant policy ({info['blind_oracle_grid_regret'] / min(GRID_BASE.values()):.1%}); the remaining {1 - info['blind_oracle_grid_regret'] / min(GRID_BASE.values()):.1%} "
        "is decided by (topic, day, time). The decision flips with the emotion in most triples, but mostly where the two values are close.\n",
        "Share of each topic's 126 grid cells that lie in emotion-sensitive triples:\n", "| topic | share |", "|---|---|"]
out += [f"| {r.topic} | {r.share_of_cells_in_sensitive_triples:.2f} |" for r in share.itertuples()]
out += ["\n### 2b. Regret on the emotion-sensitive and the emotion-insensitive subset\n",
        "Subset regret = mean regret over the scenarios (cells) of the subset. Test: 49 sensitive / 26 insensitive scenarios (small subsets, noisy); grid: 1140 / 372 cells (evaluation of the learned model, uniform over the grid). "
        "'tabular' = the paper's tabular agent (lr 0.5, topic back-off); 'tabular (tdt)' = extra, the variant of experiment 4 whose back-off is the (topic, day, time) mean (see §4).\n"]
tg = lambda n, chain: tt[(tt.n == n) & (tt.lr == "0.5") & (tt.chain == chain)].sort_values("seed")
for n in (150, 350):
    out += [f"\n#### N = {n}\n", "| method | test, sensitive (49) | test, insensitive (26) | grid, sensitive (1140) | grid, insensitive (372) |", "|---|---|---|---|---|"]
    for m in ("tact", "tact_noemo", "q_const", "q_bucket", "q_const_opt", "tact_ask_const"):
        cells = [ms(sel(cr, n=n, method=m, **{"eval": e}, subset=s).regret) for e, s in (("seen", "sens"), ("seen", "insens"), ("grid", "sens"), ("grid", "insens"))]
        out.append(f"| {NAMES[m]} | " + " | ".join(cells) + " |")
    for lab, chain in (("tabular (paper)", "full"), ("extra: tabular (tdt)", "tdt")):
        x = tg(n, chain)
        out.append(f"| {lab} | " + " | ".join(ms(x[c]) for c in ("seen_regret_sens", "seen_regret_insens", "grid_regret_sens", "grid_regret_insens")) + " |")
    out += [f"\nPaired difference −emotion minus full TACT (positive = emotion helps), N = {n}:\n", "| subset | Δ regret (−emotion − TACT) | seeds where emotion helps | Wilcoxon p |", "|---|---|---|---|"]
    for e, s, lab in (("seen", "all", "test, all 75"), ("seen", "sens", "test, sensitive"), ("seen", "insens", "test, insensitive"), ("grid", "all", "grid, all"), ("grid", "sens", "grid, sensitive"), ("grid", "insens", "grid, insensitive")):
        a, b = sel(cr, n=n, method="tact_noemo", **{"eval": e}, subset=s).regret.values, sel(cr, n=n, method="tact", **{"eval": e}, subset=s).regret.values
        _, k, p = paired(b, a, better="lower")     # k: seeds in which TACT (b) has the lower regret
        dd = a - b
        out.append(f"| {lab} | {dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(5):.4f} | {k} | {p} |")

# ================================================================== 3
gb = rd("binary_feedback_runs.csv")
gd = rd("binary_feedback_draws_runs.csv")
out += ["\n---\n## 3. Graded feedback: how coarse may the feedback be? (reviewer item 5)\n",
        "K-level labels replace the continuous labels in training and validation (ask and act rows); evaluation truth unchanged. **direct** (primary) = the user's noisy continuous rating rounded to the nearest of K levels (K = 3: "
        "{0, .5, 1}; K = 5: {0, .25, .5, .75, 1}); **stochastic** = unbiased randomised rounding of the noise-free value (for K = 2 this is exactly the Bernoulli feedback of the binary experiment). **mse** (primary: the main "
        "experiment's loss, so only the label coarseness changes) and **bce** are both given. N = 350 on `data_dpo` (75 test scenarios), N = 700 on `data_cont_n1000` (150 test scenarios). "
        "L = Saturday-fruit learning degree (1 − collapse ratio) / (1 − 0.15). 'grid' = full-grid regret / min(always-ask, never-ask on the grid). End points: continuous (re-run here, identical to `binary_feedback_runs.csv`) "
        "and binary (Bernoulli, BCE, `binary_feedback_runs.csv`; 'binary, 3 draws' pools the three independent Bernoulli label draws of `binary_feedback_draws_runs.csv`, 15 runs).\n"]
for size, n in ((500, 350), (1000, 700)):
    out += [f"\n### N = {n} ({'data_dpo' if size == 500 else 'data_cont_n1000'})\n",
            "| labels | test regret | Δ vs continuous (paired) | seeds better than continuous | test agreement | grid regret (normalised) | L Saturday-fruit |", "|---|---|---|---|---|---|---|"]
    cont = gr[(gr.n == n) & (gr.label == "continuous")].sort_values("seed")
    rows = [("continuous (re-run)", cont)]
    for mode in ("direct", "stochastic"):
        for K in (5, 3):
            for loss in ("mse", "bce"):
                x = gr[(gr.n == n) & (gr.label == f"K={K}") & (gr["mode"] == mode) & (gr["loss"] == loss)].sort_values("seed")
                tag = " **(primary)**" if mode == "direct" and loss == "mse" else ""
                rows.append((f"K = {K}, {mode}, {loss}{tag}", x))
    for lab, x in rows:
        if lab.startswith("continuous"):
            dd, kk = "–", "–"
        else:
            dd, kk, _ = paired(x.regret_seen.values, cont.regret_seen.values)
        out.append(f"| {lab} | {ms(x.regret_seen)} | {dd} | {kk} | {ms(x.acc_seen, 2)} | {ms(x.grid_norm, 2)} | {ms(x.L, 2)} |")
    b0 = gb[(gb.label == "bin") & (gb["size"] == size) & (gb.n == n)].sort_values("seed")
    out.append(f"| binary K = 2 (Bernoulli, BCE, draw 0) | {ms(b0.A_seen)} | {paired(b0.A_seen.values, cont.regret_seen.values)[0]} | {paired(b0.A_seen.values, cont.regret_seen.values)[1]} | {ms(b0.A_acc_seen, 2)} | "
               f"{ms(b0.A_grid / b0.base_grid, 2)} | {ms((1 - b0.sat_ratio) / (1 - TRUE_RATIO), 2)} |")
    dr = gd[(gd["size"] == size) & (gd.n == n)]
    if len(dr) and 0 not in set(dr.draw):          # the draws file holds draws 1 and 2 only (draw 0 is `binary_feedback_runs.csv`; at N = 350 it is re-run in the draws file as an integrity check)
        dr = pd.concat([b0.assign(draw=0), dr], ignore_index=True)
    if len(dr):
        out.append(f"| binary, 3 draws pooled ({len(dr)} runs) | {ms(dr.A_seen)} | – | – | {ms(dr.A_acc_seen, 2)} | {ms(dr.A_grid / dr.base_grid, 2)} | {ms((1 - dr.sat_ratio) / (1 - TRUE_RATIO), 2)} |")

# ================================================================== 4
CH = {"full": "full chain (paper)", "notopic": "no topic level", "tdt": "extra: (topic, day, time) level", "te": "extra: (topic, emotion) level"}
NS = sorted(tt.n.unique())
out += ["\n---\n## 4. Is the tabular baseline under-tuned? (reviewer item 13)\n",
        "Specified search: learning rate {0.1, 0.3, 0.5, 0.8, 1.0, 1/visits} × back-off chain {full (cell → (topic, action) mean → global), no topic level (cell → global)} = 12 variants. **Extra, not part of the specified search:** "
        "two finer chains, 'tdt' (cell → (topic, day, time) mean → topic mean → global) and 'te' (cell → (topic, emotion) mean → topic mean → global), 12 more variants. Selection per N by validation regret (clean values of the 75 validation "
        "scenarios; favours the baseline); ties go to the paper's configuration.\n",
        "### 4a. Why the learning rate does not matter\n",
        "Share of the training updates that hit an already visited cell (topic, emotion, day, time, action) — only these updates depend on the learning rate: " +
        ", ".join(f"N = {n}: {tt[(tt.n == n) & (tt.lr == '0.5') & (tt.chain == 'full')].revisit_share.mean():.1%}" for n in NS) + ". The 1512 cells are rarely revisited, so all learning rates give (nearly) the same decisions.\n",
        "### 4b. Mean test regret of every variant (5 seeds)\n", "| chain | learning rate | " + " | ".join(f"N = {n}" for n in NS) + " |", "|---|---|" + "---|" * len(NS)]
for ch in CH:
    for lr in ("0.1", "0.3", "0.5", "0.8", "1.0", "decay"):
        x = tt[(tt.chain == ch) & (tt.lr == lr)].groupby("n").seen_regret.mean()
        out.append(f"| {CH[ch]} | {lr} | " + " | ".join(f"{x[n]:.4f}" for n in NS) + " |")
SPEC = ["full", "notopic"]
EXT = SPEC + ["tdt", "te"]


def pick(chains):
    """per N the (lr, chain) with the lowest mean validation regret among `chains`; ties -> paper config first, then list order."""
    out_ = {}
    for n in NS:
        g = tt[(tt.n == n) & (tt.chain.isin(chains))].groupby(["lr", "chain"]).val_regret.mean()
        order = [("0.5", "full")] + [(lr, c) for c in chains for lr in ("0.1", "0.3", "0.5", "0.8", "1.0", "decay")]
        best = min(order, key=lambda k: (round(g[k], 12), order.index(k)))
        out_[n] = best
    return out_


def curve(chosen):
    return {n: tt[(tt.n == n) & (tt.lr == chosen[n][0]) & (tt.chain == chosen[n][1])].sort_values("seed") for n in NS}


ps, pe = pick(SPEC), pick(EXT)
cs, ce = curve(ps), curve(pe)
dflt = curve({n: ("0.5", "full") for n in NS})
tact_c = {n: BA(n).regret.values for n in NS}
for title, ch, cv in (("4c. Selected configuration, specified search (12 variants)", ps, cs), ("4d. Selected configuration, extended search (24 variants, includes the extra chains)", pe, ce)):
    out += [f"\n### {title}\n", "| N | selected (lr, chain) | validation regret | test regret, selected | test regret, paper tabular | test regret, TACT | Δ TACT − selected tabular (paired) | seeds TACT better | Wilcoxon p | agreement with oracle, selected |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for n in NS:
        x = cv[n]
        d, k, p = paired(tact_c[n], x.seen_regret.values)
        out.append(f"| {n} | {ch[n][0]}, {CH[ch[n][1]]} | {x.val_regret.mean():.4f} | {ms(x.seen_regret, 4)} | {dflt[n].seen_regret.mean():.4f} | {ms(tact_c[n], 4)} | {d} | {k} | {p} | {x.seen_acc.mean():.3f} |")
    m_t = {n: tact_c[n].mean() for n in NS}
    m_s = {n: cv[n].seen_regret.mean() for n in NS}
    fa = first_ahead(m_t, m_s, NS)
    out.append("\nMean curves: " + (f"TACT is below the selected tabular for every N from {fa} on." if fa else "there is no N from which TACT is below the selected tabular for all larger N.") +
               " Mean difference per N (TACT − tabular): " + ", ".join(f"N = {n}: {m_t[n] - m_s[n]:+.4f}" for n in NS) + ".")
m_def = {n: dflt[n].seen_regret.mean() for n in NS}
out.append(f"\nFor reference, against the paper's tabular: TACT below it for every N from {first_ahead({n: tact_c[n].mean() for n in NS}, m_def, NS)} on.")
best_test = {n: tt[(tt.n == n) & (tt.chain.isin(SPEC))].groupby(["lr", "chain"]).seen_regret.mean().min() for n in NS}
best_test_e = {n: tt[(tt.n == n)].groupby(["lr", "chain"]).seen_regret.mean().min() for n in NS}
out += ["\nUpper bounds (best variant ON THE TEST SET, not selectable in practice): specified search " + ", ".join(f"N = {n}: {best_test[n]:.4f}" for n in NS) +
        "; extended search " + ", ".join(f"N = {n}: {best_test_e[n]:.4f}" for n in NS) + "."]
out += ["\n### 4e. Grid regret and phrasing\n",
        "Grid regret (1512 cells, clean truth) and test_unseen regret of the paper's tabular and the validation-selected extended variant; the tabular keys on the topic, so its decisions cannot change with the question phrasing.\n",
        "| N | grid regret, paper tabular | grid regret, selected (extended) | test_unseen regret, paper tabular | test_unseen regret, selected (extended) |", "|---|---|---|---|---|"]
for n in NS:
    out.append(f"| {n} | {dflt[n].grid_regret.mean():.4f} | {ce[n].grid_regret.mean():.4f} | {dflt[n].unseen_regret.mean():.4f} | {ce[n].unseen_regret.mean():.4f} |")
out.append("\nTACT grid regret at N = 350 for comparison: " + f"{sel(cr, n=350, method='tact', **{'eval': 'grid'}, subset='all').regret.mean():.4f}; at N = 150: {sel(cr, n=150, method='tact', **{'eval': 'grid'}, subset='all').regret.mean():.4f} (the other N were not re-run on the grid).")

# ================================================================== 5
out += ["\n---\n## 5. Paired statistics for the key comparisons (reviewer item 6)\n",
        "Per-seed paired differences, mean ± SE, seeds where the first-named method is better, exact two-sided Wilcoxon signed-rank p (n = 5 seeds: p ≥ 0.0625 always; n = 10 users: p ≥ 0.002). "
        "The tabular agent hardly varies over seeds (same first-N scenarios; the seed only shuffles the update order, which changes decisions in at most a few cells, std ≤ 0.0001), so its 'paired' partner is practically a constant; the seed variation is TACT's.\n",
        "### 5a. TACT vs tabular, every N (test_seen regret; Δ = TACT − tabular, negative = TACT better)\n",
        "| N | TACT | tabular (paper) | Δ (paired) | seeds TACT better | Wilcoxon p |", "|---|---|---|---|---|---|"]
C = bud[bud.method == "C"]
for n in NS:
    c = C[C.n == n].sort_values("seed").regret.values
    d, k, p = paired(tact_c[n], c)
    out.append(f"| {n} | {ms(tact_c[n], 4)} | {c.mean():.4f} | {d} | {k} | {p} |")
out += ["\nSame comparison against the extra tabular variant selected on validation from the extended search (§4d; it is NOT the tabular of the paper):\n", "| N | TACT | tabular, extended-search pick | Δ (paired) | seeds TACT better | Wilcoxon p |", "|---|---|---|---|---|---|"]
for n in NS:
    d, k, p = paired(tact_c[n], ce[n].seen_regret.values)
    out.append(f"| {n} | {ms(tact_c[n], 4)} | {ce[n].seen_regret.mean():.4f} | {d} | {k} | {p} |")

rp = rd("replay_metrics.csv")
w7 = rp[(rp.strategy == "warm") & (rp["round"] == 7)].sort_values("seed")
s7 = rp[(rp.strategy == "scratch") & (rp["round"] == 7)].sort_values("seed")
dw = rp[(rp.strategy == "warm") & (rp["round"] >= 2)].groupby("seed").drift.mean()
ds = rp[(rp.strategy == "scratch") & (rp["round"] >= 2)].groupby("seed").drift.mean()
so = rd("dpo_semionline.csv")
so7 = so[(so.method == "A") & (so.test == "seen") & (so["round"] == 7)].sort_values("seed")
out += ["\n### 5b. Semi-online (warm start) vs from scratch at 350 interactions (round 7)\n",
        "| quantity | warm start (semi-online) | from scratch / one-shot | Δ (warm − other) | seeds warm better | Wilcoxon p |", "|---|---|---|---|---|---|"]
d, k, p = paired(w7.mse.values, s7.mse.values, 5)
out.append(f"| reward-model test MSE at round 7 (replay_metrics.csv) | {ms(w7.mse, 4)} | {ms(s7.mse, 4)} | {d} | {k} | {p} |")
d, k, p = paired(dw.values, ds.values, 4)
out.append(f"| prediction drift, mean over rounds 2–7 (replay_metrics.csv) | {ms(dw.values, 4)} | {ms(ds.values, 4)} | {d} | {k} | {p} |")
d, k, p = paired(so7.regret.values, BA(350).regret.values, 4)
out.append(f"| decision regret on test_seen: semi-online round 7 (dpo_semionline.csv, method A) vs one-shot N = 350 (budget_runs.csv, method A) | {ms(so7.regret, 4)} | {ms(BA(350).regret, 4)} | {d} | {k} | {p} |")
d, k, p = paired(rp[(rp.strategy == "warm") & (rp["round"] == 7)].sort_values("seed").spearman.values, rp[(rp.strategy == "scratch") & (rp["round"] == 7)].sort_values("seed").spearman.values, 4, better="higher")
out.append(f"| Spearman at round 7 (replay_metrics.csv; higher is better) | {ms(w7.spearman, 4)} | {ms(s7.spearman, 4)} | {d} | {k} | {p} |")

ab = rd("ablation_threshold_vs_dpo.csv")
out += ["\n### 5c. Threshold rule vs SFT + DPO policy on the same reward model (regret; Δ = threshold − DPO, negative = threshold better)\n",
        "| N | test set | threshold | DPO | Δ (paired) | seeds threshold better | Wilcoxon p |", "|---|---|---|---|---|---|---|"]
for n in (150, 350):
    x = ab[ab.n == n].sort_values("seed")
    for ts in ("seen", "hard"):
        d, k, p = paired(x[f"threshold_{ts}"].values, x[f"dpo_{ts}"].values)
        out.append(f"| {n} | {ts} | {ms(x[f'threshold_{ts}'], 4)} | {ms(x[f'dpo_{ts}'], 4)} | {d} | {k} | {p} |")
for ts in ("seen", "hard"):
    d, k, p = paired(ab[f"threshold_{ts}"].values, ab[f"dpo_{ts}"].values)
    out.append(f"| all 7 N × 5 seeds (35 pairs, not independent) | {ts} | {ab[f'threshold_{ts}'].mean():.4f} | {ab[f'dpo_{ts}'].mean():.4f} | {d} | {k} | {p} |")

MA = pd.read_csv(os.path.join(R, "multiuser_matrix_A.csv"), index_col=0).values
diag = np.diag(MA)
row_off = np.array([np.delete(MA[i], i).mean() for i in range(10)])
col_off = np.array([np.delete(MA[:, i], i).mean() for i in range(10)])
out += ["\n### 5d. 10 users: own-truth (diagonal) vs other users' truths (off-diagonal), paired by user (normalised grid regret; Δ = diagonal − off-diagonal, negative = personalisation helps)\n",
        "| comparison | diagonal mean | off-diagonal mean | Δ (paired by user) | users with diagonal better | Wilcoxon p |", "|---|---|---|---|---|---|"]
d, k, p = paired(diag, row_off, 3)
out.append(f"| user i's model on its own truth vs on the other 9 users' truths (row) | {diag.mean():.3f} | {row_off.mean():.3f} | {d} | {k} | {p} |")
d, k, p = paired(diag, col_off, 3)
out.append(f"| user i's truth, own model vs the other 9 users' models (column) | {diag.mean():.3f} | {col_off.mean():.3f} | {d} | {k} | {p} |")

# ================================================================== 1/2 extra paired tests
out += ["\n---\n## Appendix: paired tests for experiments 1 and 2 (Δ = baseline − TACT on test_seen regret; positive = TACT better)\n", "| N | baseline | Δ regret | seeds TACT better | Wilcoxon p |", "|---|---|---|---|---|"]
for n in (150, 350):
    base = sel(cr, n=n, method="tact", **{"eval": "seen"}, subset="all").regret.values
    for m in ("q_const", "q_bucket", "tact_noemo", "q_const_opt", "tact_ask_const"):
        x = sel(cr, n=n, method=m, **{"eval": "seen"}, subset="all").regret.values
        d, k, p = paired(base, x)
        dd = x - base
        out.append(f"| {n} | {NAMES[m]} | {dd.mean():+.4f} ± {dd.std(ddof=1) / np.sqrt(5):.4f} | {k} | {p} |")

with open(os.path.join(RB, "rebuttal.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
