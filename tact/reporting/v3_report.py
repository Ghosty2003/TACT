"""v3 report: results/v3_recency.md (old default vs recency-weighted default, side by side) and figures_v3/fig_semionline.pdf, figures_v3/fig_drift.pdf (+ png).

    TACT_V2=1 python -m tact.reporting.v3_report
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from scipy.stats import wilcoxon

from tact.core.model import ROOT
V2, V3 = os.path.join(ROOT, "results", "v2"), os.path.join(ROOT, "results", "v3")
FIG = os.path.join(ROOT, "figures_v3")
os.makedirs(FIG, exist_ok=True)
H = json.load(open(os.path.join(V3, "chosen_half_life.json")))["half_life"]
NEWM = f"R{H}"
PER, DRIFT_I, T_END, TARGET, CAP = 50, 150, 750, 0.05, 601
HAB3 = ("bread", "coffee", "snack")
rd2 = lambda f: pd.read_csv(os.path.join(V2, f))
rd3 = lambda f: pd.read_csv(os.path.join(V3, f))


def ms(x, nd=4):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 and x.std() > 0 else f"{x.mean():.{nd}f}"


def paired(a, b, nd=4, better="lower"):
    d = np.asarray(a, float) - np.asarray(b, float)
    k = int((d < 0).sum() if better == "lower" else (d > 0).sum())
    try:
        p = wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    except ValueError:
        p = float("nan")
    return f"{d.mean():+.{nd}f} ± {d.std(ddof=1) / np.sqrt(len(d)):.{nd}f}", f"{k}/{len(d)}", f"{p:.3f}"


FLAGS = []      # (where in the paper, quantity, old, new)


def flag(where, what, old, new):
    FLAGS.append((where, what, old, new))


out = [f"# v3: recency-weighted replay as the default semi-online retraining (`results/v3_recency.md`)\n",
       f"Chosen half-life: **{H} rounds** (selected on validation seeds 5–9 only, §1). New default = warm start on all data, every scenario weighted 0.5^(age in rounds / {H}) (age 0 = the round just added), unweighted validation loss, early stopping "
       "as before; the decision rule, architecture, hyper-parameters and the 5 reporting seeds (0–4) are unchanged. 'old' = the unweighted warm start of the paper (`results/v2`), 'new' = recency-weighted. Mean ± std over the 5 seeds; paired "
       "differences are per seed (mean ± SE, seeds in which the new default is better, exact two-sided Wilcoxon p; with 5 seeds p ≥ 0.0625). No git commit, no push; nothing in `results/v2`, `figures_v2`, `figures_final` was modified "
       "(new files are in `results/v3/`, `figures_v3/`). The targeted-reset results stay in `results/drift_targeted.md` as an archived negative result (not for the paper).\n",
       "**Contents:** §0 what changes in the paper (summary) · §1 half-life selection · §2 Sec. 5.4 semi-online stability · §3 Appendix C habit drift · §4 Appendix E partial feedback and the cold-start rule · §5 which experiments use the loop · §6 flagged numbers.\n"]

# =================================================================== 1
sel = open(os.path.join(V3, "select.md")).read().split("\n", 1)[1]
out += ["\n---\n## 1. Half-life selection (validation seeds only)\n", sel.replace("## Validation table", "### Validation table")]

# =================================================================== 2
rp_old = rd2("replay_metrics.csv")
rp = rd3("semionline_replay.csv")
bu_old = rd2("dpo_semionline.csv")
bu = rd3("semionline_budget.csv")
bud2 = rd2("budget_runs.csv")
get = lambda df, **kw: df[np.logical_and.reduce([df[k] == v for k, v in kw.items()])].sort_values("seed")
out += ["\n---\n## 2. Sec. 5.4 semi-online stability (7 rounds × 50 interactions)\n",
        "Two protocols of the paper. **(A)** `replay.py` (ask-only `train.csv`, reward-model MSE on `test.csv`, prediction drift = mean |prediction change| on `test.csv` between consecutive rounds); **(B)** `budget.semi_online` (`data_dpo`, ask + act rows, threshold rule, "
        "regret on the 75 test scenarios). In the new rows BOTH the warm start and the from-scratch retraining use the recency weights; 'from scratch' retrains the model on all data so far with the same weights, 'warm start' continues from the previous round.\n"]
# integrity
chk = []
for strat in ("scratch", "warm"):
    a = get(rp, weighting="unweighted", strategy=strat, round=7).mse.values
    b = get(rp_old, strategy=strat, round=7).mse.values
    chk.append(np.abs(a - b).max())
dd = rp[(rp.weighting == "unweighted")].sort_values(["strategy", "seed", "round"])
oo = rp_old.sort_values(["strategy", "seed", "round"])
full_chk = float(np.nanmax(np.abs(dd.drift.values - oo.drift.values)))
bchk = np.abs(get(bu, variant="warm_unweighted", test="seen", round=7).regret.values - get(bu_old, method="A", test="seen", round=7).regret.values).max()
out.append(f"**Integrity checks.** (i) The unweighted `budget.semi_online` reproduces `dpo_semionline.csv` exactly (round-7 regret, max |difference| {bchk:.1e}). (ii) Re-running `replay.py`'s protocol without weights reproduces round 1 of "
           f"`results/v2/replay_metrics.csv` (3e-9) but NOT the later rounds exactly: max |difference| in MSE over all rounds {np.abs(dd.mse.values - oo.mse.values).max():.1e} (round 7: scratch {chk[0]:.1e}, warm {chk[1]:.1e}), in drift {full_chk:.1e}. "
           "The cause is numerical, not logical: `replay.py` was run without deterministic-algorithm flags, while the `budget` module imported by the v3 code sets them, so early stopping / warm-start trajectories differ at the level of the seed noise. "
           "Consequently §2a/§2b show the paper's old numbers AND, in §2f, the unweighted rerun of the same v3 code; the effect of the weights should be read against the latter (same code, same numerics), and it is within the seed noise either way.\n")

out += ["### 2a. Reward-model test MSE per round (protocol A)\n", "| round | scenarios | old warm | new warm (recency) | old scratch | new scratch (recency) |", "|---|---|---|---|---|---|"]
for r in range(1, 8):
    out.append(f"| {r} | {50 * r} | {ms(get(rp_old, strategy='warm', round=r).mse)} | {ms(get(rp, weighting='recency', strategy='warm', round=r).mse)} | "
               f"{ms(get(rp_old, strategy='scratch', round=r).mse)} | {ms(get(rp, weighting='recency', strategy='scratch', round=r).mse)} |")
nw, ns = get(rp, weighting="recency", strategy="warm", round=7), get(rp, weighting="recency", strategy="scratch", round=7)
ow, os_ = get(rp_old, strategy="warm", round=7), get(rp_old, strategy="scratch", round=7)
d1, k1, p1 = paired(nw.mse.values, ns.mse.values, 5)
d0, k0, p0 = paired(ow.mse.values, os_.mse.values, 5)
out += [f"\nRound 7, warm − scratch (paired by seed): old {d0} ({k0} seeds warm better, p = {p0}); **new {d1}** ({k1}, p = {p1}). New warm vs old warm: {paired(nw.mse.values, ow.mse.values, 5)[0]}; "
        f"Spearman at round 7 old warm / scratch {ms(ow.spearman)} / {ms(os_.spearman)}, new warm / scratch {ms(nw.spearman)} / {ms(ns.spearman)}.\n"]
flag("Sec. 5.4 (and §8c / rebuttal 5b)", "round-7 test MSE, warm / scratch", f"{ms(ow.mse)} / {ms(os_.mse)}", f"{ms(nw.mse)} / {ms(ns.mse)}")

out += ["### 2b. Mean prediction drift between consecutive rounds (protocol A)\n", "| round | old warm | new warm (recency) | old scratch | new scratch (recency) |", "|---|---|---|---|---|"]
for r in range(2, 8):
    out.append(f"| {r} | {ms(get(rp_old, strategy='warm', round=r).drift)} | {ms(get(rp, weighting='recency', strategy='warm', round=r).drift)} | "
               f"{ms(get(rp_old, strategy='scratch', round=r).drift)} | {ms(get(rp, weighting='recency', strategy='scratch', round=r).drift)} |")
def mean_drift(df, **kw):
    x = df[df["round"] >= 2]
    for k, v in kw.items():
        x = x[x[k] == v]
    return x.groupby("seed").drift.mean().sort_index()
mo_w, mo_s = mean_drift(rp_old, strategy="warm"), mean_drift(rp_old, strategy="scratch")
mn_w, mn_s = mean_drift(rp, weighting="recency", strategy="warm"), mean_drift(rp, weighting="recency", strategy="scratch")
out += ["\nMean over rounds 2–7 (per seed, then over seeds):\n", "| | warm | scratch | warm − scratch (paired) | seeds warm lower | Wilcoxon p |", "|---|---|---|---|---|---|"]
for lab, w, s in (("old (unweighted)", mo_w, mo_s), (f"**new (recency, h = {H})**", mn_w, mn_s)):
    d, k, p = paired(w.values, s.values)
    out.append(f"| {lab} | {ms(w.values)} | {ms(s.values)} | {d} | {k} | {p} |")
out.append(f"\nNew warm vs old warm: {paired(mn_w.values, mo_w.values)[0]} ({paired(mn_w.values, mo_w.values)[1]} seeds lower); new scratch vs old scratch: {paired(mn_s.values, mo_s.values)[0]}.")
flag("Sec. 5.4 / Fig. fig_semionline (b)", "mean prediction drift, warm / scratch", f"{ms(mo_w.values)} / {ms(mo_s.values)}", f"{ms(mn_w.values)} / {ms(mn_s.values)}")

# 2c regret
a_old = get(bu_old, method="A", test="seen", round=7)
w_new = get(bu, variant="warm_recency", test="seen", round=7)
s_new = get(bu, variant="scratch_recency", test="seen", round=7)
s_unw = get(bu, variant="scratch_unweighted", test="seen", round=7)
one = bud2[(bud2.method == "A") & (bud2.test == "seen") & (bud2.n == 350)].sort_values("seed")
out += ["\n### 2c. Decision regret after 350 interactions (protocol B, 75 test scenarios, test_seen) and the one-shot comparison\n",
        "| model | regret after 350 interactions | paired Δ vs old warm | seeds better than old warm | Wilcoxon p |", "|---|---|---|---|---|"]
for lab, x in (("**old: warm start, unweighted (paper)**", a_old), (f"**new: warm start, recency (h = {H})**", w_new), (f"new: from scratch, recency (h = {H})", s_new),
               ("reference: from scratch, unweighted (semi-online seeds)", s_unw), ("reference: one-shot offline, N = 350 (`budget_runs.csv`, the 0.036)", one)):
    d, k, p = ("–", "–", "–") if x is a_old else paired(x.regret.values, a_old.regret.values)
    out.append(f"| {lab} | {ms(x.regret.values)} | {d} | {k} | {p} |")
d, k, p = paired(w_new.regret.values, one.regret.values)
out.append(f"\nNew warm start against the one-shot model (paired by seed): {d} ({k} seeds lower, p = {p}); old warm start against the one-shot: {paired(a_old.regret.values, one.regret.values)[0]}.")
flag("Sec. 5.4 / §6b table", "semi-online regret after 350 interactions (warm) vs one-shot 0.036", f"{ms(a_old.regret.values)} vs {ms(one.regret.values)}", f"{ms(w_new.regret.values)} vs {ms(one.regret.values)}")
out += ["\n### 2d. Regret and utility U after every round (protocol B; U = accuracy / (1 + ask rate)), test_seen\n",
        "| round | scenarios | old warm: regret | new warm: regret | new scratch: regret | tabular C: regret | old warm: U | new warm: U | new scratch: U |", "|---|---|---|---|---|---|---|---|---|"]
U = lambda df: df.acc.values / (1.0 + (1.0 - df.act_rate.values))
for r in range(1, 8):
    ow_, nw_, ns_ = get(bu_old, method="A", test="seen", round=r), get(bu, variant="warm_recency", test="seen", round=r), get(bu, variant="scratch_recency", test="seen", round=r)
    c_ = get(bu_old, method="C", test="seen", round=r)
    out.append(f"| {r} | {50 * r} | {ms(ow_.regret.values)} | {ms(nw_.regret.values)} | {ms(ns_.regret.values)} | {ms(c_.regret.values)} | {ms(U(ow_), 3)} | {ms(U(nw_), 3)} | {ms(U(ns_), 3)} |")
out.append("\n(The tabular agent C does not depend on the retraining rule; its numbers are those of `dpo_semionline.csv`.)")
sat = rp[(rp["round"] == 7) & (rp.weighting == "recency")]
out += ["\n### 2e. Saturday-fruit collapse ratio of the round-7 models (heatmap statistic, truth 0.15)\n", "| model | Saturday / other days |", "|---|---|"]
for lab, df in (("old from scratch, unweighted (the model of the paper's heatmap, `results/v2/models`)", rp[(rp["round"] == 7) & (rp.weighting == "unweighted") & (rp.strategy == "scratch")]),
                ("old warm, unweighted", rp[(rp["round"] == 7) & (rp.weighting == "unweighted") & (rp.strategy == "warm")]), (f"new warm, recency", sat[sat.strategy == "warm"]), (f"new scratch, recency", sat[sat.strategy == "scratch"])):
    out.append(f"| {lab} | {ms(df.sat_over_other.values, 3)} |")
out.append("\nFig. fig_heatmap is drawn from the unweighted from-scratch model (one-shot at 350 scenarios) and is **not** changed by the new default; if the paper's heatmap should come from a recency-weighted model, say so.")

rr = lambda w_, st_: rp[(rp.weighting == w_) & (rp.strategy == st_)]
out += ["\n### 2f. Same-code baseline: unweighted rerun of the v3 code against the recency-weighted runs (protocol A, 5 seeds)\n",
        "| | round-7 test MSE | mean drift, rounds 2–7 |", "|---|---|---|"]
for w_ in ("unweighted", "recency"):
    for st_ in ("warm", "scratch"):
        x = rr(w_, st_)
        out.append(f"| {'old (unweighted, v3 rerun)' if w_ == 'unweighted' else f'new (recency, h = {H})'}, {st_} | {ms(x[x['round'] == 7].mse.values)} | {ms(x[x['round'] >= 2].groupby('seed').drift.mean().values)} |")
out.append("\nWarm start vs scratch in the same-code runs: MSE differences and drift differences have the same sign and size as in §2a/§2b.")

# figure fig_semionline (new default only)
W_ = 84 / 25.4
plt.style.use("default")
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "lines.linewidth": 1.2, "lines.markersize": 3.2, "pdf.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic"})
TACT, ALT, GREY, ORANGE = "#0072B2", "#D55E00", "#9a9a9a", "#E69F00"


def save_fig(fig, name):
    assert all(not ax.get_title() for ax in fig.axes)
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    assert min(sizes) >= 7.0, min(sizes)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


fig, axes = plt.subplots(1, 2, figsize=(W_, 1.95), constrained_layout=True)
for ax, col, ylab, first, letter in ((axes[0], "mse", "test MSE", 1, "(a)"), (axes[1], "drift", "prediction drift", 2, "(b)")):
    for strat, lab, colr in (("warm", "warm start", TACT), ("scratch", "from scratch", ALT)):
        g = rp[(rp.weighting == "recency") & (rp.strategy == strat) & (rp["round"] >= first)].groupby("n_train")[col].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], "-o", color=colr, label=lab)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colr, alpha=0.18, lw=0)
    ax.set_xticks([100, 200, 300])
    ax.set_xlabel("training samples")
    ax.set_ylabel(ylab)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")
axes[0].legend(frameon=False, loc="upper right", handlelength=1.3, borderaxespad=0.1)
save_fig(fig, "fig_semionline")

# =================================================================== 3
sd, rd_, fd = rd2("drift_targeted_series.csv"), rd2("drift_targeted_rounds.csv"), rd2("drift_targeted_final.csv")
sn, rn, fn = rd3("drift_series.csv"), rd3("drift_rounds.csv"), rd3("drift_final.csv")
tab = rd2("drift_trigger_series.csv")
tab = tab[(tab.label == "cont") & (tab.run == "drift") & (tab.method == "C") & (tab.cfg == "-")]
ENT = [("old default: warm start, unweighted", sd, rd_, fd, "W"), ("windowed: from scratch, last 3 rounds", sd, rd_, fd, "S"), (f"**new default: recency-weighted, h = {H}**", sn, rn, fn, NEWM)]


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
sel_ = lambda S, R, F, m, run: (S[(S.run == run) & (S.method == m) & (S.cfg == "-")] if "cfg" in S else S[(S.run == run) & (S.method == m)],
                                R[(R.run == run) & (R.method == m) & (R.cfg == "-")] if "cfg" in R else R[(R.run == run) & (R.method == m)],
                                F[(F.run == run) & (F.method == m) & (F.cfg == "-")] if "cfg" in F else F[(F.run == run) & (F.method == m)])
stats = {}
for lab, S, R, F, m in ENT:
    s_d, r_d, f_d = sel_(S, R, F, m, "drift")
    s_c, r_c, f_c = sel_(S, R, F, m, "ctrl")
    first, stays = reach(s_d)
    grid_stream = s_d[s_d.i >= PER].assign(g=lambda d: np.where(d.i <= DRIFT_I, d.grid_old, d.grid_new)).groupby("seed").g.mean().sort_index()
    L = r_d[r_d["round"] == 15].sort_values("seed")
    stats[m] = dict(lab=lab, first=first, stays=stays, grid_stream=grid_stream, L3=L[[f"L_{h}" for h in HAB3]].mean(axis=1).values, Lh={h: L[f"L_{h}"].values for h in HAB3},
                    test_drift=f_d.sort_values("seed").test_regret.values, test_ctrl=f_c.sort_values("seed").test_regret.values,
                    grid_end=s_d[s_d.i == T_END].sort_values("seed").grid_new.values, grid_ctrl_end=s_c[s_c.i == T_END].sort_values("seed").grid_old.values,
                    fs=s_d.pivot(index="seed", columns="i", values="fs_new"), fso=s_d.pivot(index="seed", columns="i", values="fs_old"), grid=s_d.pivot(index="seed", columns="i", values="grid_old"), gridn=s_d.pivot(index="seed", columns="i", values="grid_new"),
                    fs_end=s_d[s_d.i == T_END].sort_values("seed").fs_new.values)
out += ["\n---\n## 3. Appendix C: habit drift (15 rounds × 50 interactions; the Saturday-fruit habit stops after interaction 150; 5 reporting seeds)\n",
        "Same stream, seeds and metrics as `results/v2/drift_baselines.md` / `drift_retention.md`. The old default (W) and the windowed method (S) are those of `drift_targeted.py`, which reproduce `drift_trigger_series.csv` exactly. "
        "Adaptation = interactions after the drift until the regret on the 18 drifted cells (new truth) is below 0.05: *first* crossing and *sustained* (from which it stays below until the end), mean over seeds, never = 601. "
        "Full-grid regret = 1512-cell grid under the truth of the moment (old rule up to interaction 150, new rule afterwards), averaged over interactions 50–750. Test regret = 75 test scenarios, truth of the moment.\n",
        "| method | adaptation: first | seeds reaching | adaptation: sustained | retention L (3 other habits) | full-grid regret, mean over the stream | final full-grid regret (new truth) | final regret on the drifted cells | final test regret | NO-DRIFT final test regret |",
        "|---|---|---|---|---|---|---|---|---|---|"]
for m in ("W", "S", NEWM):
    s_ = stats[m]
    out.append(f"| {s_['lab']} | {cens(s_['first']).mean():.0f} | {sum(v is not None for v in s_['first'].values())}/5 | {cens(s_['stays']).mean():.0f} | {ms(s_['L3'], 2)} | {ms(s_['grid_stream'].values)} | {ms(s_['grid_end'])} | {ms(s_['fs_end'])} | {ms(s_['test_drift'])} | {ms(s_['test_ctrl'])} |")
n_, o_ = stats[NEWM], stats["W"]
out += ["\n### Retention per unchanged habit at round 15 (learning degree L)\n", "| method | bread, Sunday mornings | coffee, weekday mornings | snack, Friday evenings | mean |", "|---|---|---|---|---|"]
for m in ("W", "S", NEWM):
    out.append(f"| {stats[m]['lab']} | " + " | ".join(ms(stats[m]['Lh'][h], 2) for h in HAB3) + f" | {ms(stats[m]['L3'], 2)} |")
out += ["\n### Regret on the 18 drifted cells (new truth) after k interactions of the drifted rule\n", "| k | old default | windowed | **new default** |", "|---|---|---|---|"]
for k in (50, 100, 150, 200, 300, 450, 600):
    out.append(f"| {k} | " + " | ".join(ms(stats[m]["fs"][DRIFT_I + k].values, 3) for m in ("W", "S", NEWM)) + " |")
out += ["\n### New default against the old default (paired by seed; Δ = new − old)\n", "| quantity | new | old | Δ (mean ± SE) | seeds where new is better |", "|---|---|---|---|---|"]
for nm, a, b, bt in (("adaptation, first (lower is better)", cens(n_["first"]), cens(o_["first"]), "lower"), ("adaptation, sustained (lower)", cens(n_["stays"]), cens(o_["stays"]), "lower"),
                     ("retention L (higher)", n_["L3"], o_["L3"], "higher"), ("full-grid regret over the stream (lower)", n_["grid_stream"].values, o_["grid_stream"].values, "lower"),
                     ("final test regret, drift (lower)", n_["test_drift"], o_["test_drift"], "lower"), ("NO-DRIFT final test regret (lower)", n_["test_ctrl"], o_["test_ctrl"], "lower")):
    d = a - b
    k = int((d < 0).sum() if bt == "lower" else (d > 0).sum())
    nd = 1 if "adaptation" in nm else 3
    out.append(f"| {nm} | {a.mean():.{nd}f} | {b.mean():.{nd}f} | {d.mean():+.{nd}f} ± {d.std(ddof=1) / np.sqrt(len(d)):.{nd}f} | {k}/{len(d)} |")
flag("Appendix C / drift table", "adaptation first / sustained (interactions): TACT warm start", f"{cens(o_['first']).mean():.0f} / {cens(o_['stays']).mean():.0f}", f"{cens(n_['first']).mean():.0f} / {cens(n_['stays']).mean():.0f}")
flag("Appendix C / drift table", "retention L of the three other habits, TACT", ms(o_["L3"], 2), ms(n_["L3"], 2))
flag("Appendix C / drift table", "full-grid regret over the stream, TACT", ms(o_["grid_stream"].values), ms(n_["grid_stream"].values))
flag("Appendix C / drift figure", "curves of the TACT line", "unweighted warm start", f"recency-weighted (h = {H})")

# drift figure
def splice(old, new):
    return pd.concat([old[[c for c in old.columns if c <= DRIFT_I]], new[[c for c in new.columns if c > DRIFT_I]]], axis=1)


fig, axes = plt.subplots(2, 1, figsize=(W_, 3.9), constrained_layout=True, sharex=True)
for ax, key, ylab, letter in ((axes[0], "fs", "regret, drifted cells", "(a)"), (axes[1], "grid", "full-grid regret", "(b)")):
    for m, lab, colr, lw in (("W", "TACT, no recency weights", GREY, 1.0), (NEWM, "TACT", TACT, 1.4), ("S", "from scratch, last 3 rounds", ORANGE, 1.0)):
        a = stats[m]["fs"] if key == "fs" else None
        if key == "fs":
            c = splice(stats[m]["fso"], stats[m]["fs"])          # regret on the Saturday-fruit cells under the truth of the moment: old rule up to interaction 150, new rule afterwards
        else:
            c = splice(stats[m]["grid"], stats[m]["gridn"])
        c = c[[i for i in c.columns if i >= 50]]
        mu, sd_ = c.mean(), c.std(ddof=1)
        ax.plot(c.columns, mu, color=colr, lw=lw, label=lab)
        ax.fill_between(c.columns, mu - sd_, mu + sd_, color=colr, alpha=0.14, lw=0)
    if key == "grid":
        t = splice(tab.pivot(index="seed", columns="i", values="grid_old"), tab.pivot(index="seed", columns="i", values="grid_new"))
        t = t[[c for c in t.columns if c >= 50]]
        ax.plot(t.columns, t.mean(), color="k", lw=0.9, ls=":", label="tabular (online)")
    else:
        t = splice(tab.pivot(index="seed", columns="i", values="fs_old"), tab.pivot(index="seed", columns="i", values="fs_new"))
        t = t[[c for c in t.columns if c >= 50]]
        ax.plot(t.columns, t.mean(), color="k", lw=0.9, ls=":", label="tabular (online)")
    ax.axvline(DRIFT_I, color="#555555", ls="-.", lw=0.8)
    ax.set_ylabel(ylab)
    ax.text(-0.02, 1.0, letter, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")
axes[0].axhline(TARGET, color="k", ls="--", lw=0.7)
axes[0].legend(frameon=False, loc="upper right", handlelength=1.5, labelspacing=0.2, borderaxespad=0.1)
axes[1].set_xlabel("interactions")
axes[1].set_xlim(50, T_END)
save_fig(fig, "fig_drift")

# =================================================================== 4
pf_old, pf = rd2("partial_feedback_runs.csv"), rd3("partial_runs.csv")
SET = ["full", "reveal", "bandit", "eps"]
NAMES = {"full": "1. full feedback", "reveal": "2. asking reveals the counterfactual", "bandit": "3. strict bandit", "eps": "3b. bandit + ε-greedy (ε = 0.1)"}
gp = lambda df, st, s, r, col: df[(df.stream == st) & (df.setting == s) & (df["round"] == r)].sort_values("seed")[col].values
out += ["\n---\n## 4. Appendix E: partial feedback and the cold-start rule\n",
        "Both use the semi-online loop (retraining after every round of 50 interactions, warm start), so both were re-run with the recency weights on the pool of the labels obtained so far (age = rounds since the label was obtained; "
        "weights 0.5^(age/h)); streams, settings, seeds, ε, thresholds are those of `v2_partial_feedback.py` / `v2_cold_start.py`.\n",
        "### 4a. Partial feedback, `dpo7` (the paper's semi-online stream), round 7\n",
        "| setting | old: test regret | **new: test regret** | old: Saturday-fruit L | **new: L** | old: paired Δ regret vs setting 1 | **new: paired Δ regret vs setting 1** | new: seeds better than old | new: test U |", "|---|---|---|---|---|---|---|---|---|"]
for s in SET:
    xo, xn = gp(pf_old, "dpo7", s, 7, "test_regret"), gp(pf, "dpo7", s, 7, "test_regret")
    do_, dn = xo - gp(pf_old, "dpo7", "full", 7, "test_regret"), xn - gp(pf, "dpo7", "full", 7, "test_regret")
    out.append(f"| {NAMES[s]} | {ms(xo)} | {ms(xn)} | {ms(gp(pf_old, 'dpo7', s, 7, 'L_sat'), 2)} | {ms(gp(pf, 'dpo7', s, 7, 'L_sat'), 2)} | "
               f"{'–' if s == 'full' else f'{do_.mean():+.4f} ± {do_.std(ddof=1) / np.sqrt(5):.4f}'} | {'–' if s == 'full' else f'{dn.mean():+.4f} ± {dn.std(ddof=1) / np.sqrt(5):.4f}'} | "
               f"{paired(xn, xo)[1]} | {ms(gp(pf, 'dpo7', s, 7, 'test_U'), 3)} |")
do_ = gp(pf_old, "dpo7", "reveal", 7, "test_regret") - gp(pf_old, "dpo7", "full", 7, "test_regret")
dn = gp(pf, "dpo7", "reveal", 7, "test_regret") - gp(pf, "dpo7", "full", 7, "test_regret")
Lo, Ln = gp(pf_old, "dpo7", "reveal", 7, "L_sat"), gp(pf, "dpo7", "reveal", 7, "L_sat")
met = lambda d, L: (d.mean() <= 0.005, L.mean() >= 0.5)
out += [f"\n**Adoption criterion of the experiment (setting 2 at round 7: regret at most 0.005 above setting 1 AND Saturday-fruit L ≥ 0.5).** Old: Δ = {do_.mean():+.4f}, L = {Lo.mean():.2f} → regret condition "
        f"{'met' if met(do_, Lo)[0] else 'not met'}, L condition {'met' if met(do_, Lo)[1] else 'not met'}. **New: Δ = {dn.mean():+.4f}, L = {Ln.mean():.2f} → regret condition {'met' if met(dn, Ln)[0] else 'not met'}, "
        f"L condition {'met' if met(dn, Ln)[1] else 'not met'}.**\n"]
flag("Appendix E / partial feedback", "setting 2 vs setting 1 at round 7: Δ test regret; L (criterion: ≤ 0.005 and ≥ 0.5)", f"{do_.mean():+.4f}; {Lo.mean():.2f}", f"{dn.mean():+.4f}; {Ln.mean():.2f}")
out += ["### 4b. Partial feedback, `drift15` (15 rounds, no drift), rounds 7 and 15\n", "| setting | round | old: test regret | new: test regret | old: L | new: L |", "|---|---|---|---|---|---|"]
for r in (7, 15):
    for s in SET:
        out.append(f"| {NAMES[s]} | {r} | {ms(gp(pf_old, 'drift15', s, r, 'test_regret'))} | {ms(gp(pf, 'drift15', s, r, 'test_regret'))} | {ms(gp(pf_old, 'drift15', s, r, 'L_sat'), 2)} | {ms(gp(pf, 'drift15', s, r, 'L_sat'), 2)} |")
out += ["\n### 4c. Per round, `dpo7` (mean over 5 seeds): test regret / L, old → new\n", "| round | " + " | ".join(NAMES[s] for s in SET) + " |", "|---|" + "---|" * 4]
for r in range(1, 8):
    out.append(f"| {r} | " + " | ".join(f"{gp(pf_old, 'dpo7', s, r, 'test_regret').mean():.4f} / {gp(pf_old, 'dpo7', s, r, 'L_sat').mean():.2f} → {gp(pf, 'dpo7', s, r, 'test_regret').mean():.4f} / {gp(pf, 'dpo7', s, r, 'L_sat').mean():.2f}" for s in SET) + " |")

co, cn = rd2("cold_start_runs.csv"), rd3("cold_runs.csv")
gc = lambda df, v, r, col: df[(df.version == v) & (df["round"] == r)].sort_values("seed")[col].values
out += ["\n### 4d. Cold start (rule: ask in the first 50 interactions), semi-online loop of the paper, old → new\n",
        "Round 1 has no weights (a single round) and the rule's decisions in it do not depend on the training, so the round-1 numbers are identical; the models after round 1 differ from round 2 on.\n",
        "| round | old: interaction regret (rule) | new: interaction regret (rule) | old: test regret | new: test regret | old: U | new: U | new: paired Δ rule − no rule (interaction regret) |", "|---|---|---|---|---|---|---|---|"]
for r in range(1, 8):
    dd_ = gc(cn, "rule", r, "interaction_regret") - gc(cn, "norule", r, "interaction_regret")
    out.append(f"| {r} | {ms(gc(co, 'rule', r, 'interaction_regret'))} | {ms(gc(cn, 'rule', r, 'interaction_regret'))} | {ms(gc(co, 'rule', r, 'test_regret'))} | {ms(gc(cn, 'rule', r, 'test_regret'))} | "
               f"{ms(gc(co, 'rule', r, 'test_U'), 3)} | {ms(gc(cn, 'rule', r, 'test_U'), 3)} | {dd_.mean():+.4f} |")
r1o, r1n = gc(co, "rule", 1, "interaction_regret"), gc(cn, "rule", 1, "interaction_regret")
late = max(np.abs(gc(cn, "rule", r, "interaction_regret") - gc(cn, "norule", r, "interaction_regret")).max() for r in range(2, 8))
out += [f"\nRound-1 regret of the rule: old {ms(r1o)}, new {ms(r1n)} (always-ask on these 50 interactions {gc(cn, 'rule', 1, 'always_ask')[0]:.4f}); largest |rule − no rule| difference in rounds 2–7 (new): {late:.1e} "
        "(zero by construction: the models from the first retraining on are identical in the two versions). Criterion of the cold-start experiment: condition 2 (later rounds ≤ +0.005) is unaffected; condition 1 is the always-ask regret of the first 50 interactions, unaffected.",
        f"After 7 rounds the cold-start loop gives test regret old {ms(gc(co, 'rule', 7, 'test_regret'))} → new {ms(gc(cn, 'rule', 7, 'test_regret'))} (identical to §2c warm start by construction of the same loop)."]
flag("Appendix E / cold start", "test regret after 7 rounds in the cold-start loop (and per-round interaction regrets)", ms(gc(co, "rule", 7, "test_regret")), ms(gc(cn, "rule", 7, "test_regret")))

# =================================================================== 5
out += ["\n---\n## 5. Which experiments use the semi-online loop (checked in the code)\n",
        "| experiment | script | uses the semi-online loop (retraining every 50 interactions, warm start)? | re-run here |", "|---|---|---|---|",
        "| Sec. 5.4 semi-online stability | `replay.py`, `budget.semi_online` | yes | **yes** (§2) |",
        "| Appendix C habit drift | `drift_trigger.py` (`v2_drift.py`, `v2_drift_retention.py`) | yes (method W) | **yes** (§3; the retention supplement is part of the table) |",
        "| Appendix E partial feedback | `v2_partial_feedback.py` | yes | **yes** (§4a, b) |",
        "| Appendix E cold start | `v2_cold_start.py` | yes | **yes** (§4d) |",
        "| Ten-user experiment | `multiuser.py` | **no**: one-shot training on 350 scenarios per user (`fit_round(350, …, warm=False)`) | no (unchanged) |",
        "| Binary / graded feedback | `binary_feedback.py`, `v2_rebuttal_graded.py` | **no**: one-shot training per N (`warm=False`); the 15-round binary drift stream of `drift_trigger.py --stage bin` is a v1 experiment and not in the v2 paper | no (unchanged) |",
        "| Budget curve, main table, paraphrase robustness, context scaling, extensibility, misspecification, ablations, interaction diagnostics, rebuttal experiments 1–4 | various | no: one-shot (`warm=False`) | no (unchanged) |",
        "| Prioritised labelling (ood_priority) | `v2_ood_priority.py` | **one** warm-start update (N = 350 model + B newly labelled scenarios), not the loop; weights were not applied | no; if the new default is meant to cover single warm-start updates, this experiment's numbers would change |",
        "| Targeted reset | `drift_targeted.py` | yes (archived) | no: archived as a negative result in `results/drift_targeted.md` |",
        "| CPU timing per round (`budget.cost_report`) | `budget.py` | yes (timing only) | no: the weighted loss has the same cost per step; the epoch count with early stopping may differ slightly, so the timing numbers were not re-measured |"]

# =================================================================== 0 and 6
summ = []
summ.append(f"- **Half-life chosen on validation seeds 5–9: {H} rounds** (§1).")
summ.append(f"- **Sec. 5.4:** round-7 test MSE warm / scratch old {ms(ow.mse)} / {ms(os_.mse)} → new {ms(nw.mse)} / {ms(ns.mse)}; mean prediction drift warm / scratch old {ms(mo_w.values)} / {ms(mo_s.values)} → new {ms(mn_w.values)} / {ms(mn_s.values)}; "
            f"regret after 350 interactions (warm) old {ms(a_old.regret.values)} → new {ms(w_new.regret.values)} (one-shot {ms(one.regret.values)}).")
summ.append(f"- **Appendix C:** adaptation (first / sustained) old {cens(o_['first']).mean():.0f} / {cens(o_['stays']).mean():.0f} → new {cens(n_['first']).mean():.0f} / {cens(n_['stays']).mean():.0f} interactions; retention L {ms(o_['L3'], 2)} → {ms(n_['L3'], 2)}; "
            f"full-grid regret over the stream {ms(o_['grid_stream'].values)} → {ms(n_['grid_stream'].values)}.")
summ.append(f"- **Appendix E:** criterion on setting 2: old Δ = {do_.mean():+.4f}, L = {Lo.mean():.2f} → new Δ = {dn.mean():+.4f}, L = {Ln.mean():.2f}.")
summ.append("- **Caveats.** (1) The half-life was selected on the Appendix-C stream itself (new seeds 5–9, but the same data stream as the reporting seeds), so the Appendix-C numbers of the new default are not an independent test; Sec. 5.4 (ask-only data / `data_dpo`), "
            "Appendix E and the cold-start loop use other data and are independent of the selection. (2) The literal selection rule picked 2 rounds; half-life 1 had the lowest drifted-cell regret but a worse no-drift regret (0.0415 vs 0.0385) and is excluded by the rule; with a "
            "looser tolerance it would have been chosen. (3) In Sec. 5.4 the weights do not improve the reward-model MSE (round 7: warm 0.0126 → 0.0133, scratch 0.0130 → 0.0128, within seed noise) and slightly INCREASE the warm start's prediction drift (0.0544 → 0.0576, 0/5 seeds lower; "
            "warm still has lower drift than scratch, 5/5 seeds). (4) Appendix E: the criterion outcome (setting 2 not within 0.005 of setting 1) does not change. (5) The replay-protocol rerun without weights does not reproduce `replay_metrics.csv` exactly beyond round 1 (numerical, §2).")
out[3:3] = ["\n## 0. Summary\n"] + summ
out += ["\n---\n## 6. Numbers and figures in the paper that change\n", "| where | quantity | old | new |", "|---|---|---|---|"]
for w_, q_, o1, n1 in FLAGS:
    out.append(f"| {w_} | {q_} | {o1} | {n1} |")
out += ["| Fig. fig_semionline | both panels redrawn with recency-weighted warm start and from scratch | `figures_final/fig_semionline.pdf` | `figures_v3/fig_semionline.pdf` |",
        "| Fig. drift (Appendix C) | TACT line, new default | `figures_v2/fig_drift.pdf` | `figures_v3/fig_drift.pdf` |",
        "| Rebuttal §5b (paired tests warm vs scratch) | numbers derived from `replay_metrics.csv` | old values | recompute from `results/v3/semionline_replay.csv` if the rebuttal text is reused |",
        "| Text | statements that describe the warm start as 'unweighted' / 'continues from the previous weights on all data' (Method, §5.4, Appendix: training details, hyper-parameters) | – | must mention the recency weights 0.5^(age/h) and the half-life |",
        "| **Not changed** | main table, budget curve and crossover, accuracy bars, heatmap (one-shot from-scratch model), paraphrase robustness, context scaling, ten-user matrix, binary feedback, rebuttal experiments 1–4 | – | – |"]
with open(os.path.join(ROOT, "results", "v3_recency.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
