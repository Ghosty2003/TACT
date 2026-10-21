"""Figures for the paper (figures/*.pdf), drawn from the saved csv/json files; nothing is trained.

    python -m tact.reporting.make_paper_figures

Springer iicol, single column: every figure is drawn at its final width of 8.6 cm (3.386 in), so nothing is rescaled in LaTeX and the
fonts below are the printed sizes (>= 7 pt). Vector PDF (TrueType fonts embedded as Type 42), matplotlib default look and colours with one
fixed role per colour: C0 = the proposed method (TACT: reward + SFT + DPO), C1 = the alternative it is compared with (direct reward
maximisation / from-scratch retraining), C2 = the tabular agent, grey/black = constant baselines. No figure carries a title (captions are
given in LaTeX); panels of multi-panel figures only carry the letters (a), (b). PNG previews go to figures_preview/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.text import Text

from tact.core.model import CONFIG, ROOT

R, OUT, PREV = os.path.join(ROOT, "results"), os.path.join(ROOT, "figures"), os.path.join(ROOT, "figures_preview")
for d in (OUT, PREV):
    os.makedirs(d, exist_ok=True)
W = 8.6 / 2.54      # single column width in inches
NS = [25, 50, 75, 100, 150, 250, 350]
NAVY = "#1f3d7a"    # the proposed method where it has to stand out (accuracy bars)
GREY = "#9a9a9a"
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5,
                     "ytick.major.size": 2.5, "lines.linewidth": 1.3, "lines.markersize": 3.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "figure.dpi": 100, "savefig.dpi": 300})
REPORT = []


def panel(ax, letter):
    ax.text(0.0, 1.01, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="left")


def save(fig, name, min_pt=7.0, png_in_figures=False):
    """Checks (no title text, smallest font >= min_pt), then writes figures/<name>.pdf and figures_preview/<name>.png
    (also figures/<name>.png if png_in_figures)."""
    assert fig._suptitle is None and all(not ax.get_title() for ax in fig.axes), f"{name}: a title is present"
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    assert min(sizes) >= min_pt, (name, min(sizes))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(PREV, name + ".png"), dpi=300)
    if png_in_figures:
        fig.savefig(os.path.join(OUT, name + ".png"), dpi=300)
    REPORT.append((name, fig.get_size_inches(), min(sizes)))
    plt.close(fig)


# ------------------------------------------------------------------ 1. heatmap (fruit, afternoon; day x emotion)
heat = json.load(open(os.path.join(R, "heatmap_stats.json")))
truth, pred = np.array(heat["truth_grid"]), np.array(heat["pred_mean_grid"])
days, emos = CONFIG["days"], CONFIG["emotions"]
fig, axes = plt.subplots(2, 1, figsize=(W, 4.0), sharex=True, constrained_layout=True)
for ax, g, letter in ((axes[0], truth, "(a)"), (axes[1], pred, "(b)")):
    im = ax.imshow(g, vmin=0, vmax=1, cmap="viridis", aspect="auto")      # same drawing as generate_dataset.py
    ax.set_yticks(range(len(emos)), emos)
    ax.set_xticks(range(7), days)
    for i in range(len(emos)):
        for d in range(7):
            ax.text(d, i, f"{g[i, d]:.2f}", ha="center", va="center", color="white" if g[i, d] < 0.55 else "black", fontsize=7)
    panel(ax, letter)
axes[1].set_xlabel("day of week")
axes[0].set_ylabel("emotion")
axes[1].set_ylabel("emotion")
fig.colorbar(im, ax=axes, label="preference score $r^*$", shrink=0.92, pad=0.02, aspect=28)
save(fig, "fig_heatmap")

# ------------------------------------------------------------------ 2. budget curve (test_seen)
bud = pd.read_csv(os.path.join(R, "budget_runs.csv"))
bs = bud[bud.test == "seen"]
a = bs[bs.method == "A"].groupby("n").regret.agg(["mean", "std"])
c = bs[bs.method == "C"].groupby("n").regret.mean()
sc = pd.read_csv(os.path.join(ROOT, "data_dpo", "test_seen.csv"))
ask, act = sc[sc.cand == "ask"].score_clean.values, sc[sc.cand == "act"].score_clean.values
best = np.maximum(ask, act)
always, never = float((best - ask).mean()), float((best - act).mean())
cross = next(n for i, n in enumerate(NS) if all(a["mean"][m] <= c[m] for m in NS[i:]))
fig, ax = plt.subplots(figsize=(W, 2.7), constrained_layout=True)
ax.plot(NS, a["mean"], "-o", color="C0", label="TACT (ours)")
ax.fill_between(NS, a["mean"] - a["std"], a["mean"] + a["std"], color="C0", alpha=0.18, lw=0)
ax.plot(NS, [c[n] for n in NS], "-s", color="C2", label="tabular C")
ax.axhline(always, color=GREY, ls="--", lw=1.0, label="always-ask")
ax.axhline(never, color="k", ls=":", lw=1.0, label="never-ask")
ax.vlines(cross, 0, 0.15, color="C3", ls="-.", lw=0.9)
ax.annotate(f"crossover\n$N={cross}$", xy=(cross, 0.035), xytext=(cross * 1.1, 0.012), color="C3", fontsize=7.5,
            arrowprops=dict(arrowstyle="->", color="C3", lw=0.7))
ax.set_xscale("log")
ax.set_xticks(NS, [str(n) for n in NS])
ax.minorticks_off()
ax.set_xlabel("training scenarios $N$")
ax.set_ylabel("mean regret")
ax.set_ylim(0, 0.25)
ax.legend(loc="upper center", ncol=2, frameon=False, columnspacing=1.0, handlelength=1.6)
save(fig, "fig_budget")

# ------------------------------------------------------------------ 3. DPO policy vs reward argmax: seen / hard
hd = pd.read_csv(os.path.join(R, "hard_paraphrase_runs.csv"))
hd = hd[hd.n == 350]
vals = {}
for m in ("A", "B"):
    seen = hd[(hd.method == m) & (hd.test == "seen")].sort_values("seed").regret.values
    hard = np.array([hd[(hd.method == m) & (hd.test.isin(["hard1", "hard2"])) & (hd.seed == s)].regret.mean() for s in range(5)])
    vals[m] = (seen, hard)
fig, ax = plt.subplots(figsize=(W, 2.6), constrained_layout=True)
w = 0.36
x = np.arange(2)
for k, (m, lab, colr) in enumerate((("A", "DPO policy", "C0"), ("B", "direct reward maximisation", "C1"))):
    means = [vals[m][0].mean(), vals[m][1].mean()]
    sds = [vals[m][0].std(ddof=1), vals[m][1].std(ddof=1)]
    ax.bar(x + (k - 0.5) * w, means, w, yerr=sds, capsize=2.5, color=colr, label=lab, error_kw={"lw": 0.8})
    for xi, mu, sd in zip(x + (k - 0.5) * w, means, sds):
        ax.text(xi, mu + sd + 0.003, f"{mu:.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks(x, ["seen\nphrasing", "hard\nparaphrases"])
ax.set_ylabel("mean regret")
ax.set_ylim(0, 0.15)
ax.legend(frameon=False, loc="upper left")
save(fig, "fig_dpo_robust")

# ------------------------------------------------------------------ 4. semi-online retraining (warm-start vs from-scratch)
rp = pd.read_csv(os.path.join(R, "replay_metrics.csv"))
fig, axes = plt.subplots(2, 1, figsize=(W, 3.7), sharex=True, constrained_layout=True)
for ax, col, ylab, first, letter in ((axes[0], "mse", "test MSE", 1, "(a)"), (axes[1], "drift", "prediction drift", 2, "(b)")):
    for strat, lab, colr in (("warm", "warm-start", "C0"), ("scratch", "from-scratch", "C1")):
        g = rp[(rp.strategy == strat) & (rp["round"] >= first)].groupby("n_train")[col].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], "-o", color=colr, label=lab)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colr, alpha=0.18, lw=0)
    ax.set_ylabel(ylab)
    ax.set_xticks(range(50, 351, 50))
    panel(ax, letter)
axes[1].set_xlabel("training samples")
axes[0].legend(frameon=False, loc="upper right")
save(fig, "fig_semionline")

# ------------------------------------------------------------------ 5. context scaling (rule learning degree, regret vs number of variables)
cs = pd.read_csv(os.path.join(R, "context_scaling_runs.csv"))
LV = [("3", 3, ()), ("4", 4, ("location",)), ("5", 5, ("location", "companion")), ("6", 6, ("location", "companion", "activity")),
      ("7", 7, ("location", "companion", "activity", "noise_level"))]
M = cs[cs.level.isin([l[0] for l in LV])]
S7 = cs[cs.level == "7 (n=1000 data)"]
RULE_COL = {"location": "C3", "companion": "C4", "activity": "C5", "noise_level": "C6"}
RULE_LAB = {"location": "location ($\\times$0.3)", "companion": "companion ($\\times$0.5)", "activity": "activity $\\times$ topic ($\\times$0.5)",
            "noise_level": "noise ($\\times$0.7)"}
fig, axes = plt.subplots(2, 1, figsize=(W, 4.6), constrained_layout=True)
ax = axes[0]
for v in ("location", "companion", "activity", "noise_level"):
    pts = [(nv, M[M.level == name][f"{v}_L"]) for name, nv, ex in LV if v in ex]
    ax.errorbar([p[0] for p in pts], [p[1].mean() for p in pts], [p[1].std(ddof=1) for p in pts], color=RULE_COL[v], marker="o", capsize=2,
                elinewidth=0.7, label=RULE_LAB[v])
    for nn_, mk, dx in ((350, "s", 0.12), (700, "D", 0.24)):
        xs = S7[S7.n == nn_][f"{v}_L"]
        ax.scatter([7 + dx], [xs.mean()], marker=mk, facecolors="none", edgecolors=RULE_COL[v], s=14, linewidths=0.8, zorder=4)
ax.axhline(1.0, color=GREY, ls=":", lw=0.9)
ax.axhline(0.5, color="k", ls="--", lw=0.9)
ax.text(3.82, 0.51, "limit: $L<0.5$", fontsize=7, va="bottom")
ax.set_xticks([4, 5, 6, 7])
ax.set_xlim(3.8, 7.5)
ax.set_xlabel("number of context variables")
ax.set_ylabel("rule learning degree $L$")
h = [Line2D([], [], color=RULE_COL[v], marker="o", ms=3.5, label=RULE_LAB[v]) for v in RULE_COL]
h += [Line2D([], [], color=GREY, marker="s", mfc="none", ls="", ms=4, label="7 var., $N=350$"),
      Line2D([], [], color=GREY, marker="D", mfc="none", ls="", ms=4, label="7 var., $N=700$")]
ax.legend(handles=h, frameon=False, loc="lower left", fontsize=7, handlelength=1.4, labelspacing=0.3)
panel(ax, "(a)")
ax = axes[1]
for m, lab, colr in (("A", "TACT (ours)", "C0"), ("B", "reward argmax", "C1"), ("C", "tabular C", "C2")):
    pts = [(nv, M[M.level == name][f"grid_nr_{m}"]) for name, nv, _ in LV]
    ax.errorbar([p[0] for p in pts], [p[1].mean() for p in pts], [p[1].std(ddof=1) for p in pts], color=colr, marker="o", capsize=2,
                elinewidth=0.7, label=lab)
    for nn_, mk, dx in ((350, "s", 0.12), (700, "D", 0.24)):
        xs = S7[S7.n == nn_][f"grid_nr_{m}"]
        ax.scatter([7 + dx], [xs.mean()], marker=mk, facecolors="none", edgecolors=colr, s=14, linewidths=0.8, zorder=4)
ax.axhline(1.0, color="k", ls="--", lw=0.9)
ax.text(7.45, 1.015, "best constant policy", fontsize=7, va="bottom", ha="right")
ax.set_xticks([3, 4, 5, 6, 7])
ax.set_xlim(2.8, 7.5)
ax.set_ylim(0.2, 1.4)
ax.set_xlabel("number of context variables")
ax.set_ylabel("normalised grid regret")
ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=7, columnspacing=0.9, handlelength=1.3, bbox_to_anchor=(0.5, 0.93))
panel(ax, "(b)")
save(fig, "context_scaling")

# ------------------------------------------------------------------ 6. decision agreement with the oracle, N = 350 (test_seen)
sc_ask = sc[sc.cand == "ask"]
oracle_ask = ask >= act
acc_always, acc_never = float(oracle_ask.mean()), float((~oracle_ask).mean())
bA = bud[(bud.method == "A") & (bud.test == "seen") & (bud.n == 350)].sort_values("seed").acc.values
bC = bud[(bud.method == "C") & (bud.test == "seen") & (bud.n == 350)].sort_values("seed").acc.values
given = {"always-ask": 0.64, "never-ask": 0.36, "tabular C": 0.613, "TACT": 0.789}
calc = {"always-ask": acc_always, "never-ask": acc_never, "tabular C": bC.mean(), "TACT": bA.mean()}
for k in given:
    assert abs(round(calc[k], 3) - given[k]) < 6e-4, (k, calc[k], given[k])
assert abs(round(bA.std(ddof=1), 3) - 0.024) < 6e-4, bA.std(ddof=1)
print("fig_accuracy data check: " + "; ".join(f"{k} {calc[k]:.4f} (given {given[k]})" for k in given) + f"; TACT std {bA.std(ddof=1):.4f}; oracle 1.0")
labels = ["always-\nask", "never-\nask", "tabular\nC", "TACT\n(ours)", "oracle"]
means = [acc_always, acc_never, bC.mean(), bA.mean(), 1.0]
cols = [GREY, GREY, "C2", NAVY, "white"]
fig, ax = plt.subplots(figsize=(W, 2.5), constrained_layout=True)
bars = ax.bar(range(5), means, 0.66, color=cols, edgecolor=["none", "none", "none", "none", "k"], linewidth=0.8)
ax.errorbar([3], [bA.mean()], yerr=[bA.std(ddof=1)], color="k", capsize=3, lw=0.9, capthick=0.9, fmt="none")   # 5 seeds, TACT only
for i, mu in enumerate(means):
    top = mu + (bA.std(ddof=1) if i == 3 else 0)
    ax.text(i, top + 0.015, f"{mu:.3f}" if i in (2, 3) else f"{mu:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold" if i == 3 else "normal")
ax.set_xticks(range(5), labels)
ax.set_ylabel("decision agreement\nwith the oracle")
ax.set_ylim(0, 1.15)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
save(fig, "fig_accuracy", png_in_figures=True)

# ------------------------------------------------------------------ 7. multi-user cross matrix (policy A only)
MA = pd.read_csv(os.path.join(R, "multiuser_matrix_A.csv"), index_col=0).values          # rows: model of user i, columns: truth of user j
fig, ax = plt.subplots(figsize=(W, 3.45), constrained_layout=True)
im = ax.imshow(MA, cmap="viridis_r", vmin=0, vmax=MA.max())
ax.set_xticks(range(10), [str(u) for u in range(1, 11)])
ax.set_yticks(range(10), [str(u) for u in range(1, 11)])
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{MA[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if MA[i, j] > 0.55 * MA.max() else "black",
                fontweight="bold" if i == j else "normal")
    ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.0))
for u in (9, 10):                                   # the two non-monotone users
    ax.get_xticklabels()[u - 1].set_color("#b22222")
    ax.get_yticklabels()[u - 1].set_color("#b22222")
ax.set_xlabel("user whose truth is used")
ax.set_ylabel("user whose model is evaluated")
ax.tick_params(length=0)
fig.colorbar(im, ax=ax, label="normalised grid regret", shrink=0.92, pad=0.02, aspect=26)
save(fig, "multiuser_matrix", min_pt=6.0)

for name, size, fs in REPORT:
    print(f"{name}: {size[0] * 2.54:.2f} x {size[1] * 2.54:.2f} cm, smallest font {fs:.1f} pt")
