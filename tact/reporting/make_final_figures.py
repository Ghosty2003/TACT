"""Final figures of the paper (figures_final/*.pdf, figures_final_png/*.png), drawn from results/v2 (nothing is trained).

    python -m tact.reporting.make_final_figures

Seven single-column figures, 84 mm wide, every text >= 7 pt at final size (asserted), no titles, vector PDF (TrueType fonts embedded), PNG copies in a separate folder.
One palette (Okabe-Ito, colour-blind safe), one role per colour: TACT = blue in every figure, tabular agent = orange, the alternative it is compared with (retraining from scratch) = vermillion,
constant policies = grey / black. Every number quoted in the request is re-computed from the csv/json files and checked (results/v2/figures_final_check.md).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text

from tact.core.model import ROOT
R = os.path.join(ROOT, "results", "v2")
OUT, PNG = os.path.join(ROOT, "figures_final"), os.path.join(ROOT, "figures_final_png")
for d in (OUT, PNG):
    os.makedirs(d, exist_ok=True)
W = 84 / 25.4                                  # single column, 84 mm
TACT, TAB, ALT, GREY, DGREY = "#0072B2", "#E69F00", "#D55E00", "#9a9a9a", "#555555"
NS = [25, 50, 75, 100, 150, 250, 350]
plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.2, "ytick.major.size": 2.2, "lines.linewidth": 1.2,
                     "lines.markersize": 3.2, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300, "figure.dpi": 100, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans", "mathtext.it": "Liberation Sans:italic", "mathtext.bf": "Liberation Sans:bold", "axes.spines.top": False, "axes.spines.right": False})
CHECK = []                                     # (figure, quantity, expected, computed, source, ok)


def expect(fig, what, expected, computed, src, tol):
    ok = abs(expected - computed) <= tol
    CHECK.append((fig, what, expected, computed, src, ok))


def panel(ax, letter, x=-0.02, y=1.0):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=7, fontweight="bold", va="bottom", ha="left")


def vectorize_colorbar(cb, im):
    """matplotlib >= 3.11 embeds the colorbar gradient as a raster strip; redraw it from 256 vector cells."""
    cb.solids.remove()
    y = np.linspace(im.norm.vmin, im.norm.vmax, 257)
    cb.ax.pcolormesh([0, 1], y, ((y[:-1] + y[1:]) / 2)[:, None], cmap=im.cmap, norm=im.norm, edgecolors="face", linewidth=0.3)
    cb.ax.set_ylim(im.norm.vmin, im.norm.vmax)


ONLY = {x for x in os.environ.get("FIG_ONLY", "").split(",") if x}      # regenerate only these figures (the others are produced by other scripts since v3)


def save(fig, name):
    if ONLY and name not in ONLY:
        plt.close(fig)
        return
    assert fig._suptitle is None and all(not ax.get_title() for ax in fig.axes), f"{name}: a title is present"
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip() and t.get_visible()]
    assert min(sizes) >= 7.0, (name, min(sizes))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    fig.savefig(os.path.join(PNG, name + ".png"), dpi=300)
    w, h = fig.get_size_inches() * 25.4
    CHECK.append((name, "file: width / smallest font", "84 mm / ≥ 7 pt", f"{w:.1f} mm / {min(sizes):.1f} pt", "-", True))
    plt.close(fig)


rd = lambda f: pd.read_csv(os.path.join(R, f))
bud, hp, rp, cs = rd("budget_runs.csv"), rd("hard_paraphrase_runs.csv"), rd("replay_metrics.csv"), rd("context_scaling_runs.csv")
heat = json.load(open(os.path.join(R, "heatmap_stats.json")))
sc = pd.read_csv(os.path.join(ROOT, "data_dpo", "test_seen.csv"))
ask, act = sc[sc.cand == "ask"].score_clean.values, sc[sc.cand == "act"].score_clean.values
best = np.maximum(ask, act)
always, never = float((best - ask).mean()), float((best - act).mean())
from tact.core.model import CONFIG as cfg

# ------------------------------------------------------------------ fig_heatmap
truth, pred = np.array(heat["truth_grid"]), np.array(heat["pred_mean_grid"])
ORDER = ["happy", "calm", "neutral", "sad", "fearful", "angry"]           # valence, high to low
idx = [cfg["emotions"].index(e) for e in ORDER]
assert [cfg["valence"][e] for e in ORDER] == sorted((cfg["valence"][e] for e in ORDER), reverse=True)
truth, pred = truth[idx], pred[idx]
emos, days = ORDER, ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
ratio = lambda g: g[:, 5].mean() / np.delete(g, 5, axis=1).mean()
fig = plt.figure(figsize=(W, 1.95))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.07], wspace=0.12, left=0.155, right=0.86, top=0.90, bottom=0.2)
axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
cax = fig.add_subplot(gs[2])
for ax, g, letter in zip(axes, (truth, pred), ("(a)", "(b)")):
    im = ax.pcolormesh(np.arange(-0.5, g.shape[1] + 0.5), np.arange(-0.5, g.shape[0] + 0.5), g, vmin=0, vmax=1, cmap="viridis", edgecolors="face", linewidth=0.3)   # vector cells (imshow embeds a raster)
    ax.set_xlim(-0.5, g.shape[1] - 0.5)
    ax.set_ylim(g.shape[0] - 0.5, -0.5)
    ax.set_xticks(range(7), days)
    ax.tick_params(length=0)
    for i in range(len(emos)):
        for d in range(7):
            ax.text(d, i, f"{g[i, d]:.2f}".lstrip("0"), ha="center", va="center", color="white" if g[i, d] < 0.55 else "black", fontsize=7)
    ax.add_patch(Rectangle((4.5, -0.5), 1, len(emos), fill=False, ec=ALT, lw=1.0))
    ax.set_xlabel(f"Sat / other = {ratio(g):.2f}")
    for s in ax.spines.values():
        s.set_visible(False)
    panel(ax, letter, x=0.0, y=1.01)
axes[0].set_yticks(range(len(emos)), emos)
axes[1].set_yticks([])
axes[0].set_ylabel("emotion")
cbh = fig.colorbar(im, cax=cax)
vectorize_colorbar(cbh, im)
cbh.set_label("asking utility", fontsize=7)
cax.tick_params(labelsize=7)
expect("fig_heatmap", "Saturday/other ratio, truth (fruit, afternoon)", 0.15, ratio(truth), "heatmap_stats.json (truth_grid)", 0.005)
expect("fig_heatmap", "Saturday/other ratio, mean prediction (~0.30)", 0.30, heat["mean_of_seeds"]["sat_over_other"], "heatmap_stats.json (mean_of_seeds)", 0.01)
expect("fig_heatmap", "ratio computed from the plotted prediction grid", heat["mean_of_seeds"]["sat_over_other"], ratio(pred), "heatmap_stats.json (pred_mean_grid)", 0.002)
save(fig, "fig_heatmap")

# ------------------------------------------------------------------ fig_budget
bs = bud[bud.test == "seen"]
a = bs[bs.method == "A"].groupby("n").regret.agg(["mean", "std"])
c = bs[bs.method == "C"].groupby("n").regret.mean()
cross = next(n for i, n in enumerate(NS) if all(a["mean"][m] <= c[m] for m in NS[i:]))
fig, ax = plt.subplots(figsize=(W, 2.35), constrained_layout=True)
ax.plot(NS, a["mean"], "-o", color=TACT, label="TACT (ours)", zorder=3)
ax.fill_between(NS, a["mean"] - a["std"], a["mean"] + a["std"], color=TACT, alpha=0.18, lw=0)
ax.plot(NS, [c[n] for n in NS], "-s", color=TAB, label="tabular")
ax.axhline(always, color=GREY, ls="--", lw=1.0, label="always-ask")
ax.axhline(never, color="k", ls=":", lw=1.0, label="never-ask")
ax.axvline(cross, color=DGREY, ls="-.", lw=0.8)
ax.annotate(f"crossover\n$N={cross}$", xy=(cross, 0.075), xytext=(cross * 1.08, 0.012), fontsize=7, color=DGREY, arrowprops=dict(arrowstyle="-", color=DGREY, lw=0.6))
ax.set_xscale("log")
ax.set_xticks(NS, [str(n) for n in NS])
ax.minorticks_off()
ax.set_xlabel("training scenarios $N$")
ax.set_ylabel("mean regret")
ax.set_ylim(0, 0.28)
ax.legend(loc="upper right", ncol=2, frameon=False, columnspacing=0.9, handlelength=1.5, borderaxespad=0.2)
expect("fig_budget", "first N from which TACT <= tabular (crossover)", 50, cross, "budget_runs.csv (method A vs C, test seen)", 0)
expect("fig_budget", "TACT regret at N = 350", 0.036, a["mean"][350], "budget_runs.csv", 5e-4)
expect("fig_budget", "tabular regret at N = 350", 0.079, c[350], "budget_runs.csv", 5e-4)
expect("fig_budget", "always-ask reference line", 0.158, always, "data_dpo/test_seen.csv", 1e-3)
expect("fig_budget", "never-ask reference line", 0.157, never, "data_dpo/test_seen.csv", 1e-3)
save(fig, "fig_budget")

# ------------------------------------------------------------------ fig_accuracy
oracle_ask = ask >= act
bA = bs[(bs.method == "A") & (bs.n == 350)].sort_values("seed").acc.values
bC = bs[(bs.method == "C") & (bs.n == 350)].sort_values("seed").acc.values
means = [float(oracle_ask.mean()), float((~oracle_ask).mean()), bC.mean(), bA.mean(), 1.0]
labels = ["always-\nask", "never-\nask", "tabular", "TACT\n(ours)", "oracle"]
fig, ax = plt.subplots(figsize=(W, 2.2), constrained_layout=True)
ax.bar(range(5), means, 0.66, color=[GREY, DGREY, TAB, TACT, "white"], edgecolor=["none"] * 4 + ["k"], linewidth=0.8)
ax.errorbar([3], [bA.mean()], yerr=[bA.std(ddof=1)], color="k", capsize=2.5, lw=0.8, capthick=0.8, fmt="none")
for i, mu in enumerate(means):
    top = mu + (bA.std(ddof=1) if i == 3 else 0)
    ax.text(i, top + 0.02, f"{mu:.3f}", ha="center", va="bottom", fontsize=7, fontweight="bold" if i == 3 else "normal")
ax.set_xticks(range(5), labels)
ax.set_ylabel("decision agreement\nwith the oracle")
ax.set_ylim(0, 1.18)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
for nm, exp_, got in (("always-ask", 0.64, means[0]), ("never-ask", 0.36, means[1]), ("tabular", 0.613, means[2]), ("TACT", 0.797, means[3]), ("oracle", 1.0, means[4])):
    expect("fig_accuracy", f"bar: {nm}", exp_, got, "budget_runs.csv (N = 350, test seen) / data_dpo/test_seen.csv", 6e-4)
save(fig, "fig_accuracy")

# ------------------------------------------------------------------ fig_paraphrase (N = 150 and N = 350, TACT: seen / unseen / hard)
vals = {}
for n in (150, 350):
    h = hp[(hp.method == "A") & (hp.n == n)]
    s = h[h.test == "seen"].sort_values("seed").regret.values
    u = h[h.test == "unseen"].sort_values("seed").regret.values
    d = np.array([h[(h.test.isin(["hard1", "hard2"])) & (h.seed == k)].regret.mean() for k in range(5)])
    vals[n] = (s, u, d)
fig, ax = plt.subplots(figsize=(W, 2.3), constrained_layout=True)
bw = 0.25
kinds = [("seen phrasing", TACT, None), ("unseen phrasing", "#7fb8dc", None), ("hard paraphrases", "white", "////")]
for gi, n in enumerate((150, 350)):
    for k, (lab, col, hatch) in enumerate(kinds):
        v = vals[n][k]
        x = gi + (k - 1) * bw * 1.15
        ax.bar(x, v.mean(), bw, yerr=v.std(ddof=1), capsize=2, color=col, edgecolor=TACT, hatch=hatch, linewidth=0.7, error_kw={"lw": 0.7},
               label=lab if gi == 0 else None)
        ax.text(x, v.mean() + v.std(ddof=1) + 0.003, f"{v.mean():.3f}", ha="center", va="bottom", fontsize=7)
ax.set_xticks([0, 1], ["$N=150$", "$N=350$"])
ax.set_ylabel("mean regret")
ax.set_ylim(0, 0.135)
ax.legend(frameon=False, loc="upper right", handlelength=1.4, borderaxespad=0.1, labelspacing=0.25)
for n, exp3 in ((350, (0.036, 0.034, 0.035)), (150, (0.053, 0.068, 0.081))):
    for k, (lab, _, _) in enumerate(kinds):
        expect("fig_paraphrase", f"N = {n}, {lab}", exp3[k], vals[n][k].mean(), "hard_paraphrase_runs.csv (method A)", 6e-4)
save(fig, "fig_paraphrase")

# ------------------------------------------------------------------ fig_semionline
fig, axes = plt.subplots(1, 2, figsize=(W, 1.95), constrained_layout=True)
for ax, col, ylab, first, letter in ((axes[0], "mse", "test MSE", 1, "(a)"), (axes[1], "drift", "prediction drift", 2, "(b)")):
    for strat, lab, colr in (("warm", "warm start", TACT), ("scratch", "from scratch", ALT)):
        g = rp[(rp.strategy == strat) & (rp["round"] >= first)].groupby("n_train")[col].agg(["mean", "std"])
        ax.plot(g.index, g["mean"], "-o", color=colr, label=lab)
        ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=colr, alpha=0.18, lw=0)
    ax.set_xticks([100, 200, 300])
    ax.set_xlabel("training samples")
    ax.set_ylabel(ylab)
    panel(ax, letter, x=-0.02)
axes[0].legend(frameon=False, loc="upper right", handlelength=1.3, borderaxespad=0.1)
last = rp[rp["round"] == 7]
per = rp[rp["round"] >= 2].groupby(["strategy", "seed"]).drift.mean()
expect("fig_semionline", "test MSE at 350 samples, warm start", 0.0126, last[last.strategy == "warm"].mse.mean(), "replay_metrics.csv (round 7)", 6e-5)
expect("fig_semionline", "test MSE at 350 samples, from scratch", 0.0130, last[last.strategy == "scratch"].mse.mean(), "replay_metrics.csv (round 7)", 6e-5)
expect("fig_semionline", "prediction drift, mean over rounds 2-7, warm start", 0.054, per.loc["warm"].mean(), "replay_metrics.csv", 6e-4)
expect("fig_semionline", "prediction drift, mean over rounds 2-7, from scratch (asked: 0.076)", 0.076, per.loc["scratch"].mean(), "replay_metrics.csv", 6e-4)
save(fig, "fig_semionline")

# ------------------------------------------------------------------ multiuser_matrix
MA = pd.read_csv(os.path.join(R, "multiuser_matrix_A.csv"), index_col=0).values      # rows = model of user i, columns = truth of user j (normalized grid regret)
MN = MA / MA.mean(axis=0, keepdims=True)                                                # each column divided by its own mean: removes how hard a user's truth is for every model
fig, ax = plt.subplots(figsize=(W, 3.2), constrained_layout=True)
im = ax.pcolormesh(np.arange(-0.5, 10.5), np.arange(-0.5, 10.5), MN, cmap="viridis_r", vmin=0, vmax=MN.max(), edgecolors="face", linewidth=0.3)     # vector cells
ax.set_xlim(-0.5, 9.5)
ax.set_ylim(9.5, -0.5)
ax.set_aspect("equal")
users = [str(u) for u in range(1, 11)]
ax.set_xticks(range(10), [u + "*" if u in ("9", "10") else u for u in users])
ax.set_yticks(range(10), [u + "*" if u in ("9", "10") else u for u in users])
for i in range(10):
    for j in range(10):
        ax.text(j, i, f"{MN[i, j]:.2f}".lstrip("0"), ha="center", va="center", fontsize=7, color="white" if MN[i, j] > 0.55 * MN.max() else "black", fontweight="bold" if i == j else "normal")
    ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec=ALT, lw=1.0))
for t in list(ax.get_xticklabels())[8:] + list(ax.get_yticklabels())[8:]:
    t.set_color(ALT)
ax.set_xlabel("user whose truth is used  (* = non-monotone user)")
ax.set_ylabel("user whose model is evaluated")
ax.tick_params(length=0)
for s_ in ax.spines.values():
    s_.set_visible(False)
cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02, aspect=26)
vectorize_colorbar(cb, im)
cb.set_label("regret / column mean", fontsize=7)
cb.ax.tick_params(labelsize=7)
off = MA[~np.eye(10, dtype=bool)]
expect("multiuser_matrix", "mean of the diagonal, raw matrix (own truth)", 0.58, np.diag(MA).mean(), "multiuser_matrix_A.csv", 5e-3)
expect("multiuser_matrix", "mean of the off-diagonal, raw matrix", 0.98, off.mean(), "multiuser_matrix_A.csv", 5e-3)
expect("multiuser_matrix", "every plotted column has mean 1", 1.0, float(MN.mean(axis=0).min()), "multiuser_matrix_A.csv (column-normalized)", 1e-9)
expect("multiuser_matrix", "every plotted column has mean 1 (max)", 1.0, float(MN.mean(axis=0).max()), "multiuser_matrix_A.csv (column-normalized)", 1e-9)
expect("multiuser_matrix", "plotted diagonal is the column minimum for how many of 10 users (info)", 10, float(sum(MN[j, j] == MN[:, j].min() for j in range(10))), "multiuser_matrix_A.csv (column-normalized)", 10)
save(fig, "multiuser_matrix")

# ------------------------------------------------------------------ context_scaling
LV = [("3", 3, ()), ("4", 4, ("location",)), ("5", 5, ("location", "companion")), ("6", 6, ("location", "companion", "activity")), ("7", 7, ("location", "companion", "activity", "noise_level"))]
M = cs[cs.level.isin([l[0] for l in LV])]
S7 = cs[cs.level == "7 (n=1000 data)"]
STYLE = {"activity": dict(ls="-", marker="o", lw=1.8, ms=3.6, label="activity × topic (× 0.5)"), "location": dict(ls="--", marker="s", lw=0.9, ms=3.0, label="location (× 0.3)"),
         "companion": dict(ls=":", marker="^", lw=1.1, ms=3.2, label="companion (× 0.5)"), "noise_level": dict(ls="-.", marker="D", lw=0.9, ms=2.8, label="noise (× 0.7)")}
fig, axes = plt.subplots(2, 1, figsize=(W, 4.5), constrained_layout=True)
ax = axes[0]
for v, st in STYLE.items():
    pts = [(nv, M[M.level == name][f"{v}_L"]) for name, nv, ex in LV if v in ex]
    ax.errorbar([p[0] for p in pts], [p[1].mean() for p in pts], [p[1].std(ddof=1) for p in pts], color=TACT, capsize=1.8, elinewidth=0.6, **st)
    for nn_, mk, dx in ((350, "s", 0.12), (700, "D", 0.24)):
        ax.scatter([7 + dx], [S7[S7.n == nn_][f"{v}_L"].mean()], marker=mk, facecolors="none", edgecolors=TACT, s=11, linewidths=0.7, zorder=4)
ax.axhline(1.0, color=GREY, ls=":", lw=0.8)
ax.axhline(0.5, color="k", ls="--", lw=0.8)
ax.text(3.9, 0.52, "limit $L = 0.5$", fontsize=7, va="bottom")
ax.set_xticks([3, 4, 5, 6, 7])
ax.set_xlim(2.8, 7.55)
ax.set_ylim(-0.25, 1.2)
ax.set_xlabel("number of context variables")
ax.set_ylabel("rule learning degree $L$")
h = [Line2D([], [], color=TACT, **{k: v for k, v in st.items()}) for st in STYLE.values()]
h += [Line2D([], [], color=TACT, marker="s", mfc="none", ls="", ms=4, label="7 var., $N=350$, larger data"), Line2D([], [], color=TACT, marker="D", mfc="none", ls="", ms=4, label="7 var., $N=700$, larger data")]
ax.legend(handles=h, frameon=False, loc="lower left", fontsize=7, handlelength=1.8, labelspacing=0.2, borderaxespad=0.1, ncol=1)
panel(ax, "(a)", x=-0.02)
ax = axes[1]
for m_, lab, colr, mk in (("A", "TACT (ours)", TACT, "o"), ("C", "tabular", TAB, "s")):
    pts = [(nv, M[M.level == name][f"grid_nr_{m_}"]) for name, nv, _ in LV]
    ax.errorbar([p[0] for p in pts], [p[1].mean() for p in pts], [p[1].std(ddof=1) for p in pts], color=colr, marker=mk, capsize=1.8, elinewidth=0.6, label=lab)
    for nn_, mk2, dx in ((350, "s", 0.12), (700, "D", 0.24)):
        ax.scatter([7 + dx], [S7[S7.n == nn_][f"grid_nr_{m_}"].mean()], marker=mk2, facecolors="none", edgecolors=colr, s=11, linewidths=0.7, zorder=4)
ax.axhline(1.0, color="k", ls="--", lw=0.8)
ax.text(7.5, 1.02, "best constant policy", fontsize=7, va="bottom", ha="right")
ax.set_xticks([3, 4, 5, 6, 7])
ax.set_xlim(2.8, 7.55)
ax.set_ylim(0.1, 1.3)
ax.set_xlabel("number of context variables")
ax.set_ylabel("normalized grid regret")
ax.legend(frameon=False, loc="lower right", ncol=1, handlelength=1.4, borderaxespad=0.1)
panel(ax, "(b)", x=-0.02)
a6 = M[M.level == "6"].activity_L.mean()
a7 = M[M.level == "7"].activity_L.mean()
expect("context_scaling", "L activity × topic, 6 variables (low at default sampling)", 0.085, a6, "context_scaling_runs.csv", 2e-3)
expect("context_scaling", "L activity × topic, 7 variables", 0.007, a7, "context_scaling_runs.csv", 2e-3)
expect("context_scaling", "normalized regret TACT, 7 variables", 0.651, M[M.level == "7"].grid_nr_A.mean(), "context_scaling_runs.csv", 1e-3)
expect("context_scaling", "normalized regret tabular, 7 variables", 0.785, M[M.level == "7"].grid_nr_C.mean(), "context_scaling_runs.csv", 1e-3)
save(fig, "context_scaling")

# ------------------------------------------------------------------ MISSING.txt and the check table
rows = ["# Figure check (`make_final_figures.py`)\n", "Every number quoted for the final figures, re-computed from the csv/json files of `results/v2`.\n",
        "| figure | quantity | expected | computed | source | ok |", "|---|---|---|---|---|---|"]
for fg, what, e, cmp_, src, ok in CHECK:
    fmt = lambda x: x if isinstance(x, str) else (f"{x:.4f}" if abs(x) < 1 else f"{x:.3f}")
    rows.append(f"| {fg} | {what} | {fmt(e)} | {fmt(cmp_)} | {src} | {'ok' if ok else '**MISMATCH**'} |")
with open(os.path.join(R, "figures_final_check.md"), "w") as f:
    f.write("\n".join(rows) + "\n")
print("\n".join(rows))
bad = [r for r in CHECK if not r[5]]
print(f"\n{len(CHECK) - len(bad)} of {len(CHECK)} checks ok" + (f"; MISMATCH: {[(r[0], r[1]) for r in bad]}" if bad else ""))
