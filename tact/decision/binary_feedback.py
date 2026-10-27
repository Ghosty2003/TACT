"""Experiment A: 0/1 (accept / reject) feedback instead of continuous scores (results/binary_feedback.md, .png).

    python -m tact.core.generate_dataset --n 500  --seed 42 --with-act --label binary --outdir data_bin
    python -m tact.core.generate_dataset --n 1000 --seed 42 --with-act --label binary --outdir data_bin_n1000
    python -m tact.core.generate_dataset --n 2000 --seed 42 --with-act --label binary --outdir data_bin_n2000
    python -m tact.core.generate_dataset --n 1000 --seed 42 --with-act --outdir data_cont_n1000     # continuous reference, N = 700
    python -m tact.core.generate_dataset --n 2000 --seed 42 --with-act --outdir data_cont_n2000     # continuous reference, N = 1400
    python -m tact.decision.binary_feedback                # train + evaluate (writes results/binary_feedback_runs.csv), then the report
    python -m tact.decision.binary_feedback --report-only  # report only

Truth, scenarios, splits, phrasings and seeds are those of the main experiment; only the observed label changes:
y ~ Bernoulli(r*), r* = the noise-free ask value n*t (ask) or 1-n (act), no thresholding. Training differs from budget.py in
exactly three places: the reward-model loss is binary cross-entropy (the sigmoid output is the acceptance probability), the SFT
target of a scenario whose two observed labels are equal (a tie: both 1 or both 0) is 0.5, and tabular Q is the acceptance
frequency per cell. DPO is unchanged (preference pairs labelled by the reward model). Evaluation is the code of the main
experiment (regret on the continuous noise-free truth), plus the same quantities on the full 1512-cell grid, which does not
depend on the size of the test split. Hyper-parameters: budget.HP, untouched.
"""
import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, ttest_ind

from tact.decision import budget as B
from tact.core import plots
from tact.drift.habit_drift import collapse_ratio, truth_grid
from tact.core.hard_paraphrase_texts import HARD
from tact.core.model import RESULTS_DIR, ROOT, V2, FrozenBert, get_device

DIRS = {("cont", 500): "data_dpo", ("cont", 1000): "data_cont_n1000", ("cont", 2000): "data_cont_n2000",
        ("bin", 500): "data_bin", ("bin", 1000): "data_bin_n1000", ("bin", 2000): "data_bin_n2000"}
# (data size, training scenarios N): the main grid on the 500-scenario data, N = 700 / 1400 on the larger data, and N = 350
# repeated on every data size as a bridge between the test splits (75 / 150 / 300 test scenarios).
PLAN = [(500, n) for n in B.NS] + [(1000, 350), (1000, 700), (2000, 350), (2000, 1400)]
NS_ALL = B.NS + [700, 1400]
RUNS_CSV = os.path.join(RESULTS_DIR, "binary_feedback_runs.csv")
DRAWS_CSV = os.path.join(RESULTS_DIR, "binary_feedback_draws_runs.csv")
# label-sampling variance: independent Bernoulli draws for the same scenarios (draw 0 = data_bin*, the first report)
DRAW_PLAN = [(500, 350), (1000, 700), (2000, 1400)]
DRAWS = (1, 2)


def draw_dir(size, d):
    return DIRS[("bin", size)] if d == 0 else f"{DIRS[('bin', size)]}_d{d}"


# ------------------------------------------------------------------ 0/1 training
def bce_loss(m, b):
    return F.binary_cross_entropy(m(b["cls"], b["emotion"], b["day"], b["time"], b.get("location"), b.get("extra")), b["y"])


def sft_target(s_ask, s_act):
    """Observed better action; 0.5 when the two observed labels are equal (they carry no preference)."""
    return 0.5 * (1.0 + torch.sign(s_ask - s_act))


class OursBin(B.Ours):
    """B.Ours with the BCE reward loss and the tie-aware SFT target; everything else is inherited / copied unchanged."""

    epochs = None

    def fit_reward(self, n, rnd_seed, warm):
        d, dev = self.d, self.d.dev
        _, rm_tr, _ = d.prefix(n)
        if self.rm is None or not warm:
            self.rm = B.new_scorer(d.enc, dev, rnd_seed, sigmoid=True)
        h = self._hp("reward")
        self.epochs = B.fit(self.rm, bce_loss, rm_tr, d.rm_val if self.val else None, lr=h["lr"], max_epochs=h["max_epochs"],
                            patience=h["patience"], seed=rnd_seed)

    def fit_policy(self, n, rnd_seed, mode):
        """mode 'scratch': new policy, SFT, DPO; 'continue': DPO only from the current policy (used by drift_trigger.py)."""
        assert mode in ("scratch", "continue")
        d, dev = self.d, self.d.dev
        sc, _, f_tr = d.prefix(n)
        ra, rc = B.reward_scores(self.rm, f_tr)
        va_ra, va_rc = B.reward_scores(self.rm, d.f_val)
        sign_tr = torch.where(ra > rc, 1.0, -1.0)
        sign_va = torch.where(va_ra > va_rc, 1.0, -1.0)
        if mode == "scratch":
            self.pol = B.new_scorer(d.enc, dev, rnd_seed + 100000, sigmoid=False)
            t = lambda x: torch.tensor(x.values, device=dev, dtype=torch.float32)
            y_tr, y_va = sft_target(t(sc.s_ask), t(sc.s_act)), sft_target(t(d.sc_val.s_ask), t(d.sc_val.s_act))
            h = self._hp("sft")
            B.fit(self.pol, B.sft_loss, B.with_extra(f_tr, y_ask=y_tr), B.with_extra(d.f_val, y_ask=y_va) if self.val else None,
                  lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed + 1)
        ref = B.copy.deepcopy(self.pol).eval()
        with torch.no_grad():
            ref_tr, ref_va = B.margin(ref, f_tr), B.margin(ref, d.f_val)
        h = self._hp("dpo")
        B.fit(self.pol, B.dpo_loss, B.with_extra(f_tr, sign=sign_tr, ref=ref_tr),
              B.with_extra(d.f_val, sign=sign_va, ref=ref_va) if self.val else None,
              lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed + 2, train_mode=False)


class TabFreq(B.Tabular):
    """Tabular agent for 0/1 labels: Q[(topic, emotion, day, time, action)] = acceptance frequency of that cell (running
    mean). Fallbacks (topic, action) mean -> global action mean and the tie rule (ask iff Q_ask > Q_act) are those of the
    main experiment's tabular agent."""

    def update(self, sc, order):
        cs = self.__dict__.setdefault("cs", {})
        for i in order:
            r = sc.iloc[i]
            for a, val in (("ask", r.s_ask), ("act", r.s_act)):
                k = (r[self.key_col], *[r[c] for c in self.ctx_cols], a)
                s, c = cs.get(k, (0.0, 0))
                cs[k] = (s + val, c + 1)
                self.Q[k] = (s + val) / (c + 1)
                self.sum[(r[self.key_col], a)] = self.sum.get((r[self.key_col], a), 0.0) + val
                self.cnt[(r[self.key_col], a)] = self.cnt.get((r[self.key_col], a), 0) + 1
                self.gsum[a] += val
                self.gcnt[a] += 1


# ------------------------------------------------------------------ evaluation helpers
def regret(ask, c_ask, c_act):
    return float((np.maximum(c_ask, c_act) - np.where(ask, c_ask, c_act)).mean())


def bayes_bce(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)).mean())


def run_one(label, size, n, seed, data, ctx):
    g, f_grid, grid_base, hard_sets = ctx["g"], ctx["f_grid"], ctx["grid_base"], ctx["hard"][size]
    s = seed * 1000 + n
    ours = (OursBin if label == "bin" else B.Ours)(data, seed)
    ours.fit_round(n, s, warm=False)
    sc_tr, rm_tr, f_tr = data.prefix(n)
    order = np.random.default_rng([seed, n]).permutation(n)
    tabs = {"C": (TabFreq if label == "bin" else B.Tabular)("topic")}
    if label == "bin":
        tabs["Cq"] = B.Tabular("topic")   # reference: the main experiment's constant-lr (0.5) update applied to 0/1 labels
    for t in tabs.values():
        t.update(sc_tr, order)

    row = {"label": label, "size": size, "n": n, "seed": seed, "data": DIRS[(label, size)]}
    sets = {"seen": (data.tests["seen"], data.f_test["seen"]), "unseen": (data.tests["unseen"], data.f_test["unseen"]), **hard_sets}
    dec = {}
    for name, (sc, f) in sets.items():
        a, b = ours.decide(f)
        d = {"A": a, "B": b, **{m: t.decide(sc)[0] for m, t in tabs.items()}}
        dec[name] = d
        for m, x in d.items():
            ev = B.evaluate(x, sc)
            row[f"{m}_{name}"], row[f"{m}_acc_{name}"] = ev["regret"], ev["acc"]
    bl = B.baselines(data.tests["seen"])
    row["base_seen"] = min(bl["always_ask"]["regret"], bl["never_ask"]["regret"])
    for m in dec["seen"]:
        row[f"{m}_hard"] = (row[f"{m}_hard1"] + row[f"{m}_hard2"]) / 2
        row[f"{m}_flip"] = float(np.mean([(dec["seen"][m] != dec[h][m]).mean() for h in ("hard1", "hard2")]))
    # full 1512-cell grid
    a, b = ours.decide(f_grid)
    cg, _ = tabs["C"].decide(g)
    gd = {"A": a, "B": b, "C": cg}
    if "Cq" in tabs:
        gd["Cq"] = tabs["Cq"].decide(g)[0]
    for m, x in gd.items():
        row[f"{m}_grid"] = regret(x, g.c_ask.values, g.c_act.values)
    row["base_grid"] = grid_base
    # what the reward model learned about the ask value (probability / score), on the grid
    with torch.no_grad():
        pa, pc = (x.cpu().numpy() for x in B.reward_scores(ours.rm, f_grid))
    row["spearman_ask"] = float(spearmanr(pa, g.c_ask.values)[0])
    row["spearman_act"] = float(spearmanr(pc, g.c_act.values)[0])
    row["mae_ask"] = float(np.abs(pa - g.c_ask.values).mean())
    row["sat_ratio"] = collapse_ratio(pa, g)
    row["AB_agree_seen"] = float((dec["seen"]["A"] == dec["seen"]["B"]).mean())
    # training diagnostics
    with torch.no_grad():
        row["train_loss"], row["val_loss"] = float(bce_loss(ours.rm, rm_tr) if label == "bin" else B.reward_loss(ours.rm, rm_tr)), \
            float(bce_loss(ours.rm, data.rm_val) if label == "bin" else B.reward_loss(ours.rm, data.rm_val))
    if label == "bin":
        row["epochs_reward"] = ours.epochs
        both = np.concatenate([sc_tr.c_ask.values, sc_tr.c_act.values])
        row["bayes_bce_train"] = bayes_bce(both)
        row["tie_frac"] = float((sc_tr.s_ask.values == sc_tr.s_act.values).mean())
    return row


def make_ctx(enc, dev, datas):
    g = truth_grid(False)
    f_grid = B.feats(g, enc, dev)
    grid_base = min(regret(np.ones(len(g), bool), g.c_ask.values, g.c_act.values),
                    regret(np.zeros(len(g), bool), g.c_ask.values, g.c_act.values))
    hard = {}
    for size in (500, 1000, 2000):
        d = datas[("cont", size)]
        hs = {}
        for j in (0, 1):
            sc = d.tests["seen"].copy()
            sc["question"] = sc.topic.map(lambda t: HARD[t][j])
            hs[f"hard{j + 1}"] = (sc, B.feats(sc, enc, dev))
        hard[size] = hs
    return {"g": g, "f_grid": f_grid, "grid_base": grid_base, "hard": hard}


def ensure_data():
    """Generate any missing dataset (same seed as the main experiment); makes `python -m tact.decision.binary_feedback` self-contained."""
    import subprocess, sys
    for (label, size), d in DIRS.items():
        if not os.path.exists(os.path.join(ROOT, d, "train.csv")):
            cmd = [sys.executable, "-m", "tact.core.generate_dataset", "--n", str(size), "--seed", "42", "--with-act",
                   "--outdir", os.path.join(ROOT, d)] + (["--label", "binary"] if label == "bin" else [])
            subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def ensure_draw_data():
    import subprocess, sys
    for size, _ in DRAW_PLAN:
        for d in DRAWS:
            path = os.path.join(ROOT, draw_dir(size, d))
            if not os.path.exists(os.path.join(path, "train.csv")):
                subprocess.run([sys.executable, "-m", "tact.core.generate_dataset", "--n", str(size), "--seed", "42", "--with-act",
                                "--label", "binary", "--label-offset", str(9 + 100 * d), "--outdir", path], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def train_draws(seeds, out):
    """Re-train the 0/1 models on independent label draws (N = 350 / 700 / 1400); draw 0 at N = 350 is re-run as an integrity check."""
    ensure_data()
    ensure_draw_data()
    dev = get_device()
    enc = FrozenBert(dev)
    datas = {}
    for key in [("cont", 500), ("bin", 500), ("cont", 1000), ("bin", 1000), ("cont", 2000), ("bin", 2000)]:   # same construction order as train_and_eval
        datas[key] = B.Data(enc, dev, data_dir=os.path.join(ROOT, DIRS[key]))
    drawn = {(size, d): B.Data(enc, dev, data_dir=os.path.join(ROOT, draw_dir(size, d))) for size, _ in DRAW_PLAN for d in DRAWS}
    enc.encode(sorted({h for t in HARD.values() for h in t}))
    ctx = make_ctx(enc, dev, datas)
    rows = []
    jobs = [(0, 500, 350)] + [(d, size, n) for d in DRAWS for size, n in DRAW_PLAN]
    for d, size, n in jobs:
        data = datas[("bin", size)] if d == 0 else drawn[(size, d)]
        for seed in seeds:
            r = run_one("bin", size, n, seed, data, ctx)
            r["draw"], r["data"] = d, draw_dir(size, d)
            rows.append(r)
            print(f"[draw {d} size={size} N={n:4d} seed={seed}] test A/B/C = {r['A_seen']:.3f}/{r['B_seen']:.3f}/{r['C_seen']:.3f} "
                  f"grid A = {r['A_grid']:.3f} sat {r['sat_ratio']:.2f}", flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


def train_and_eval(labels, plan, seeds, out):
    ensure_data()
    dev = get_device()
    enc = FrozenBert(dev)
    datas = {}
    for key in [("cont", 500), ("bin", 500), ("cont", 1000), ("bin", 1000), ("cont", 2000), ("bin", 2000)]:   # cont-500 first: same cache priming as budget.py
        datas[key] = B.Data(enc, dev, data_dir=os.path.join(ROOT, DIRS[key]))
    enc.encode(sorted({h for t in HARD.values() for h in t}))   # after the standard texts: the models do not change
    ctx = make_ctx(enc, dev, datas)
    rows = []
    for label in labels:
        for size, n in plan:
            for seed in seeds:
                t0 = time.perf_counter()
                r = run_one(label, size, n, seed, datas[(label, size)], ctx)
                rows.append(r)
                print(f"[{label} size={size} N={n:4d} seed={seed}] test regret A/B/C = {r['A_seen']:.3f}/{r['B_seen']:.3f}/{r['C_seen']:.3f} "
                      f"grid A/C = {r['A_grid']:.3f}/{r['C_grid']:.3f} spearman {r['spearman_ask']:.3f} sat {r['sat_ratio']:.2f} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
                pd.DataFrame(rows).to_csv(out, index=False)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ report
def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f}"


def main_rows(df, label):
    """The N grid used for the curves: N <= 350 from the 500-scenario data, 700 from the 1000 data, 1400 from the 2000 data."""
    d = df[df.label == label]
    keep = ((d["size"] == 500) | ((d["size"] == 1000) & (d.n == 700)) | ((d["size"] == 2000) & (d.n == 1400)))
    return d[keep]


def curve(df, label, col):
    d = main_rows(df, label)
    g = d.groupby("n")[col].agg(["mean", "std"])
    return g.reindex(NS_ALL)


def first_reach(mean_by_n, thr):
    for n in NS_ALL:
        if n in mean_by_n.index and mean_by_n[n] <= thr:
            return n
    return None


def overtake(a, c):
    """First N from which a < c for every larger N on the grid (None if never / not at the last N)."""
    ok = [n for n in NS_ALL if a[n] < c[n]]
    for n in NS_ALL:
        if all(m in ok for m in NS_ALL if m >= n):
            return n
    return None


def interp_n(mean_by_n, thr):
    """log-linear interpolation of the N at which the mean curve crosses thr (None if it never does)."""
    ns = [n for n in NS_ALL if n in mean_by_n.index]
    for i in range(1, len(ns)):
        a, b = mean_by_n[ns[i - 1]], mean_by_n[ns[i]]
        if a > thr >= b:
            return float(np.exp(np.log(ns[i - 1]) + (a - thr) / (a - b) * (np.log(ns[i]) - np.log(ns[i - 1]))))
    return None


def figure(df, fname):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    spec = [("test_seen: 75 / 150 / 300 held-out scenarios (main-experiment evaluation)", "seen", False),
            ("full 1512-cell grid (normalised by min(always-ask, never-ask))", "grid", True)]
    for ax, (title, key, norm) in zip(axes, spec):
        for lab, ls_, mk in (("cont", "-", "o"), ("bin", "--", "s")):
            for m, col, nm in (("A", "#2a78d6", ("TACT (threshold)" if V2 else "TACT (SFT + DPO)")), ("C", "#1baf7a", "tabular")):
                cs = curve(df, lab, f"{m}_{key}")
                scale = curve(df, lab, f"base_{key}")["mean"] if norm else 1.0
                mu, sd = cs["mean"] / scale, cs["std"] / scale
                ax.plot(NS_ALL, mu, color=col, ls=ls_, marker=mk, ms=4.5, lw=1.8,
                        label=f"{nm}, {'continuous' if lab == 'cont' else '0/1 feedback'}")
                ax.fill_between(NS_ALL, mu - sd, mu + sd, color=col, alpha=0.10, lw=0)
        if not norm:
            b = curve(df, "cont", "base_seen")["mean"]
            ax.plot(NS_ALL, b, color=plots.INK2, ls=":", lw=1.2, label="min(always-ask, never-ask) on that test split")
        else:
            ax.axhline(1.0, color=plots.INK2, ls=":", lw=1.2)
        ax.set_xscale("log")
        ax.minorticks_off()
        ax.set_xticks(NS_ALL, [str(x) for x in NS_ALL], fontsize=8)
        ax.set_xlabel("training scenarios N (700, 1400: larger datasets)", fontsize=9)
        ax.set_title(title, fontsize=9.5, loc="left", color=plots.INK)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("regret (clean truth; lower is better)")
    axes[1].set_ylabel("normalised regret")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, fontsize=8.5)
    fig.suptitle("Continuous scores vs. 0/1 accept-reject feedback (mean ± std over 5 seeds)", x=0.01, ha="left", fontsize=11, color=plots.INK2)
    fig.tight_layout(rect=(0, 0.11, 1, 0.95), w_pad=3)
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=170)
    plt.close(fig)


def report(df):
    out = ["# Experiment A: 0/1 (accept / reject) feedback instead of continuous scores (`binary_feedback.py`)\n"]
    cont, bin_ = main_rows(df, "cont"), main_rows(df, "bin")
    out += ["## Summary of findings (numbers from the tables below; 5 optimisation seeds, one Bernoulli label draw per dataset)\n",
            "1. **0/1 feedback costs data, not the method.** TACT with 0/1 labels reaches a test regret of 0.068 ± 0.033 at N = 350 (continuous: "
            "0.036 ± 0.008) and 0.050 ± 0.004 at N = 1400 (continuous 0.021 ± 0.007). The continuous N = 350 regret of 0.036 is **not reached by "
            "N = 1400** on the test-split measure (closest 0.050, 1.39×). On the full grid the specified target (0.911 normalised) is reached at "
            "N = 700 (2×), the dataset-averaged target (0.746) between 700 and 1400 (interpolated ≈ 3×, first grid point 1400 = 4×). "
            "A fair summary is **2–4× more scenarios or more**; the target itself moves by ±30% with the dataset, so no finer figure is supported.",
            "2. **The crossover with the tabular agent moves right.** Continuous: TACT is below tabular from N = 75 (test split) / 50 (grid). 0/1: "
            "from N = 250 (test split) / 350 (grid). Tabular itself is hurt more by 0/1 labels than TACT is at large N (grid, N = 1400: tabular 1.32 "
            "vs TACT 0.65 in normalised regret).",
            "3. **What is lost is mostly the habit rule and fine ranking.** Spearman against r* is lower at every N from 50 (0.73 vs 0.85 at "
            "N = 350; 0.87 vs 0.96 at N = 1400) and the mean absolute error two to three times as large (0.128 vs 0.066 at N = 350; 0.073 vs 0.026 at N = 1400). The Saturday-fruit rule is **not learned at "
            "any N tested**: collapse ratio 0.94 ± 0.05 at N = 350 (L = 0.07), and above 1 at N = 700 / 1400 (1.21 / 1.18: Saturday predicted "
            "*higher* than the other days), against 0.68 (L = 0.38) and 0.36 (L = 0.76) for continuous labels at N = 350 / 1400. Each such scenario "
            "gives one Bernoulli(≈0.07) ask label.",
            "4. **Small N is degenerate.** At N ≤ 75 the 0/1 TACT policy is constant never-ask in all 15 runs, and at N = 100–150 its decisions "
            "flip on 35–80% of scenarios under paraphrase (both TACT and argmax). The 0/1 learning curve is also not monotone (0.131 at N = 100, 0.148 "
            "at N = 150) and seed-dependent at N = 250–350 (one of five seeds stays at 0.12 at N = 350).",
            "5. **DPO's robustness advantage over argmax persists only partly.** Decision-flip rate under hard paraphrases, TACT vs argmax: N = 350: "
            "4.9% vs 24.8% (TACT lower in 5 of 5 seeds; continuous 4.8% vs 20.0%) — the advantage persists. N = 700: 8.9% vs 10.5% (4 of 5 seeds, paired "
            "difference −0.017 ± 0.027; continuous 8.5% vs 25.1%) — it is essentially gone; N = 1400: 13.5% vs 9.7% (reversed). In regret on the "
            "paraphrases, TACT is not better than argmax under 0/1 labels at N = 350 (0.067 vs 0.063) or 700 (0.072 vs 0.071), whereas under continuous "
            "labels it is (0.050 vs 0.092; 0.049 vs 0.071). So under 0/1 labels the benefit of DPO is a lower flip rate at some N, not a lower regret.",
            "6. **Hyper-parameters were not touched**: BCE converged with the main learning rate (16–41 epochs of 100 in every run).\n",
            "## Deviations from the plan and things to keep in mind\n",
            "- Continuous references for N = 700 / 1400 did not exist (the main experiment stops at 350); they were generated "
            "(`data_cont_n1000`, `data_cont_n2000`) and trained with the same code. N = 700 / 1400 use different datasets and larger test splits "
            "(150 / 300) than N ≤ 350 (75): the full-grid measure is added because it does not change with the split; levels on the two panels "
            "of the figure are therefore not comparable across the data-size switch at N = 350 → 700.",
            "- SFT tie rule (target 0.5 when both labels are equal; 49% of the training scenarios at N = 350) and the tabular rule (frequency per cell) "
            "are my choices. Both the frequency tabular agent and the lr-0.5 variant are reported; they differ little.",
            "- **Only one Bernoulli draw per dataset.** The 5 seeds do not cover label-sampling variance, so the p-values and ± values understate the "
            "uncertainty of the 0/1 curves, particularly at N = 250–350 where seeds already disagree. A second label draw was not run.\n"]
    d_bin = pd.read_csv(os.path.join(ROOT, DIRS[("bin", 500)], "dataset.csv"))
    # ---- setup
    out += ["## Setup\n",
            "- **Truth unchanged.** ask r* = N·T, act r* = 1 − N (noise-free, `score_clean`); evaluation (regret, accuracy, grid) is the "
            "main experiment's, on the continuous truth. Only the observed label changes: y ~ Bernoulli(r*), one draw per scenario and "
            "candidate, no thresholding (`generate_dataset.py --label binary`; the default continuous output is byte-identical to the "
            "existing data: checked for dataset/train/val/test_seen/test_unseen/config of data_dpo and for the 3-variable files).",
            f"- Data: `data_bin` (500 scenarios, 350/75/75; same scenarios, order and splits as `data_dpo`), `data_bin_n1000` / "
            "`data_bin_n2000` (700 / 1400 training scenarios; 150 / 300 validation and test scenarios). Continuous references for "
            "N = 700 / 1400: `data_cont_n1000`, `data_cont_n2000` (plain `--n 1000/2000 --with-act`; the main experiment stops at "
            "N = 350). The N ≤ 350 continuous curve is re-run here with the identical code and reproduces `budget_runs.csv` "
            f"(max abs difference of A/B/C test_seen regret: {{CHECK}}).",
            f"- Mean label vs mean truth in `data_bin` (ask rows): {d_bin[d_bin.cand == 'ask'].score.mean():.3f} vs "
            f"{d_bin[d_bin.cand == 'ask'].score_clean.mean():.3f}; act rows: {d_bin[d_bin.cand == 'act'].score.mean():.3f} vs "
            f"{d_bin[d_bin.cand == 'act'].score_clean.mean():.3f}. The share of training scenarios whose ask and act labels are "
            f"equal (ties; no preference information for SFT): {{TIE}}.",
            "- **Training changes (only these).** Reward model: same architecture (the sigmoid output is the acceptance probability), loss "
            "MSE → binary cross-entropy. SFT target: 1 if only ask was accepted, 0 if only act was accepted, **0.5 for ties** (my "
            "choice; the continuous data have no ties). DPO: unchanged (pairs labelled by reward-model predictions; β = 0.1; reference = "
            "SFT policy). Tabular C: Q = acceptance frequency per (topic, emotion, day, time, action) cell, fallbacks and the "
            "ask-iff-Q_ask>Q_act rule as in the main experiment (**Cq** = the main experiment's constant-lr 0.5 update applied to 0/1 "
            "labels, shown as a reference). Hyper-parameters: `budget.HP`, learning rates, epochs and patience untouched — no tuning. "
            "Seeds 0–4 and the per-run seeds `seed·1000 + N` are those of the main experiment; the 5 seeds vary initialisation and "
            "minibatch order, **not** the Bernoulli draw (one label realisation per dataset).",
            "- Early stopping uses the validation split of each dataset (75 / 150 / 300 scenarios, labelled with the same kind of feedback): "
            "extra data beyond N, as in the main experiment.\n"]

    # ---- convergence
    b350 = df[(df.label == "bin") & (df["size"] == 500) & (df.n == 350)]
    out += ["## Does BCE converge with the main hyper-parameters?\n",
            "| N | epochs run (max 100) | train BCE | val BCE | BCE of the true probabilities r* on the training rows (floor) |", "|---|---|---|---|---|"]
    for n in NS_ALL:
        d = bin_[bin_.n == n]
        out.append(f"| {n} | {ms(d.epochs_reward, 0)} | {ms(d.train_loss)} | {ms(d.val_loss)} | {d.bayes_bce_train.mean():.3f} |")
    bb = df[df.label == "bin"]
    out.append(f"\nEvery one of the {len(bb)} 0/1 runs stopped early ({int(bb.epochs_reward.min())}–{int(bb.epochs_reward.max())} epochs, limit 100), "
               "so BCE converges with the main learning rate and **no learning-rate change was made**. The floor column is the BCE of the true "
               "probabilities r* on the (realised) training rows; the validation BCE stays above it in every row, and "
               f"{int((bb.train_loss < bb.bayes_bce_train).sum())} of {len(bb)} runs fit their realised training labels below it (mild over-fitting of the "
               "Bernoulli draw, which early stopping on the validation split limits).\n")

    # ---- curve
    out += ["## Regret vs. N: continuous vs. 0/1 feedback (mean ± std over 5 seeds)\n",
            "Test split regret as in the main experiment (75 test scenarios for N ≤ 350, 150 for N = 700, 300 for N = 1400, so the "
            "level shifts a little at the data-size switch), and the 1512-cell grid, whose size does not change. "
            "A = TACT policy (SFT + DPO), B = argmax of the reward model, C = tabular (frequency for 0/1 labels).\n",
            "| N | TACT cont | TACT 0/1 | B cont | B 0/1 | tabular cont | tabular 0/1 | tabular 0/1 (lr 0.5) | min(always-, never-ask) on the split |",
            "|---|---|---|---|---|---|---|---|---|"]
    for n in NS_ALL:
        c, b = cont[cont.n == n], bin_[bin_.n == n]
        out.append(f"| {n} | {ms(c.A_seen)} | {ms(b.A_seen)} | {ms(c.B_seen)} | {ms(b.B_seen)} | {ms(c.C_seen)} | {ms(b.C_seen)} | "
                   f"{ms(b.Cq_seen)} | {c.base_seen.mean():.3f} |")
    out += ["\nSame, on the full 1512-cell grid (regret divided by min(always-ask, never-ask) = "
            f"{df.base_grid.iloc[0]:.3f}; 1 = no better than the better constant policy):\n",
            "| N | TACT cont | TACT 0/1 | B cont | B 0/1 | tabular cont | tabular 0/1 | tabular 0/1 (lr 0.5) |", "|---|---|---|---|---|---|---|---|"]
    for n in NS_ALL:
        c, b = cont[cont.n == n], bin_[bin_.n == n]
        k = df.base_grid.iloc[0]
        f = lambda x: ms(x / k)
        out.append(f"| {n} | {f(c.A_grid)} | {f(b.A_grid)} | {f(c.B_grid)} | {f(b.B_grid)} | {f(c.C_grid)} | {f(b.C_grid)} | {f(b.Cq_grid)} |")
    nb = bin_[bin_.n <= 75]
    bl500 = B.baselines(B.scenarios(B.load_long("test_seen.csv", os.path.join(ROOT, DIRS[("bin", 500)]))))
    out.append(f"\n**Degenerate region.** For N ≤ 75 the 0/1 TACT policy is the constant *never-ask* policy in "
               f"{int(np.isclose(nb.A_seen, nb.base_seen).sum())} of {len(nb)} runs "
               f"(regret equals the never-ask regret {bl500['never_ask']['regret']:.3f}, test accuracy {nb.A_acc_seen.mean():.2f} vs never-ask accuracy "
               f"{bl500['never_ask']['acc']:.2f}, 0% flips under paraphrase); at N = 25–75 the 0/1 reward model predicts a higher acceptance "
               "probability for act than for ask on (nearly) every context (the mean act label, 0.35, is above the mean ask label, 0.31). "
               "The 0/1 tabular agent has the same regret as the lr-0.5 variant (Cq) up to N = 250 because each cell is visited at most about twice.")
    out += ["\nBridge (the same N = 350 on each data size, to separate the effect of N from the effect of changing the dataset/test split):\n",
            "| data size | label | TACT test | TACT grid (normalised) | tabular test | tabular grid (normalised) |", "|---|---|---|---|---|---|"]
    for size in (500, 1000, 2000):
        for lab in ("cont", "bin"):
            d = df[(df.label == lab) & (df["size"] == size) & (df.n == 350)]
            k = d.base_grid.iloc[0]
            out.append(f"| {size} | {'continuous' if lab == 'cont' else '0/1'} | {ms(d.A_seen)} | {ms(d.A_grid / k)} | {ms(d.C_seen)} | {ms(d.C_grid / k)} |")
    out.append("")

    # ---- label efficiency
    _tt = " / ".join(f"{df[(df.label == 'cont') & (df['size'] == z) & (df.n == 350)].A_seen.mean():.3f}" for z in (500, 1000, 2000))
    _gg = " / ".join(f"{(df[(df.label == 'cont') & (df['size'] == z) & (df.n == 350)].A_grid / df.base_grid.iloc[0]).mean():.2f}" for z in (500, 1000, 2000))
    out += ["## Label efficiency\n",
            f"The continuous N = 350 target is not stable across datasets: the same N = 350 gives a test regret of {_tt} and a normalised "
            f"grid regret of {_gg} on the 500 / 1000 / 2000-scenario data (bridge table), a spread larger than the seed std. The "
            "multiples below therefore come in two forms: the target as specified (the main experiment's N = 350 on `data_dpo`) and a "
            "dataset-averaged target (mean of the continuous N = 350 results on the three data sizes).\n"]
    for key, nm, norm in (("seen", "test split (the main experiment's measure)", False), ("grid", "full grid (normalised)", True)):
        cm = curve(df, "cont", f"A_{key}")["mean"]
        bm = curve(df, "bin", f"A_{key}")["mean"]
        if norm:
            cm, bm = cm / df.base_grid.iloc[0], bm / df.base_grid.iloc[0]
        thr = cm[350]
        n_reach = first_reach(bm, thr)
        n_int = interp_n(bm, thr)
        closest = min(NS_ALL, key=lambda n: abs(bm[n] - thr))
        if n_reach is None:
            s = (f"0/1 feedback does **not** reach it up to N = 1400; closest value: N = {closest}, regret {bm[closest]:.3f} "
                 f"({bm[closest] / thr:.2f}× the target)")
        else:
            s = f"first N on the grid at or below it: **{n_reach}** (log-linear interpolation of the mean curve: {n_int:.0f})" if n_int else f"first N on the grid at or below it: **{n_reach}**"
            s += f"; multiple of the continuous budget: **{n_reach / 350:.1f}×** (interpolated {n_int / 350:.1f}×)" if n_int else f"; multiple {n_reach / 350:.1f}×"
        out.append(f"- {nm}, specified target: continuous TACT at N = 350 has regret {thr:.3f}; {s}. 0/1 TACT regret at N = 350: {bm[350]:.3f}, 700: {bm[700]:.3f}, 1400: {bm[1400]:.3f}.")
        col = f"A_{key}"
        vals = [df[(df.label == "cont") & (df["size"] == z) & (df.n == 350)][col].mean() / (df.base_grid.iloc[0] if norm else 1.0) for z in (500, 1000, 2000)]
        thr2 = float(np.mean(vals))
        r2, i2 = first_reach(bm, thr2), interp_n(bm, thr2)
        near = min(NS_ALL, key=lambda n: abs(bm[n] - thr2))
        out.append(f"- {nm}, dataset-averaged target {thr2:.3f} (mean of {', '.join(f'{v:.3f}' for v in vals)}): " +
                   (f"first N at or below it: **{r2}** ({r2 / 350:.0f}× of 350; interpolated {i2:.0f}, {i2 / 350:.1f}×)." if r2 else
                    f"**not reached** up to N = 1400; closest N = {near}, regret {bm[near]:.3f} ({bm[near] / thr2:.2f}× the target)."))
    out.append("")
    # welch at 350 and the last N
    tests = []
    for n in (350, 700, 1400):
        c, b = cont[cont.n == n].A_grid.values, bin_[bin_.n == n].A_grid.values
        tests.append(f"N = {n}: {np.mean(b) / np.mean(c):.2f}× (Welch p = {ttest_ind(b, c, equal_var=False).pvalue:.3g})")
    out.append("Ratio of grid regret, 0/1 TACT vs continuous TACT: " + "; ".join(tests) + ". (p-values use only the 5 optimisation seeds; "
               "they do not cover the Bernoulli draw.)\n")

    # ---- crossover
    out += ["## Crossover with the tabular agent\n"]
    for key, nm in (("seen", "test split"), ("grid", "full grid")):
        row = []
        for lab, ln in (("cont", "continuous"), ("bin", "0/1")):
            a, c = curve(df, lab, f"A_{key}")["mean"], curve(df, lab, f"C_{key}")["mean"]
            first = [n for n in NS_ALL if a[n] < c[n]]
            row.append(f"{ln}: TACT first below tabular at N = {first[0] if first else 'never'}, and for every larger N from N = "
                       f"{overtake(a, c) if overtake(a, c) else 'never'}")
        out.append(f"- {nm}: " + "; ".join(row) + ".")
    out.append("")

    out += ["## Spread over seeds at N = 350 (0/1 labels, `data_bin`)\n",
            "| seed | TACT test regret | argmax test regret | TACT grid regret (normalised) | Spearman | collapse ratio |", "|---|---|---|---|---|---|"]
    for r in b350.sort_values("seed").itertuples():
        out.append(f"| {r.seed} | {r.A_seen:.3f} | {r.B_seen:.3f} | {r.A_grid / r.base_grid:.2f} | {r.spearman_ask:.3f} | {r.sat_ratio:.2f} |")
    if V2:
        out.append(f"\nTACT test regret over the 5 seeds at N = 350 (this label draw): minimum {b350.A_seen.min():.3f}, maximum {b350.A_seen.max():.3f}; this is one label draw only "
                   "(see the label-sampling section below for the others).\n")
    else:
        out.append("\nThe 0/1 results at N = 250–350 are driven by individual seeds (one seed out of five stays at regret 0.12 at N = 350): "
                   "with 0/1 labels the learning curve is less reliable than the mean suggests.\n")
    # ---- spearman / collapse
    out += ["## What the reward model learned (full grid, N = 350 and others)\n",
            "Spearman between the predicted ask value (continuous score, or acceptance probability for 0/1) and the true ask value r* over "
            "the 1512 grid cells; mean absolute error of that prediction; Saturday-fruit collapse ratio (Saturday / mean of the other six "
            "days, fruit, afternoon, ask score; truth 0.15 — the habit rule — and 1.0 if a model ignored the rule), and the rule-learning "
            "degree L = (1 − predicted ratio)/(1 − 0.15) (L = 1 learned fully, 0 not at all).\n",
            "| N | Spearman cont | Spearman 0/1 | MAE cont | MAE 0/1 | collapse ratio cont | collapse ratio 0/1 | L cont | L 0/1 |", "|---|---|---|---|---|---|---|---|---|"]
    truth_ratio = collapse_ratio(truth_grid(False).c_ask.values, truth_grid(False))
    for n in NS_ALL:
        c, b = cont[cont.n == n], bin_[bin_.n == n]
        L = lambda d: (1 - d.sat_ratio) / (1 - truth_ratio)
        out.append(f"| {n} | {ms(c.spearman_ask)} | {ms(b.spearman_ask)} | {ms(c.mae_ask)} | {ms(b.mae_ask)} | {ms(c.sat_ratio, 2)} | "
                   f"{ms(b.sat_ratio, 2)} | {ms(L(c), 2)} | {ms(L(b), 2)} |")
    cnt = {}
    for z, dn in ((500, "data_dpo"), (1000, "data_cont_n1000"), (2000, "data_cont_n2000")):
        tr = B.scenarios(B.load_long("train.csv", os.path.join(ROOT, dn)))
        cnt[z] = [int(((tr.topic == "fruit") & (tr.day == 5)).iloc[:n].sum()) for n in (350, 700, 1400)]
    out.append(f"\nTruth collapse ratio = {truth_ratio:.2f}. **Evidence for the Saturday-fruit rule in the training data** (scenarios with topic "
               f"fruit and day Saturday among the first N; the same scenarios for both label types): `data_dpo` N = 350: {cnt[500][0]}; 1000-scenario "
               f"data N = 350 / 700: {cnt[1000][0]} / {cnt[1000][1]}; 2000-scenario data N = 350 / 700 / 1400: {cnt[2000][0]} / {cnt[2000][1]} / {cnt[2000][2]}. "
               + ("The count does not grow smoothly with N across the three datasets; the continuous L at N = 350 / 700 / 1400 is "
                  + " / ".join(f"{((1 - cont[cont.n == n].sat_ratio) / (1 - truth_ratio)).mean():.2f}" for n in (350, 700, 1400))
                  + " (table above); other causes were not isolated.\n" if V2 else
                  "The count does not grow smoothly with N across the three datasets, which is one reason the continuous L is not monotone "
                  "(0.38 at 350, 0.12 at 700, 0.76 at 1400); I did not isolate other causes.\n"))

    # ---- hard paraphrases
    out += ["## Robustness to hard paraphrases: DPO policy (A) vs direct argmax (B)\n",
            "Zero-shot, as in `hard_paraphrase.md`: the question of the test scenarios is replaced by one of the 24 hand-written "
            "statement-form paraphrases (hard1, hard2); `flip` = share of decisions that change from the seen wording to the paraphrase "
            "(mean of the two). N = 350 on the 500-scenario data (75 test scenarios), N = 700 on the 1000-scenario data (150).\n",
            "| N | label | A regret seen → hard | B regret seen → hard | A flip rate | B flip rate | paired (A − B) flip, mean ± std | "
            "seeds with A flip < B flip |", "|---|---|---|---|---|---|---|---|"]
    for n in (350, 700):
        for lab in ("cont", "bin"):
            d = main_rows(df, lab)
            d = d[d.n == n].sort_values("seed")
            diff = d.A_flip.values - d.B_flip.values
            out.append(f"| {n} | {'continuous' if lab == 'cont' else '0/1'} | {ms(d.A_seen)} → {ms(d.A_hard)} | {ms(d.B_seen)} → {ms(d.B_hard)} | "
                       f"{d.A_flip.mean():.1%} ± {d.A_flip.std(ddof=1):.1%} | {d.B_flip.mean():.1%} ± {d.B_flip.std(ddof=1):.1%} | "
                       f"{diff.mean():+.3f} ± {diff.std(ddof=1):.3f} | {int((diff < 0).sum())} of 5 |")
    out.append("\nAll N (flip rate and regret on the paraphrases):\n")
    out += ["| N | A flip cont | A flip 0/1 | B flip cont | B flip 0/1 | A hard regret cont | A hard regret 0/1 | B hard regret cont | B hard regret 0/1 |", "|---|---|---|---|---|---|---|---|---|"]
    for n in NS_ALL:
        c, b = cont[cont.n == n], bin_[bin_.n == n]
        out.append(f"| {n} | {c.A_flip.mean():.1%} | {b.A_flip.mean():.1%} | {c.B_flip.mean():.1%} | {b.B_flip.mean():.1%} | "
                   f"{ms(c.A_hard)} | {ms(b.A_hard)} | {ms(c.B_hard)} | {ms(b.B_hard)} |")
    return out


METRICS = [("A_seen", "TACT test regret", 1.0), ("A_grid", "TACT grid regret (normalised)", None), ("L", "habit L (Saturday-fruit)", 1.0),
           ("A_flip", "TACT flip rate", 1.0), ("B_flip", "argmax flip rate", 1.0), ("flip_gap", "flip A − B", 1.0), ("spearman_ask", "Spearman", 1.0)]


def with_derived(d):
    d = d.copy()
    d["L"] = (1 - d.sat_ratio) / (1 - 0.15)
    d["A_grid"] = d.A_grid / d.base_grid
    d["flip_gap"] = d.A_flip - d.B_flip
    return d


def draws_section(df, dr):
    """Label-sampling variance: draw 0 (the first report) + draws 1, 2, each with the 5 optimisation seeds."""
    base = df[(df.label == "bin")]
    first = pd.concat([base[(base["size"] == sz) & (base.n == n)] for sz, n in DRAW_PLAN]).assign(draw=0)
    new = dr[dr.draw > 0]
    allr = with_derived(pd.concat([first, new], ignore_index=True))
    cont = with_derived(pd.concat([df[(df.label == "cont") & (df["size"] == sz) & (df.n == n)] for sz, n in DRAW_PLAN]))
    chk = dr[dr.draw == 0].sort_values("seed")
    ref = base[(base["size"] == 500) & (base.n == 350)].sort_values("seed")
    diff = max(float((chk[c].values - ref[c].values).__abs__().max()) for c in ("A_seen", "B_seen", "C_seen", "A_grid", "A_flip", "sat_ratio"))
    out = ["\n## Added after the first report: label-sampling variance (3 independent Bernoulli draws × 5 optimisation seeds)\n",
           "Requested follow-up. For N = 350 / 700 / 1400 (on the 500 / 1000 / 2000-scenario data) the 0/1 labels of the *same* scenarios were drawn "
           "two more times (`generate_dataset.py --label binary --label-offset 109 / 209`, directories `data_bin*_d1`, `data_bin*_d2`; draw 0 = "
           "`data_bin*`, the data of everything above), and the models were retrained with seeds 0–4 on each. Nothing else changed (no tuning). "
           "Scenarios, splits and truth are identical across draws; the labels of one draw agree with another in only about 65% of the "
           f"scenario/candidate rows. Integrity check: re-running draw 0 at N = 350 reproduces the numbers above (max abs difference over test regret "
           f"A/B/C, grid regret, flip rate and collapse ratio: {diff:.1e}). The 15 runs per N are 3 draws × 5 seeds; the draw-0 runs are the "
           "ones from the main table (not re-run, except for the check).\n",
           "Metrics: test_seen regret of the TACT policy; its grid regret normalised by min(always-, never-ask) = 0.050; habit learning degree "
           "L = (1 − collapse ratio)/(1 − 0.15) of the Saturday-fruit rule; decision-flip rate under the hard paraphrases for TACT and for "
           "argmax and their difference; Spearman of the predicted ask probability against r*.\n"]
    for n in (350, 700, 1400):
        out += [f"### N = {n}: per draw (mean ± std over the 5 seeds)\n", "| draw | " + " | ".join(m[1] for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
        for d in (0, 1, 2):
            g = allr[(allr.n == n) & (allr.draw == d)]
            out.append(f"| {d}{' (first report)' if d == 0 else ''} | " + " | ".join(ms(g[m[0]], 3) for m in METRICS) + " |")
        c = cont[cont.n == n]
        out.append("| continuous, 1 draw of the noise | " + " | ".join(ms(c[m[0]], 3) for m in METRICS) + " |\n")
    out += ["### Pooled over the 3 draws (15 runs per N)\n",
            "`total std` = std over the 15 runs (optimisation and label-sampling variance together); `within` = root-mean-square of the three "
            "per-draw 5-seed stds (the only variance the first report could show); `between` = std of the three draw means; `std of draw 0 alone` is "
            "what the first report printed. With three draws the between-draw std is itself rough.\n",
            "| N | metric | pooled mean | total std (15 runs) | within-draw | between-draw | std of draw 0 alone | draw means 0 / 1 / 2 |", "|---|---|---|---|---|---|---|---|"]
    for n in (350, 700, 1400):
        for key, name, _ in METRICS:
            g = allr[allr.n == n]
            per = [g[g.draw == d][key].values for d in (0, 1, 2)]
            within = float(np.sqrt(np.mean([x.std(ddof=1) ** 2 for x in per])))
            between = float(np.std([x.mean() for x in per], ddof=1))
            out.append(f"| {n} | {name} | {g[key].mean():.3f} | {g[key].std(ddof=1):.3f} | {within:.3f} | {between:.3f} | {per[0].std(ddof=1):.3f} | "
                       + " / ".join(f"{x.mean():.3f}" for x in per) + " |")
    out += ["\n### Pooled 0/1 vs continuous and the DPO-vs-argmax flip comparison (15 runs per N)\n",
            "| N | TACT test regret 0/1 pooled | continuous | ratio | TACT grid 0/1 pooled | continuous | ratio | L 0/1 pooled | L continuous | "
            "flip A / flip B (0/1 pooled) | A flip < B flip in runs | continuous flip A / B |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for n in (350, 700, 1400):
        g, c = allr[allr.n == n], cont[cont.n == n]
        out.append(f"| {n} | {ms(g.A_seen)} | {ms(c.A_seen)} | {g.A_seen.mean() / c.A_seen.mean():.2f}× | {ms(g.A_grid)} | {ms(c.A_grid)} | "
                   f"{g.A_grid.mean() / c.A_grid.mean():.2f}× | {ms(g.L, 2)} | {ms(c.L, 2)} | {g.A_flip.mean():.1%} / {g.B_flip.mean():.1%} | "
                   f"{int((g.flip_gap < 0).sum())} of {len(g)} | {c.A_flip.mean():.1%} / {c.B_flip.mean():.1%} |")
    out += ["\n### TACT vs the tabular agent on the same draws (tabular is deterministic given a draw, so each of the 15 TACT runs is compared with its draw's tabular value)\n",
            "| N | TACT test regret (pooled) | tabular test regret, draws 0 / 1 / 2 | TACT < tabular (test), runs | TACT grid (norm., pooled) | tabular grid (norm.), draws 0 / 1 / 2 | TACT < tabular (grid), runs |",
            "|---|---|---|---|---|---|---|"]
    for n in (350, 700, 1400):
        g = allr[allr.n == n]
        ct = [float(g[g.draw == d].C_seen.iloc[0]) for d in (0, 1, 2)]
        cg = [float((g[g.draw == d].C_grid / g[g.draw == d].base_grid).iloc[0]) for d in (0, 1, 2)]
        wins_t = int(sum((g[g.draw == d].A_seen < ct[d]).sum() for d in (0, 1, 2)))
        wins_g = int(sum((g[g.draw == d].A_grid < cg[d]).sum() for d in (0, 1, 2)))
        out.append(f"| {n} | {ms(g.A_seen)} | " + " / ".join(f"{x:.3f}" for x in ct) + f" | {wins_t} of 15 | {ms(g.A_grid)} | "
                   + " / ".join(f"{x:.2f}" for x in cg) + f" | {wins_g} of 15 |")
    out += ["\n### Corrections to the summary at the top of this file (which rests on draw 0 alone; the text above is left unchanged)\n",
            "- **Draw 0 was the most favourable of the three at N = 350.** Pooled over the 15 runs the 0/1 TACT test regret at N = 350 is "
            "**0.109 ± 0.042** (draw means 0.068 / 0.117 / 0.142), 3.0× the continuous 0.036, not 0.068 (1.9×); grid regret 2.14 ± 1.00 "
            "(1.6 / 1.8 / 3.0). At N = 350 the between-draw std (0.038) is larger than the within-draw std (0.029): the ± values of the first report "
            "understated the uncertainty (L: std over 15 runs 0.32 vs 0.06 for draw 0 alone; grid regret 1.00 vs 0.75). At N = 700 and 1400 the "
            "draws agree much better (test regret 0.054 ± 0.015 and 0.052 ± 0.005; between-draw std 0.016 and 0.002).",
            "- **Label efficiency (summary item 1) stands, with a worse N = 350.** The test-split target 0.036 is still not reached at N = 1400 "
            "(pooled 0.052 ± 0.005, 1.44× the target; 1.22× the dataset-averaged target 0.043). On the grid, the specified target 0.911 is met "
            "on average at N = 700 (pooled 0.863 ± 0.197, draw means 0.89 / 0.70 / 1.00, i.e. it is a borderline hit) and the dataset-averaged "
            "target 0.746 is met at N = 1400 (0.644 ± 0.067) but not at N = 700. So 2–4× more scenarios or more remains the supported statement.",
            "- **Crossover with the tabular agent (item 2) is less certain than reported.** TACT beats the tabular agent of the same draw in only "
            "9 of 15 runs at N = 350 (test split and grid alike), but in 15 of 15 at N = 700 and at N = 1400. The 0/1 crossover is therefore "
            "somewhere between N = 350 and N = 700 (the first report's \"N = 250 / 350\" came from draw 0 alone; N = 250 was not re-drawn).",
            "- **The habit rule (item 3) is confirmed as not learned:** pooled L = −0.12 ± 0.32 (N = 350), −0.19 ± 0.17 (700), −0.23 ± 0.06 (1400) "
            "against 0.38 / 0.12 / 0.76 for continuous labels; the pooled Spearman is lower at every N (0.66 / 0.81 / 0.86 vs 0.85 / 0.89 / 0.96).",
            "- **DPO's flip-rate advantage (item 5) was too pessimistic for N = 700 / 1400.** TACT flips less than argmax in 13 of 15 runs at "
            "N = 350 (10.1% vs 23.1%), 14 of 15 at N = 700 (8.4% vs 13.8%) and 11 of 15 at N = 1400 (11.6% vs 15.2%); the paired difference "
            "(A − B) is −0.130, −0.055 and −0.036, against −0.152, −0.166 and −0.107 with continuous labels. The advantage persists in direction "
            "at every N but shrinks relative to continuous labels as N grows; the \"essentially gone at N = 700, reversed at N = 1400\" reading of "
            "the first report was specific to draw 0 (its A − B at N = 1400 is +0.038, against −0.072 in both other draws). Item 5's statement "
            "that TACT is not better than argmax in regret on the paraphrases was not re-checked on the other draws.\n"]
    return out


def strip_v1_narrative(text):
    """With TACT_V2=1 the hand-written conclusions of the original report (numbers of the DPO-era model) must not be repeated next to new tables."""
    i, j = text.find("## Summary of findings"), text.find("## Setup")
    if 0 <= i < j:
        text = text[:i] + ("> **v2 (final architecture, threshold rule).** The policy is the threshold rule on the reward model (ask iff r_hat(ask) − r_hat(act) > 0): no SFT, no DPO, no "
                           "tie rule. The hand-written summary of the original report is omitted here; sentences of the Setup section about SFT/ties are legacy text. Numbers only.\n\n") + text[j:]
    k = text.find("### Corrections to the summary")
    return text[:k] if k >= 0 else text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--labels", default="cont,bin")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--draws", action="store_true", help="re-train on independent Bernoulli label draws (writes the draws csv, then the report)")
    ap.add_argument("--only-n", default="", help="debug: comma-separated N values to run")
    ap.add_argument("--out", default=RUNS_CSV)
    args = ap.parse_args()
    if args.draws:
        train_draws([int(s) for s in args.seeds.split(",")], DRAWS_CSV)
    elif not args.report_only:
        plan = [p for p in PLAN if not args.only_n or str(p[1]) in args.only_n.split(",")]
        df = train_and_eval(args.labels.split(","), plan, [int(s) for s in args.seeds.split(",")], args.out)
        if args.out != RUNS_CSV:
            return
    df = pd.read_csv(RUNS_CSV)
    figure(df, "binary_feedback.png")
    out = report(df)
    # integrity: the continuous N <= 350 runs must reproduce budget_runs.csv (seed-by-seed)
    br = pd.read_csv(os.path.join(RESULTS_DIR, "budget_runs.csv"))
    br = br[br.test == "seen"].pivot_table(index=["n", "seed"], columns="method", values="regret")
    mine = df[(df.label == "cont") & (df["size"] == 500)].set_index(["n", "seed"])
    diff = max(float((mine[f"{m}_seen"] - br[m].reindex(mine.index)).abs().max()) for m in ("A", "B", "C"))
    tie = df[(df.label == "bin") & (df["size"] == 500) & (df.n == 350)].tie_frac.mean()
    if os.path.exists(DRAWS_CSV):
        out += draws_section(df, pd.read_csv(DRAWS_CSV))
    text = "\n".join(out).replace("{CHECK}", f"{diff:.1e}").replace("{TIE}", f"{tie:.0%}")
    if V2:
        text = strip_v1_narrative(text)
    with open(os.path.join(RESULTS_DIR, "binary_feedback.md"), "w") as f:
        f.write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
