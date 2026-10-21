"""Transfer table: key findings with the Context MLP (BertScorer) vs. the one-hot encoding, side by side.

    python -m tact.reporting.transfer     # after: ARCH=onehot python -m tact.decision.budget / hard_paraphrase.py / utility.py / heatmap_arch.py
                           # -> results/onehot_as_main/transfer_table.md

Decision rule, fixed BEFORE the one-hot results were looked at:
  * direction : every row states a qualitative claim (listed in the table); it must hold under both encodings;
  * magnitude : |a - b| <= max(0.30 * max(|a|, |b|), sd_a, sd_b), sd = seed-to-seed standard deviation of the value;
                for positions on the N grid {25, 50, 75, 100, 150, 250, 350} (and day counts) the two must be within one grid step;
  * verdict   : direction holds in both AND magnitude similar -> "transfers"; direction only -> "direction only";
                otherwise -> "does not transfer".
Values for BertScorer come from results/, for OneHotScorer from results/onehot_as_main/ (budget family, same seeds, GPU) and, for
the warm-start stability and the heatmap, from the replay.py runs of both encodings (CPU, 350-row checkpoints).
"""
import json
import os

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.model import ROOT

RB, RO = os.path.join(ROOT, "results"), os.path.join(ROOT, "results", "onehot_as_main")
GRID = B.NS


def load(d):
    r = lambda f: pd.read_csv(os.path.join(d, f))
    return {"bud": r("budget_runs.csv"), "hard": r("hard_paraphrase_runs.csv"), "flips": r("hard_paraphrase_flips.csv"),
            "semi": r("dpo_semionline.csv"), "heat": json.load(open(os.path.join(d, "heatmap_stats.json")))}


D = {"bert": load(RB), "onehot": load(RO)}
REPLAY = {"bert": pd.read_csv(os.path.join(RB, "replay_metrics.csv")), "onehot": pd.read_csv(os.path.join(RB, "replay_metrics_onehot.csv"))}
bl = B.baselines(B.scenarios(B.load_long("test_seen.csv")))
BASE = min(bl["always_ask"]["regret"], bl["never_ask"]["regret"])
U_NEVER = bl["never_ask"]["acc"] / (1 + (1 - bl["never_ask"]["act_rate"]))


def per_seed(df, method, test, n, col):
    return df[(df.method == method) & (df.test == test) & (df.n == n)].sort_values("seed")[col].values


def U(df):
    return df.acc / (1.0 + (1.0 - df.act_rate))


def idx(n):
    return None if n is None else GRID.index(n)


def cross_point(df, ts):
    """First N from which A's mean regret is <= C's for every larger N on the grid."""
    a = df[(df.method == "A") & (df.test == ts)].groupby("n").regret.mean()
    c = df[(df.method == "C") & (df.test == ts)].groupby("n").regret.mean()
    ok = [a[n] <= c[n] for n in GRID]
    for i in range(len(GRID)):
        if all(ok[i:]):
            return GRID[i]
    return None


ROWS = []


def add(group, finding, claim, vb, vo, dir_ok, sdb=0.0, sdo=0.0, pos=False, fmt="{:.3f}"):
    if pos:
        if vb is None or vo is None:
            mag = False
        elif vb in GRID and vo in GRID:
            mag = abs(idx(vb) - idx(vo)) <= 1          # positions on the N grid: within one step
        else:
            mag = abs(vb - vo) <= 1                    # counts (days)
        f = lambda v: "not reached" if v is None else str(v)
    else:
        mag = abs(vb - vo) <= max(0.30 * max(abs(vb), abs(vo)), sdb, sdo)
        f = lambda v: fmt.format(v)
    verdict = "transfers" if (dir_ok and mag) else ("direction only" if dir_ok else "does not transfer")
    sd = lambda v, s: f(v) + (f" ± {s:.3f}" if s else "")
    ROWS.append({"group": group, "finding": finding, "claim": claim, "bert": sd(vb, sdb), "onehot": sd(vo, sdo),
                 "direction": "same" if dir_ok else "differs", "magnitude": "similar" if mag else "differs", "verdict": verdict})


def both(fn):
    return {a: fn(D[a]) for a in D}


# ---------------------------------------------------------------- 1. crossing curves
for ts in ("seen", "unseen"):
    cp = both(lambda d: cross_point(d["bud"], ts))
    add("1 curves", f"A overtakes C for good (test_{ts}): first N from which A ≤ C at every larger N",
        "A overtakes C within the grid", cp["bert"], cp["onehot"], cp["bert"] is not None and cp["onehot"] is not None, pos=True)
for ts in ("seen",):
    h = both(lambda d: B.first_n(d["bud"], "A", ts, BASE, "half"))
    add("1 curves", "N at which A's mean regret ≤ half of min(always-ask, never-ask)", "reached within N ≤ 350 in both",
        h["bert"], h["onehot"], h["bert"] is not None and h["onehot"] is not None, pos=True)
    s1 = both(lambda d: B.first_n(d["bud"], "A", ts, BASE, "sd"))
    add("1 curves", "N at which A's mean + 1 std ≤ min(always-ask, never-ask)", "reached within N ≤ 350 in both",
        s1["bert"], s1["onehot"], s1["bert"] is not None and s1["onehot"] is not None, pos=True)
gap = {a: per_seed(D[a]["bud"], "A", "seen", 50, "regret") - per_seed(D[a]["bud"], "C", "seen", 50, "regret") for a in D}
gap25 = {a: per_seed(D[a]["bud"], "A", "seen", 25, "regret") - per_seed(D[a]["bud"], "C", "seen", 25, "regret") for a in D}
add("1 curves", "A − C regret at N = 50 (test_seen)", "C is better than A at N = 25 and N = 50 in both (A − C > 0)",
    gap["bert"].mean(), gap["onehot"].mean(), all(gap[a].mean() > 0 and gap25[a].mean() > 0 for a in D),
    gap["bert"].std(ddof=1), gap["onehot"].std(ddof=1), fmt="{:+.3f}")
a350 = {a: per_seed(D[a]["bud"], "A", "seen", 350, "regret") for a in D}
c350 = {a: per_seed(D[a]["bud"], "C", "seen", 350, "regret") for a in D}
add("1 curves", "A regret at N = 350 (test_seen)", "A beats the tabular agent C at N = 350 in both",
    a350["bert"].mean(), a350["onehot"].mean(), all(a350[a].mean() < c350[a].mean() for a in D),
    a350["bert"].std(ddof=1), a350["onehot"].std(ddof=1))
du = {a: per_seed(D[a]["bud"], "A", "unseen", 350, "regret") - a350[a] for a in D}
add("1 curves", "A regret on unseen phrasing minus seen, N = 350", "A does not drop meaningfully (|change| < 0.02) in both",
    du["bert"].mean(), du["onehot"].mean(), all(abs(du[a].mean()) < 0.02 for a in D), du["bert"].std(ddof=1), du["onehot"].std(ddof=1), fmt="{:+.3f}")
so = {a: D[a]["semi"][(D[a]["semi"].method == "A") & (D[a]["semi"].test == "seen") & (D[a]["semi"]["round"] == 7)].sort_values("seed").regret.values for a in D}
soc = {a: D[a]["semi"][(D[a]["semi"].method == "C") & (D[a]["semi"].test == "seen") & (D[a]["semi"]["round"] == 7)].regret.mean() for a in D}
add("1 curves", "semi-online, round 7: A regret (test_seen)", "A ends below the tabular agent C in both",
    so["bert"].mean(), so["onehot"].mean(), all(so[a].mean() < soc[a] for a in D), so["bert"].std(ddof=1), so["onehot"].std(ddof=1))

# ---------------------------------------------------------------- 2. DPO vs argmax
dis = {}
for a in D:
    g = D[a]["bud"]
    g = g[(g.method == "A_vs_B") & (g.test == "seen") & (g.n >= 100)].groupby("seed").acc.mean()
    dis[a] = 1 - g.values
add("2 DPO vs argmax", "A–B decision disagreement, mean over N ≥ 100 (test_seen)", "A and B differ on at least 3% of scenarios in both",
    dis["bert"].mean(), dis["onehot"].mean(), all(dis[a].mean() >= 0.03 for a in D), dis["bert"].std(ddof=1), dis["onehot"].std(ddof=1))
ab = {a: per_seed(D[a]["bud"], "A", "seen", 350, "regret") - per_seed(D[a]["bud"], "B", "seen", 350, "regret") for a in D}
add("2 DPO vs argmax", "regret A − B at N = 350 (test_seen)", "A and B within noise (|mean| ≤ std) in both",
    ab["bert"].mean(), ab["onehot"].mean(), all(abs(ab[a].mean()) <= ab[a].std(ddof=1) for a in D),
    ab["bert"].std(ddof=1), ab["onehot"].std(ddof=1), fmt="{:+.3f}")
fl = {a: {m: D[a]["flips"][(D[a]["flips"].n == 350) & (D[a]["flips"].method == m)].sort_values("seed").flip_rate.values for m in ("A", "B")} for a in D}
okflip = all(fl[a]["B"].mean() > fl[a]["A"].mean() for a in D)
for m in ("A", "B"):
    add("2 DPO vs argmax", f"hard paraphrases, N = 350: share of decisions changed seen → hard, {m}",
        "B changes more decisions than A in both", fl["bert"][m].mean(), fl["onehot"][m].mean(), okflip,
        fl["bert"][m].std(ddof=1), fl["onehot"][m].std(ddof=1))
hr = {a: {m: np.array([D[a]["hard"][(D[a]["hard"].n == 350) & (D[a]["hard"].method == m) & (D[a]["hard"].test.isin(["hard1", "hard2"])) &
                                   (D[a]["hard"].seed == s)].regret.mean() for s in B.SEEDS]) - per_seed(D[a]["hard"], m, "seen", 350, "regret")
              for m in ("A", "B")} for a in D}
okh = all(hr[a]["B"].mean() > hr[a]["A"].mean() for a in D)
for m in ("A", "B"):
    add("2 DPO vs argmax", f"hard paraphrases, N = 350: regret change hard − seen, {m}", "B degrades more than A in both",
        hr["bert"][m].mean(), hr["onehot"][m].mean(), okh, hr["bert"][m].std(ddof=1), hr["onehot"][m].std(ddof=1), fmt="{:+.3f}")

# ---------------------------------------------------------------- 3. warm-start stability (replay, rounds 2-7)
names = {"drift": "prediction drift", "first_batch_mse": "first-batch MSE", "kendall_tau": "Kendall τ"}
better = {"drift": -1, "first_batch_mse": -1, "kendall_tau": +1}
for col in ("drift", "first_batch_mse", "kendall_tau"):
    d = {}
    for a in D:
        g = REPLAY[a][REPLAY[a]["round"] >= 2].groupby(["strategy", "seed"])[col].mean().unstack("strategy")
        d[a] = (g["warm"] - g["scratch"]).values
    add("3 warm-start stability", f"warm-start − from-scratch, {names[col]} (mean over rounds 2–7)", "warm-start is the better one in both",
        d["bert"].mean(), d["onehot"].mean(), all(np.sign(d[a].mean()) == better[col] for a in D), d["bert"].std(ddof=1), d["onehot"].std(ddof=1),
        fmt="{:+.4f}")

# ---------------------------------------------------------------- 4. clarification utility
ua = {a: U(D[a]["bud"][(D[a]["bud"].method == "A") & (D[a]["bud"].test == "seen") & (D[a]["bud"].n == 350)].sort_values("seed")).values for a in D}
uc = {a: U(D[a]["bud"][(D[a]["bud"].method == "C") & (D[a]["bud"].test == "seen") & (D[a]["bud"].n == 350)]).mean() for a in D}
add("4 clarification utility", "U of A at N = 350 (test_seen)", f"A beats C and the never-ask level {U_NEVER:.2f} in both",
    ua["bert"].mean(), ua["onehot"].mean(), all(ua[a].mean() > max(uc[a], U_NEVER) for a in D), ua["bert"].std(ddof=1), ua["onehot"].std(ddof=1))
uh = {}
for a in D:
    h = D[a]["hard"]
    ua_h = U(h[(h.n == 350) & (h.method == "A") & (h.test.isin(["hard1", "hard2"]))]).groupby(h[(h.n == 350) & (h.method == "A") & (h.test.isin(["hard1", "hard2"]))].seed).mean()
    ub_h = U(h[(h.n == 350) & (h.method == "B") & (h.test.isin(["hard1", "hard2"]))]).groupby(h[(h.n == 350) & (h.method == "B") & (h.test.isin(["hard1", "hard2"]))].seed).mean()
    uh[a] = (ua_h.values - ub_h.values)
add("4 clarification utility", "U(A) − U(B) on the hard paraphrases, N = 350", "A has the higher U on the hard set in both",
    uh["bert"].mean(), uh["onehot"].mean(), all(uh[a].mean() > 0 for a in D), uh["bert"].std(ddof=1), uh["onehot"].std(ddof=1), fmt="{:+.3f}")


def first_u(df):
    g = df[(df.method == "A") & (df.test == "seen")].copy()
    g["U"] = U(g)
    m = g.groupby("n").U.mean()
    for n in GRID:
        if m[n] > U_NEVER:
            return n
    return None


fu = both(lambda d: first_u(d["bud"]))
add("4 clarification utility", f"smallest N at which A's mean U exceeds the never-ask level {U_NEVER:.2f}", "reached within the grid in both",
    fu["bert"], fu["onehot"], fu["bert"] is not None and fu["onehot"] is not None, pos=True)

# ---------------------------------------------------------------- 5. Saturday-fruit collapse (heatmap, final from-scratch models)
H = {a: D[a]["heat"] for a in D}
tr = H["bert"]["truth"]["sat_over_other"]
ps = {a: np.array([H[a]["per_seed"][str(s)]["sat_over_other"] for s in range(5)]) for a in D}
add("5 Saturday collapse", "Saturday / other-day mean of the fruit heatmap (mean of 5 seeds' grid; truth " + f"{tr:.2f})",
    "collapse partly learned: truth < ratio < 1 in both", H["bert"]["mean_of_seeds"]["sat_over_other"], H["onehot"]["mean_of_seeds"]["sat_over_other"],
    all(tr < H[a]["mean_of_seeds"]["sat_over_other"] < 1 for a in D), ps["bert"].std(ddof=1), ps["onehot"].std(ddof=1))
fe = {}
for a in D:
    g, t = np.array(H[a]["pred_mean_grid"]), None
    fe[a] = float(g[5, 5] / np.delete(g[5], 5).mean())     # fearful row (index 5), Saturday column (index 5)
add("5 Saturday collapse", "fearful row: Saturday / other-day ratio", "the fearful cell is not collapsed (ratio > 0.5) in both",
    fe["bert"], fe["onehot"], all(fe[a] > 0.5 for a in D))
add("5 Saturday collapse", "mean Spearman between valence and prediction over the 7 days", "emotion direction mostly right (≥ 0.8) in both",
    H["bert"]["mean_of_seeds"]["mean_spearman_valence_vs_pred"], H["onehot"]["mean_of_seeds"]["mean_spearman_valence_vs_pred"],
    all(H[a]["mean_of_seeds"]["mean_spearman_valence_vs_pred"] >= 0.8 for a in D))
mono = {a: H[a]["mean_of_seeds"]["days_fully_monotonic"] for a in D}
add("5 Saturday collapse", "days (of 7) with a fully monotonic emotion order", "not fully monotonic (≤ 4 of 7 days) in both",
    mono["bert"], mono["onehot"], all(mono[a] <= 4 for a in D), pos=True)

# ---------------------------------------------------------------- report
T = pd.DataFrame(ROWS)
T.to_csv(os.path.join(RO, "transfer_table.csv"), index=False)
cnt = T.verdict.value_counts().to_dict()
# integrity of C / Ct (encoding-independent)
b, o = D["bert"]["bud"], D["onehot"]["bud"]
key = ["n", "seed", "method", "test"]
m = b[b.method.isin(["C", "Ct"])].merge(o[o.method.isin(["C", "Ct"])], on=key, suffixes=("_b", "_o"))
cdiff = float((m.regret_b - m.regret_o).abs().max())
out = ["# Transfer table: Context MLP (BertScorer) vs. one-hot encoding as the main reward/policy model\n",
       "Every experiment of the budget family was re-run with `OneHotScorer` as the reward model and, without its final sigmoid, as the "
       "policy (5 seeds, all other settings unchanged: `ARCH=onehot python -m tact.decision.budget`, `hard_paraphrase.py`, `utility.py`, "
       "`heatmap_arch.py`; outputs in this directory). The tabular agents C and Ct do not depend on the encoding, and "
       f"they reproduce the BertScorer runs exactly (max abs regret difference {cdiff:.1e}). Warm-start stability and the heatmap use "
       "the replay.py runs of both encodings (350-row checkpoints, CPU).\n",
       "## Decision rule (fixed before looking at the one-hot results)\n",
       "- **Direction:** each row states a qualitative claim; it must hold under both encodings.\n"
       "- **Magnitude:** |a − b| ≤ max(0.30 · max(|a|, |b|), sd_a, sd_b), where sd is the seed-to-seed std of the value; "
       "for positions on the N grid {25, 50, 75, 100, 150, 250, 350} and for day counts the two must be within one grid step.\n"
       "- **Verdict:** direction holds in both and magnitude is similar → *transfers*; direction only → *direction only*; "
       "otherwise → *does not transfer*.\n",
       f"## Result: {cnt.get('transfers', 0)} of {len(T)} rows transfer, {cnt.get('direction only', 0)} transfer in direction only, "
       f"{cnt.get('does not transfer', 0)} do not\n",
       "| group | finding | claim that must hold in both | BertScorer | OneHotScorer | direction | magnitude | verdict |",
       "|---|---|---|---|---|---|---|---|"]
for r in ROWS:
    out.append(f"| {r['group']} | {r['finding']} | {r['claim']} | {r['bert']} | {r['onehot']} | {r['direction']} | {r['magnitude']} | **{r['verdict']}** |")
out += ["\n## By group\n", "| group | transfers | direction only | does not transfer |", "|---|---|---|---|"]
for g, x in T.groupby("group"):
    c = x.verdict.value_counts()
    out.append(f"| {g} | {c.get('transfers', 0)} | {c.get('direction only', 0)} | {c.get('does not transfer', 0)} |")
with open(os.path.join(RO, "transfer_table.md"), "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
