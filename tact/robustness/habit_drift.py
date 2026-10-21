"""Habit drift: the Saturday-fruit habit stops being true after round 3 (results/habit_drift.md, results/habit_drift.png).

    python -m tact.robustness.habit_drift

Seven rounds of 50 training scenarios (the first 350 scenarios of data_dpo, file order). Rounds 1-3 carry the original labels. From round 4 on
the Saturday-fruit rule no longer holds: for the scenarios of rounds 4-7 with topic = fruit and day = Saturday the necessity is the base 0.80
(not 0.10), i.e. ask value 0.8 x receptiveness and act value 0.2 instead of 0.9; everything else (scenarios, the other habit rules, valence,
availability, noise level) is unchanged. The validation scenarios used for early stopping follow the same rule change (original labels up to
round 3, drifted labels from round 4).
Methods, evaluated after every round on the 1512-cell grid under the OLD and under the NEW truth:
  W  Ours with warm start (as in budget.py's semi-online loop: the reward model continues, the DPO policy continues) on ALL data so far;
  S  control 1: trained FROM SCRATCH, using only the most recent three rounds of data (150 scenarios);
  C  control 2: the tabular agent (constant learning rate 0.5), updated with each new batch.
Decision metric: regret against the oracle of the respective truth (A = policy, B = argmax of the reward model); drift metric: the
Saturday-fruit collapse ratio (mean predicted ask score of the Saturday column over the mean of the other six days of the afternoon fruit heatmap,
as in plots.py) of the reward model (for C: of its Q / fallback values).
Criterion fixed beforehand: W "adapts in time" if its collapse ratio at round 5 (two drifted rounds) exceeds 0.74, the midpoint between
the pre-drift 0.47 and 1.0.
"""
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.generate_dataset import QUESTIONS, availability
from tact.reward.grid_eval import TOPICS
from tact.core.model import CONFIG, EMOTIONS, RESULTS_DIR, ROOT, FrozenBert, INPUT_COLS, get_device, make_tensors
from tact.core.plots import is_habit, necessity

DRIFT_FROM = 3      # rounds 1..3 original, 4..7 drifted
NEW_NEC = CONFIG["base_necessity"]["fruit"]
SRC = os.path.join(ROOT, "data_dpo")
DST = os.path.join(ROOT, "data_drift")
ROUNDS = range(1, 8)


# ---------------------------------------------------------------- data
def relabel(long_df, first_scenario, seed):
    """Rows of the scenarios at position >= first_scenario (ask rows in file order) with topic fruit and day Saturday get the new-rule labels."""
    df = long_df.copy()
    ask = df[df.cand == "ask"].reset_index()
    pos = {i: p for p, i in enumerate(ask.id)}
    rng = np.random.default_rng(seed)
    n_changed = 0
    for _, r in ask.iterrows():
        if pos[r.id] < first_scenario or r.topic != "fruit" or r.day != 5:
            continue
        t = CONFIG["valence"][r.emotion] * availability(r.day, r.time)
        ca, cc = NEW_NEC * t, 1.0 - NEW_NEC
        m_ask = (df.id == r.id) & (df.cand == "ask")
        m_act = (df.id == r.id) & (df.cand == "act")
        df.loc[m_ask, ["score_clean", "score"]] = [round(ca, 3), round(float(np.clip(ca + rng.normal(0, 0.05), 0, 1)), 3)]
        df.loc[m_act, ["score_clean", "score"]] = [round(cc, 3), round(float(np.clip(cc + rng.normal(0, 0.05), 0, 1)), 3)]
        df.loc[(df.id == r.id), "necessity"] = NEW_NEC
        df.loc[(df.id == r.id), "habit_hit"] = np.nan
        n_changed += 1
    return df, n_changed


OLD_DESC = [h["desc"] for h in CONFIG["habits"] if h["topic"] == "fruit"][0]
DENSE_PER_BATCH, DENSE_VAL = 8, 6      # supplementary variant: 8 of every 50 training scenarios (6 of 75 validation) become fruit-Saturday scenarios


def inject(long_df, positions, specs, new_rule):
    """Overwrite the scenarios at the given positions (ask rows in file order) by fruit-Saturday scenarios (spec = emotion, time, phrasing, 4 noise draws)."""
    df = long_df.copy()
    ids = df[df.cand == "ask"].id.tolist()
    nec = NEW_NEC if new_rule else 0.1
    for pos, (emo, tm, ph, nz) in zip(positions, specs):
        i = ids[pos]
        t = CONFIG["valence"][emo] * availability(5, tm)
        ca, cc = nec * t, 1.0 - nec
        for cand, text, clean, noise in (("ask", QUESTIONS["fruit"][ph - 1], ca, nz[0]), ("act", B.ACT_TEXT, cc, nz[1])):
            m = (df.id == i) & (df.cand == cand)
            df.loc[m, ["question", "topic", "emotion", "day", "day_name", "time", "time_name", "necessity", "receptiveness",
                       "score_clean", "score", "habit_hit"]] = [text, "fruit", emo, 5, "Sat", tm, CONFIG["times"][tm], nec, round(t, 3),
                                                                round(clean, 3), round(float(np.clip(clean + noise, 0, 1)), 3),
                                                                np.nan if new_rule else OLD_DESC]
            df.loc[m, "phrasing"] = ph if cand == "ask" else 0
    return df


def dense_specs(rng, n_batches, k, per_batch=50):
    positions, specs = [], []
    for b in range(n_batches):
        pos = rng.choice(per_batch, k, replace=False) + b * per_batch
        positions += list(pos)
        specs += [(EMOTIONS[rng.integers(6)], int(rng.integers(3)), int(rng.integers(1, 3)), rng.normal(0, 0.05, 2)) for _ in range(k)]
    return positions, specs


def build_dirs(dense=False):
    """Writes data_drift[_dense]/{old_val,new_val}/ (same training file; validation with the old / the new labels).
    Returns the number of fruit-Saturday scenarios per training batch and the numbers of relabelled scenarios."""
    global DST
    DST = os.path.join(ROOT, "data_drift_dense" if dense else "data_drift")
    tr, va = pd.read_csv(os.path.join(SRC, "train.csv")), pd.read_csv(os.path.join(SRC, "val.csv"))
    if dense:
        rng = np.random.default_rng(777)
        p_tr, s_tr = dense_specs(rng, 7, DENSE_PER_BATCH)
        p_va, s_va = dense_specs(rng, 1, DENSE_VAL, per_batch=75)
        early = [k for k, p in enumerate(p_tr) if p < 50 * DRIFT_FROM]
        late = [k for k, p in enumerate(p_tr) if p >= 50 * DRIFT_FROM]
        tr = inject(tr, [p_tr[k] for k in early], [s_tr[k] for k in early], False)     # batches 1-3: old rule
        tr = inject(tr, [p_tr[k] for k in late], [s_tr[k] for k in late], True)        # batches 4-7: new rule
        va_old = inject(va, p_va, s_va, False)
        va_new = inject(va, p_va, s_va, True)
        tr_new, n_tr = relabel(tr, 50 * DRIFT_FROM, 4242)       # the original fruit-Saturday scenarios of batches 4-7 follow the new rule
        va_new, n_va = relabel(va_new, 0, 4243)
        va = va_old
    else:
        tr_new, n_tr = relabel(tr, 50 * DRIFT_FROM, 4242)
        va_new, n_va = relabel(va, 0, 4243)
    for sub, val in (("old_val", va), ("new_val", va_new)):
        d = os.path.join(DST, sub)
        os.makedirs(d, exist_ok=True)
        tr_new.to_csv(os.path.join(d, "train.csv"), index=False)
        val.to_csv(os.path.join(d, "val.csv"), index=False)
        for f in ("test_seen.csv", "test_unseen.csv"):
            shutil.copy(os.path.join(SRC, f), os.path.join(d, f))
    ask_tr = tr_new[tr_new.cand == "ask"].reset_index(drop=True)
    fs = ask_tr[(ask_tr.topic == "fruit") & (ask_tr.day == 5)]
    per_batch = fs.groupby(fs.index // 50 + 1).size().reindex(range(1, 8), fill_value=0)
    return per_batch, n_tr, n_va


def truth_grid(new_rule):
    rows = []
    for t in TOPICS:
        for e in EMOTIONS:
            for d in range(7):
                for m in range(3):
                    rows.append({"topic": t, "question": QUESTIONS[t][0], "emotion": e, "day": d, "time": m})
    g = pd.DataFrame(rows)
    nec = np.array([NEW_NEC if (new_rule and t == "fruit" and d == 5) else necessity(t, d, m) for t, d, m in zip(g.topic, g.day, g.time)])
    tt = np.array([CONFIG["valence"][e] * availability(d, m) for e, d, m in zip(g.emotion, g.day, g.time)])
    g["c_ask"], g["c_act"] = nec * tt, 1.0 - nec
    g["score"] = 0.0
    return g


def regret(ask, ca, cc, mask=None):
    best = np.maximum(ca, cc)
    r = best - np.where(ask, ca, cc)
    return float(r.mean() if mask is None else r[mask].mean())


def collapse_ratio(pred, g):
    sel = (g.topic == "fruit").values & (g.time == 1).values
    arr = pred[sel].reshape(len(EMOTIONS), 7)               # rows: emotions, columns: days (grid order: emotion, day)
    return float(arr[:, 5].mean() / np.delete(arr, 5, axis=1).mean())


def tab_ask_values(tab, g):
    out = []
    for r in g.itertuples():
        k = (r.topic, r.emotion, r.day, r.time, "ask")
        if k in tab.Q:
            out.append(tab.Q[k])
        elif (r.topic, "ask") in tab.cnt:
            out.append(tab.sum[(r.topic, "ask")] / tab.cnt[(r.topic, "ask")])
        else:
            out.append(tab.gsum["ask"] / max(tab.gcnt["ask"], 1))
    return np.array(out)


def window_prefix(data, a, b):
    def prefix(n):
        sc = data.sc_train.iloc[a:b]
        rows = data.train_long[data.train_long.id.isin(set(sc.id))]
        return sc, make_tensors(rows[INPUT_COLS + data.loc_cols], data.enc, data.dev), B.feats(sc, data.enc, data.dev)
    return prefix


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def run_variant(dense, enc, dev):
    tag = "_dense" if dense else ""
    per_batch, n_tr, n_va = build_dirs(dense)
    D = {(p, v): B.Data(enc, dev, data_dir=os.path.join(DST, f"{v}_val")) for p in ("W", "S") for v in ("old", "new")}
    g_old, g_new = truth_grid(False), truth_grid(True)
    f_grid = B.feats(g_old, enc, dev)
    fs = ((g_old.topic == "fruit") & (g_old.day == 5)).values
    base = {k: min(regret(np.ones(len(g), bool), g.c_ask.values, g.c_act.values), regret(np.zeros(len(g), bool), g.c_ask.values, g.c_act.values))
            for k, g in (("old", g_old), ("new", g_new))}
    truth_ratio = {k: collapse_ratio(g.c_ask.values, g) for k, g in (("old", g_old), ("new", g_new))}
    rows = []

    def record(rnd, seed, proto, a, b, c_dec, ratio):
        for rule, g in (("old", g_old), ("new", g_new)):
            ca, cc = g.c_ask.values, g.c_act.values
            for m, d_ in (("A", a), ("B", b), ("C", c_dec)):
                if d_ is None:
                    continue
                rows.append({"round": rnd, "seed": seed, "proto": proto, "method": m, "rule": rule,
                             "grid_nr": regret(d_, ca, cc) / base[rule], "fs_regret": regret(d_, ca, cc, fs),
                             "fs_act_rate": float((~d_[fs]).mean()), "ratio": ratio})

    for seed in B.SEEDS:
        ours_w = B.Ours(D[("W", "old")], seed)
        tab = B.Tabular("topic")
        for rnd in ROUNDS:
            key = "old" if rnd <= DRIFT_FROM else "new"
            s = seed * 1000 + 900 + rnd
            # --- W: warm start on all data so far
            ours_w.d = D[("W", key)]
            ours_w.fit_round(50 * rnd, s, warm=rnd > 1)
            a, b = ours_w.decide(f_grid)
            record(rnd, seed, "W", a, b, None, collapse_ratio(ours_w.rm.predict(g_old), g_old))
            # --- S: from scratch on the last three rounds
            ds = D[("S", key)]
            lo = 50 * max(0, rnd - 3)
            ds.prefix = window_prefix(ds, lo, 50 * rnd)
            ours_s = B.Ours(ds, seed)
            ours_s.fit_round(50 * rnd - lo, s, warm=False)
            a, b = ours_s.decide(f_grid)
            record(rnd, seed, "S", a, b, None, collapse_ratio(ours_s.rm.predict(g_old), g_old))
            # --- C: tabular, one batch per round
            batch = D[("W", key)].sc_train.iloc[50 * (rnd - 1):50 * rnd].reset_index(drop=True)
            tab.update(batch, np.random.default_rng([seed, rnd, 7]).permutation(50))
            c_dec, _ = tab.decide(g_old)
            record(rnd, seed, "C", None, None, c_dec, collapse_ratio(tab_ask_values(tab, g_old), g_old))
        print(f"[drift] seed {seed} done", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(RESULTS_DIR, f"habit_drift{tag}_runs.csv"), index=False)

    def get(proto, method, rule, col, rnd):
        return R[(R.proto == proto) & (R.method == method) & (R.rule == rule) & (R["round"] == rnd)].sort_values("seed")[col].values

    ratio = {p: {r: get(p, "A" if p != "C" else "C", "new", "ratio", r) for r in ROUNDS} for p in ("W", "S", "C")}
    first = {}
    for p in ("W", "S", "C"):
        first[p] = next((r for r in ROUNDS if r > DRIFT_FROM and ratio[p][r].mean() >= 0.74), None)
    w5 = ratio["W"][5]
    on_time = w5.mean() > 0.74

    # ------------------------------------------------ figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    COL = {"W": "#2a78d6", "S": "#eb6834", "C": "#1baf7a"}
    LAB = {"W": "Ours, warm start on all data", "S": "Ours, from scratch on the last 3 rounds", "C": "tabular C (lr 0.5)"}
    ax = axes[0]
    for p in ("W", "S", "C"):
        m = np.array([ratio[p][r].mean() for r in ROUNDS])
        sd = np.array([ratio[p][r].std(ddof=1) for r in ROUNDS])
        ax.plot(list(ROUNDS), m, "-o", color=COL[p], lw=2, label=LAB[p])
        ax.fill_between(list(ROUNDS), m - sd, m + sd, color=COL[p], alpha=0.14)
    ax.axhline(truth_ratio["old"], color="gray", ls=":", lw=1)
    ax.axhline(truth_ratio["new"], color="gray", ls=":", lw=1)
    ax.axhline(0.74, color="k", ls="--", lw=1)
    ax.text(1.05, truth_ratio["old"] + 0.02, f"old rule: {truth_ratio['old']:.2f}", fontsize=9)
    ax.text(1.05, truth_ratio["new"] + 0.02, f"new rule: {truth_ratio['new']:.2f}", fontsize=9)
    ax.text(1.05, 0.76, "criterion 0.74", fontsize=9, va="bottom")
    ax.set_ylabel("Saturday / other-day ratio (fruit, afternoon)")
    ax.set_title("Collapse ratio of the learned preference", fontsize=10, loc="left")
    for ax, (col, ttl) in zip(axes[1:], (("new", "Regret on the 18 cells, NEW rule (adaptation)"),
                                         ("old", "Regret on the 18 cells, OLD rule (forgetting)"))):
        for p, meth in (("W", "A"), ("S", "A"), ("C", "C")):
            v = np.array([get(p, meth, col, "fs_regret", r) for r in ROUNDS])
            ax.plot(list(ROUNDS), v.mean(1), "-o", color=COL[p], lw=2)
            ax.fill_between(list(ROUNDS), v.mean(1) - v.std(1, ddof=1), v.mean(1) + v.std(1, ddof=1), color=COL[p], alpha=0.14)
        ax.set_title(ttl, fontsize=9.5, loc="left")
        ax.set_ylabel("mean regret on those cells")
    for ax in axes:
        ax.axvline(DRIFT_FROM + 0.5, color="#c44", ls="-.", lw=1)
        ax.set_xlabel("round (50 scenarios each)")
        ax.set_xticks(list(ROUNDS))
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].text(DRIFT_FROM + 0.55, axes[0].get_ylim()[1] * 0.02 + axes[0].get_ylim()[0], "drift", color="#c44", fontsize=9)
    fig.suptitle("Habit drift: the Saturday-fruit rule stops holding after round 3 (mean ± std over 5 seeds)"
                 + (" — dense-evidence variant" if dense else ""), x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, f"habit_drift{tag}.png"), dpi=170)
    plt.close(fig)

    return {"R": R, "ratio": ratio, "per_batch": per_batch, "truth_ratio": truth_ratio, "n_tr": n_tr, "n_va": n_va, "first": first,
            "w5": w5, "on_time": on_time, "get": get}


def tables(v, h):
    """Markdown sections of one variant; h = heading prefix."""
    get, ratio, per_batch, tr = v["get"], v["ratio"], v["per_batch"], v["truth_ratio"]
    cum_new = np.cumsum([0 if r <= DRIFT_FROM else per_batch[r] for r in ROUNDS])
    out = [f"{h} How much drift evidence the learner sees\n",
           "Fruit-Saturday scenarios per training batch (all of them follow the original rule in rounds 1–3 and the drifted rule from round 4):\n",
           "| round | fruit-Saturday scenarios in the batch | original-rule observations so far | drifted observations so far (all data) | "
           "drifted observations in the last-3-rounds window |", "|---|---|---|---|---|"]
    for r in ROUNDS:
        old_so_far = int(sum(per_batch[k] for k in range(1, min(r, DRIFT_FROM) + 1)))
        win_new = int(sum(per_batch[k] for k in range(max(1, r - 2), r + 1) if k > DRIFT_FROM))
        win_old = int(sum(per_batch[k] for k in range(max(1, r - 2), r + 1) if k <= DRIFT_FROM))
        out.append(f"| {r} | {per_batch[r]} | {old_so_far} | {int(cum_new[r - 1])} | {win_new} (old in window: {win_old}) |")
    out.append(f"\nRelabelled scenarios: {v['n_tr']} in the training set, {v['n_va']} in the validation set.\n")
    out += [f"{h} 1. Collapse ratio of the Saturday-fruit column (mean ± std over 5 seeds)\n",
            f"Truth: {tr['old']:.2f} under the old rule, {tr['new']:.2f} under the new one; criterion 0.74.\n",
            "| round | W warm start, all data | S from scratch, last 3 rounds | C tabular |", "|---|---|---|---|"]
    for r in ROUNDS:
        out.append(f"| {r}{' (drift starts)' if r == DRIFT_FROM + 1 else ''} | {ms(ratio['W'][r])} | {ms(ratio['S'][r])} | {ms(ratio['C'][r])} |")
    f_ = v["first"]
    out.append(f"\nFirst drifted round with a mean ratio ≥ 0.74: W {f_['W'] or 'never'}, S {f_['S'] or 'never'}, C {f_['C'] or 'never'}.\n")
    for rule, ttl in (("new", f"{h} 2. Adaptation: regret under the NEW rule\n"), ("old", f"{h} 3. Residual of the old habit: regret under the OLD rule\n")):
        out += [ttl,
                "Cells = the 18 fruit-Saturday cells (regret there / share of those cells in which the method acts) and the whole grid (normalised). "
                + ("A good adapter's regret on these cells falls after round 3 (under the new rule the oracle asks in most of them)."
                   if rule == "new" else "Under the old rule the oracle always acts there; a rising regret means the old habit is forgotten."),
                "", "| round | A, W: cells | A, S: cells | C: cells | A, W: grid | A, S: grid | C: grid |", "|---|---|---|---|---|---|---|"]
        for r in ROUNDS:
            cells = [f"{ms(get(p, m, rule, 'fs_regret', r))} / {get(p, m, rule, 'fs_act_rate', r).mean():.2f}" for p, m in (("W", "A"), ("S", "A"), ("C", "C"))]
            gr = [ms(get(p, m, rule, "grid_nr", r)) for p, m in (("W", "A"), ("S", "A"), ("C", "C"))]
            out.append(f"| {r} | " + " | ".join(cells) + " | " + " | ".join(gr) + " |")
        out.append("")
    out += [f"{h} 4. Argmax of the reward model (B) on the same cells\n",
            "| round | W: new rule | W: old rule | S: new rule | S: old rule |", "|---|---|---|---|---|"]
    for r in ROUNDS:
        out.append(f"| {r} | {ms(get('W', 'B', 'new', 'fs_regret', r))} | {ms(get('W', 'B', 'old', 'fs_regret', r))} | "
                   f"{ms(get('S', 'B', 'new', 'fs_regret', r))} | {ms(get('S', 'B', 'old', 'fs_regret', r))} |")
    return out


def verdict(v, name):
    ratio, get = v["ratio"], v["get"]
    pre = ratio["W"][3]
    vac = pre.mean() > 0.74
    w_new = [get("W", "A", "new", "fs_regret", r).mean() for r in (3, 7)]
    s_new = [get("S", "A", "new", "fs_regret", r).mean() for r in (3, 7)]
    c_new = [get("C", "C", "new", "fs_regret", r).mean() for r in (3, 7)]
    w_old = [get("W", "A", "old", "fs_regret", r).mean() for r in (3, 7)]
    s_old = [get("S", "A", "old", "fs_regret", r).mean() for r in (3, 7)]
    return [f"**{name}.** Criterion: W's collapse ratio at round 5 is {ms(v['w5'])} (> 0.74 required) → "
            f"{'formally met' if v['on_time'] else 'not met'}. The ratio at round 3, before the drift, is {ms(pre)}: "
            + ("**it is already above 0.74 before any drift, so the criterion cannot show adaptation here (the old habit had not been learned in the score).** "
               if vac else "the old habit had been learned before the drift (ratio well below 0.74), so the criterion is meaningful. ")
            + f"Decision level, regret on the 18 fruit-Saturday cells under the new rule, round 3 → 7: W {w_new[0]:.3f} → {w_new[1]:.3f}, "
              f"S {s_new[0]:.3f} → {s_new[1]:.3f}, C {c_new[0]:.3f} → {c_new[1]:.3f}; under the old rule (forgetting): W {w_old[0]:.3f} → {w_old[1]:.3f}, "
              f"S {s_old[0]:.3f} → {s_old[1]:.3f}.\n"]


def main():
    dev = get_device()
    enc = FrozenBert(dev)
    v1, v2 = run_variant(False, enc, dev), run_variant(True, enc, dev)
    tr = v1["truth_ratio"]
    out = ["# Habit drift (`habit_drift.py`)\n",
           "**Setting.** 7 rounds of 50 training scenarios (the first 350 of data_dpo). Rounds 1–3 have the original labels. From round 4 the "
           "Saturday-fruit habit no longer holds: fruit scenarios on Saturday get the base necessity 0.80 (ask value 0.8 × receptiveness, act value "
           "0.2 instead of 0.9); the other habit rules and everything else are unchanged. The validation scenarios follow the same change (original up "
           "to round 3, drifted from round 4). Evaluation after every round on the 1512-cell grid under the old and under the new truth (they differ only "
           "in the 18 fruit-Saturday cells); regrets are against the oracle of the respective truth and normalised by its min(always-ask, never-ask).\n",
           "**Methods.** W = Ours (reward model warm-started and DPO policy continued, as in the semi-online loop) on all data so far; S = control 1, "
           "trained from scratch on the last three rounds only (150 scenarios); C = control 2, the tabular agent with constant learning rate 0.5 "
           "updated with each batch. 5 seeds, same seeds in all methods.\n",
           "**Criterion fixed beforehand.** W adapts in time if its collapse ratio at round 5 (two drifted rounds) exceeds 0.74, the midpoint between "
           "the pre-drift 0.47 and 1.0. Under the new rule the true ratio is not 1.0 but "
           f"{tr['new']:.2f} (Saturday has availability 1.0 against 0.8 on weekdays); under the old rule it is {tr['old']:.2f}.\n",
           "## Result of the experiment as specified\n"] + verdict(v1, "As specified") + tables(v1, "###") + \
          ["\n---\n", "## Supplementary variant: denser evidence (not part of the specification)\n",
           "In the specified data only 7 fruit-Saturday scenarios fall into rounds 1–3 and 1, 2, 5, 3 into rounds 4–7, too few for the old habit to be "
           "learned before the drift. As a one-shot supplementary run (fixed in advance, no tuning) 8 of every 50 training scenarios (6 of the 75 "
           "validation scenarios) were replaced by random fruit-Saturday scenarios (random emotion, time and phrasing), labelled with the old rule in "
           "rounds 1–3 and the new rule from round 4. Everything else is as above.\n"] + verdict(v2, "Dense-evidence variant") + tables(v2, "###")
    with open(os.path.join(RESULTS_DIR, "habit_drift.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
