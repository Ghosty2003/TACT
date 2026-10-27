"""Experiment B: opposition-triggered local relearning (results/drift_trigger.md, .png).

    python -m tact.drift.drift_trigger                 # data, all simulations (resumable), report
    python -m tact.drift.drift_trigger --report-only

Stream: 15 rounds x 50 interactions (scenarios of data_drift15_src = `generate_dataset.py --n 1100 --with-act`, first 750 training
scenarios). Dense-evidence variant of habit_drift.py: 8 of every 50 scenarios (13 of the 165 validation scenarios) are replaced by random
fruit-Saturday scenarios. Rounds 1-3 follow the original labels; from round 4 on (interaction 151) the Saturday-fruit rule no longer holds
(necessity 0.8 instead of 0.1; the validation scenarios used for early stopping follow). `ctrl` runs: the same stream with the original
labels throughout (no habit change). Labels: `cont` = continuous scores, `bin` = 0/1 accept/reject (Bernoulli of the clean value).

Methods (all decide with the DPO policy A; deployed model = the one retrained at the last round end):
  W  warm start on all data (paper default);          S  from scratch on the last three rounds (150 scenarios);
  C  tabular agent updated online after every interaction;   Cb  tabular updated once per round (habit_drift.py style);
  N  warm start + opposition-triggered local relearning.
N, per new interaction (from round 2 on, when a model exists): the executed action is the policy's decision (ask if the class is overridden);
residual = |label of the executed action - r_hat(executed action)|; opposition if residual > delta (bin: r_hat > 0.8 and label 0, or
r_hat < 0.2 and label 1). Class: two interactions are the same class iff the cosine of their frozen-BERT question CLS vectors is > 0.9 AND the
day and the time of day are equal (the topic field is never used; `topic` class mode is a labelled diagnostic only). When a class has
accumulated k oppositions: (i) override = decisions in that class (grid: same day, same time, CLS cosine > 0.9 with the triggering question)
are forced to "ask" until the next retrain; (ii) all samples of that class that are older than the first opposition of the group are deleted
from the training pool; all other samples are kept; the next retrain is warm-started as usual.
Interpretations I had to make are listed in the report.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import shutil
import subprocess
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tact.decision import budget as B
from tact.drift import habit_drift as H
from tact.core import plots
from tact.decision.binary_feedback import OursBin, TabFreq
from tact.core.generate_dataset import QUESTIONS
from tact.core.model import INPUT_COLS, RESULTS_DIR, ROOT, FrozenBert, get_device, make_tensors
from tact.core.plots import is_habit

SRC = os.path.join(ROOT, "data_drift15_src")
DST = os.path.join(ROOT, "data_drift15")
ROUNDS, PER, DRIFT_ROUND = 15, 50, 3          # rounds 1..3 original rules, 4..15 drifted
T_END = ROUNDS * PER
SEEDS = [0, 1, 2, 3, 4]
COS = 0.9
MAIN = dict(delta=0.15, k=2, cls="cls")
VARIANTS = [dict(delta=0.10, k=2, cls="cls"), dict(delta=0.20, k=2, cls="cls"), dict(delta=0.15, k=1, cls="cls"), dict(delta=0.15, k=3, cls="cls")]
DIAG = dict(delta=0.15, k=2, cls="topic")      # diagnostic only: the class uses the topic field (not allowed in the method)
TARGET = 0.05
HAB3 = ("bread", "coffee", "snack")
SERIES_CSV = os.path.join(RESULTS_DIR, "drift_trigger_series.csv")
ROUNDS_CSV = os.path.join(RESULTS_DIR, "drift_trigger_rounds.csv")
EVENTS_CSV = os.path.join(RESULTS_DIR, "drift_trigger_events.csv")


def cfg_name(c):
    return "-" if c is None else f"d{c['delta']:.2f}_k{c['k']}" + ("_topicclass" if c["cls"] == "topic" else "")


# ------------------------------------------------------------------ data
def ensure_src():
    if not os.path.exists(os.path.join(SRC, "train.csv")):
        subprocess.run([sys.executable, "-m", "tact.core.generate_dataset", "--n", "1100", "--seed", "42", "--with-act", "--outdir", SRC],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def build_data(label, run):
    """data_drift15/{label}_{run}/{old_val,new_val}: same training file, validation with the old / the new labels."""
    ensure_src()
    base = os.path.join(DST, f"{label}_{run}")
    if os.path.exists(os.path.join(base, "new_val", "train.csv")):
        return base
    tr, va = pd.read_csv(os.path.join(SRC, "train.csv")), pd.read_csv(os.path.join(SRC, "val.csv"))
    keep = tr[tr.cand == "ask"].id.tolist()[:T_END]
    tr = tr[tr.id.isin(set(keep))].reset_index(drop=True)
    n_val = int((va.cand == "ask").sum())
    rng = np.random.default_rng(777)
    p_tr, s_tr = H.dense_specs(rng, ROUNDS, H.DENSE_PER_BATCH)
    p_va, s_va = H.dense_specs(rng, 1, 13, per_batch=n_val)
    early = [k for k, p in enumerate(p_tr) if p < PER * DRIFT_ROUND]
    late = [k for k, p in enumerate(p_tr) if p >= PER * DRIFT_ROUND]
    if run == "drift":
        tr = H.inject(tr, [p_tr[k] for k in early], [s_tr[k] for k in early], False)
        tr = H.inject(tr, [p_tr[k] for k in late], [s_tr[k] for k in late], True)
        va_old, va_new = H.inject(va, p_va, s_va, False), H.inject(va, p_va, s_va, True)
        tr, _ = H.relabel(tr, PER * DRIFT_ROUND, 4242)
        va_new, _ = H.relabel(va_new, 0, 4243)
    else:
        tr = H.inject(tr, p_tr, s_tr, False)
        va_old = va_new = H.inject(va, p_va, s_va, False)
    if label == "bin":   # 0/1 feedback: Bernoulli draw of the clean value, one draw per row
        brng = np.random.default_rng(4321)
        for df in (tr, va_old, va_new):
            df["score"] = (brng.random(len(df)) < df["score_clean"].values).astype(float)
    for sub, val in (("old_val", va_old), ("new_val", va_new)):
        d = os.path.join(base, sub)
        os.makedirs(d, exist_ok=True)
        tr.to_csv(os.path.join(d, "train.csv"), index=False)
        val.to_csv(os.path.join(d, "val.csv"), index=False)
        for f in ("test_seen.csv", "test_unseen.csv"):
            shutil.copy(os.path.join(SRC, f), os.path.join(d, f))
    return base


def pool_prefix(data, keep):
    """Training pool = the scenarios at the given positions of data.sc_train (n is ignored)."""
    def prefix(n):
        sc = data.sc_train.iloc[keep]
        rows = data.train_long[data.train_long.id.isin(set(sc.id))]
        return sc, make_tensors(rows[INPUT_COLS + data.loc_cols], data.enc, data.dev), B.feats(sc, data.enc, data.dev)
    return prefix


# ------------------------------------------------------------------ context (grid, masks, class geometry)
class Ctx:
    def __init__(self, enc, dev):
        self.enc, self.dev = enc, dev
        self.g_old, self.g_new = H.truth_grid(False), H.truth_grid(True)
        g = self.g_old
        self.f_grid = B.feats(g, enc, dev)
        self.ca_old, self.cc_old, self.ca_new, self.cc_new = g.c_ask.values, g.c_act.values, self.g_new.c_ask.values, self.g_new.c_act.values
        self.fs = ((g.topic == "fruit") & (g.day == 5)).values
        self.hab = {t: np.array([(tt == t) and is_habit(tt, d, m) for tt, d, m in zip(g.topic, g.day, g.time)]) for t in HAB3}
        self.own = {t: (g.topic == t).values for t in HAB3}
        self.truth_ratio = {t: self.ca_old[self.hab[t]].mean() / self.ca_old[self.own[t] & ~self.hab[t]].mean() for t in HAB3}
        self.truth_collapse = {"old": H.collapse_ratio(self.ca_old, self.g_old), "new": H.collapse_ratio(self.ca_new, self.g_new)}
        self.base = {k: min(H.regret(np.ones(len(g), bool), a, c), H.regret(np.zeros(len(g), bool), a, c))
                     for k, (a, c) in (("old", (self.ca_old, self.cc_old)), ("new", (self.ca_new, self.cc_new)))}
        texts = g.question.tolist()
        self.Qg = F.normalize(enc.encode(texts), dim=1).cpu().numpy()          # grid question CLS (unit)
        self.g_day, self.g_time, self.g_topic = g.day.values, g.time.values, g.topic.values

    def metrics(self, ask):
        out = {}
        for rule, a, c in (("old", self.ca_old, self.cc_old), ("new", self.ca_new, self.cc_new)):
            r = np.maximum(a, c) - np.where(ask, a, c)
            out[f"grid_{rule}"], out[f"fs_{rule}"] = float(r.mean()), float(r[self.fs].mean())
            if rule == "old":
                for t in HAB3:
                    out[f"h_{t}"] = float(r[self.hab[t]].mean())
        return out

    def learning(self, ask_pred):
        """Saturday-fruit collapse ratio and the learning degree L of the three unchanged habits (from predicted ask values on the grid)."""
        out = {"collapse": H.collapse_ratio(ask_pred, self.g_old)}
        for t in HAB3:
            r = ask_pred[self.hab[t]].mean() / ask_pred[self.own[t] & ~self.hab[t]].mean()
            out[f"L_{t}"] = float((1 - r) / (1 - self.truth_ratio[t]))
        return out


class Stream:
    """Per (label, run): the two Data objects (old / new validation), scenario geometry and the interactions in order."""

    def __init__(self, ctx, label, run):
        base = build_data(label, run)
        self.label, self.run, self.ctx = label, run, ctx
        self.D = {k: B.Data(ctx.enc, ctx.dev, data_dir=os.path.join(base, f"{k}_val")) for k in ("old", "new")}
        d = self.D["old"]
        sc = d.sc_train.iloc[:T_END]
        assert len(sc) == T_END
        self.sc = sc.reset_index(drop=True)
        self.Q = F.normalize(ctx.enc.encode(self.sc.question.tolist()), dim=1).cpu().numpy()
        self.day, self.time, self.topic = self.sc.day.values, self.sc.time.values, self.sc.topic.values
        self.s_ask, self.s_act = self.sc.s_ask.values, self.sc.s_act.values
        self.fs_new = (self.topic == "fruit") & (self.day == 5) & (np.arange(T_END) >= PER * DRIFT_ROUND) & (run == "drift")
        self.Ours = OursBin if label == "bin" else B.Ours

    def key(self, rnd):   # validation labels follow the drift
        return "old" if (self.run == "ctrl" or rnd <= DRIFT_ROUND) else "new"

    def rel(self, o_day, o_time, o_q, o_topic, mode):
        """Boolean mask over all training scenarios: same class as the interaction described by (day, time, question vector[, topic])."""
        m = (self.day == o_day) & (self.time == o_time) & (self.Q @ o_q > COS)
        return m & (self.topic == o_topic) if mode == "topic" else m

    def grid_rel(self, o_day, o_time, o_q, o_topic, mode):
        c = self.ctx
        m = (c.g_day == o_day) & (c.g_time == o_time) & (c.Qg @ o_q > COS)
        return m & (c.g_topic == o_topic) if mode == "topic" else m

    def fit(self, ours, rnd, keep, warm):
        d = self.D[self.key(rnd)]
        d.prefix = pool_prefix(d, keep)
        ours.d = d
        ours.fit_round(len(keep), self.seed_base + 900 + rnd, warm=warm)

    def decide_grid(self, ours):
        a, _ = ours.decide(self.ctx.f_grid)
        with torch.no_grad():
            pa = B.reward_scores(ours.rm, self.ctx.f_grid)[0].cpu().numpy()
        return a, pa


# ------------------------------------------------------------------ simulations
def run_step(st, seed, method):
    """W (warm start, all data) or S (scratch, last three rounds): retrain at every round end; metrics step-wise."""
    st.seed_base = seed * 1000
    ours = st.Ours(st.D["old"], seed)
    rnds = []
    dec = {}
    for r in range(1, ROUNDS + 1):
        lo = PER * max(0, r - 3) if method == "S" else 0
        keep = list(range(lo, PER * r))
        if method == "S":
            ours = st.Ours(st.D[st.key(r)], seed)
        st.fit(ours, r, keep, warm=(method == "W" and r > 1))
        a, pa = st.decide_grid(ours)
        dec[r] = a
        rnds.append({"round": r, "n_pool": len(keep), "n_deleted": 0, "n_triggers": 0, **st.ctx.learning(pa)})
    series = {}
    for i in range(1, T_END + 1):
        r = i // PER
        series[i] = st.ctx.metrics(dec[r]) if r >= 1 else None
    return series, rnds, []


def run_new(st, seed, cfg):
    """N: warm start + opposition-triggered local relearning."""
    ctx = st.ctx
    st.seed_base = seed * 1000
    ours = st.Ours(st.D["old"], seed)
    delta, k, mode = cfg["delta"], cfg["k"], cfg["cls"]
    deleted = np.zeros(T_END, bool)
    series, rnds, events = {}, [], []
    n_trig = 0
    base_dec = None
    for r in range(1, ROUNDS + 1):
        idx = np.arange(PER * (r - 1), PER * r)
        overrides, pending = [], []     # overrides: (day, time, q, topic); pending oppositions: (index, day, time, q, topic)
        if base_dec is not None:
            sc_r = st.sc.iloc[idx].reset_index(drop=True)
            f_r = B.feats(sc_r, ctx.enc, ctx.dev)
            with torch.no_grad():
                ra, rc = (x.cpu().numpy() for x in B.reward_scores(ours.rm, f_r))
                pol_ask = (B.margin(ours.pol, f_r) > 0).cpu().numpy()
        for j, gi in enumerate(idx):
            i = gi + 1
            if base_dec is not None:
                d_, t_, q_, tp_ = st.day[gi], st.time[gi], st.Q[gi], st.topic[gi]
                overridden = any(st.rel_one(o, d_, t_, q_, tp_, mode) for o in overrides)
                ask = bool(pol_ask[j]) or overridden
                rhat, y = (ra[j], st.s_ask[gi]) if ask else (rc[j], st.s_act[gi])
                opp = ((rhat > 0.8 and y == 0) or (rhat < 0.2 and y == 1)) if st.label == "bin" else abs(y - rhat) > delta
                if opp and not overridden:
                    pending.append((gi, d_, t_, q_, tp_))
                    group = [p for p in pending if st.rel_one(p, d_, t_, q_, tp_, mode)]
                    if len(group) >= k:
                        t0 = min(p[0] for p in group)
                        m = st.rel(d_, t_, q_, tp_, mode)
                        new_del = m & (np.arange(T_END) < t0) & ~deleted
                        deleted |= new_del
                        overrides.append((d_, t_, q_, tp_))
                        pending = [p for p in pending if not st.rel_one(p, d_, t_, q_, tp_, mode)]
                        n_trig += 1
                        hab_other = np.array([is_habit(tp, dd, tt) and tp != "fruit" for tp, dd, tt in zip(st.topic, st.day, st.time)])
                        events.append({"i": int(i), "round": r, "day": int(d_), "time": int(t_), "topic": tp_, "true_drift": bool(st.fs_new[gi]),
                                       "group_n": len(group), "group_fs_new": int(sum(st.fs_new[p[0]] for p in group)), "t0": int(t0),
                                       "deleted": int(new_del.sum()), "deleted_fruit_sat": int((new_del & (st.topic == "fruit") & (st.day == 5)).sum()),
                                       "deleted_other_habit": int((new_del & hab_other).sum()), "deleted_other_topics": int((new_del & (st.topic != "fruit")).sum())})
                dec = base_dec.copy()
                for o in overrides:
                    dec[st.grid_rel(o[0], o[1], o[2], o[3], mode)] = True
                series[i] = ctx.metrics(dec)
            else:
                series[i] = None
        keep = [x for x in range(PER * r) if not deleted[x]]
        st.fit(ours, r, keep, warm=r > 1)
        base_dec, pa = st.decide_grid(ours)
        rnds.append({"round": r, "n_pool": len(keep), "n_deleted": int(deleted.sum()), "n_triggers": n_trig, **ctx.learning(pa)})
        series[PER * r] = ctx.metrics(base_dec)      # the retrained model is deployed, overrides cleared
    return series, rnds, events


def run_tab(st, seed, online):
    """Tabular agent: online (after every interaction) or per round (batch, shuffled order as in habit_drift.py).
    The state during round r is the table after round r - 1 (online: after every interaction)."""
    ctx = st.ctx
    tab = (TabFreq if st.label == "bin" else B.Tabular)("topic")
    sc = st.sc
    series, rnds = {}, []
    prev = None
    for r in range(1, ROUNDS + 1):
        batch = sc.iloc[PER * (r - 1):PER * r].reset_index(drop=True)
        if online:
            for j in range(PER):
                tab.update(batch, [j])
                dec, _ = tab.decide(ctx.g_old)
                series[PER * (r - 1) + j + 1] = ctx.metrics(dec)
        else:
            tab.update(batch, np.random.default_rng([seed, r, 7]).permutation(PER))
            cur, _ = tab.decide(ctx.g_old)
            for i in range(PER * (r - 1) + 1, PER * r):
                if prev is not None:
                    series[i] = ctx.metrics(prev)
            series[PER * r] = ctx.metrics(cur)
            prev = cur
        pa = H.tab_ask_values(tab, ctx.g_old)
        rnds.append({"round": r, "n_pool": PER * r, "n_deleted": 0, "n_triggers": 0, **ctx.learning(pa)})
    return series, rnds, []


def _rel_one(self, o, day, time, q, topic, mode):
    """Is the interaction (day, time, q, topic) in the class of o = (..., day, time, q, topic)? o may be a pending tuple (with leading index)."""
    od, ot, oq, otp = o[-4:]
    if od != day or ot != time or float(oq @ q) <= COS:
        return False
    return (otp == topic) if mode == "topic" else True


Stream.rel_one = _rel_one


# ------------------------------------------------------------------ driver
JOBS = []


def jobs(stage):
    out = []
    if stage in ("main", "all"):
        for run in ("drift", "ctrl"):
            out += [("cont", run, "W", None), ("cont", run, "S", None), ("cont", run, "N", MAIN)]
        out += [("cont", "drift", "C", None), ("cont", "drift", "Cb", None)]
    if stage in ("variants", "all"):
        for run in ("drift", "ctrl"):
            out += [("cont", run, "N", c) for c in VARIANTS + [DIAG]]
    if stage == "v2":       # final-architecture baselines of the 15-round habit drift (no local relearning): warm start, last-3-rounds retrain, tabular
        out += [("cont", "drift", "W", None), ("cont", "drift", "S", None), ("cont", "drift", "C", None), ("cont", "drift", "Cb", None)]
    if stage in ("bin", "all"):
        for run in ("drift", "ctrl"):
            out += [("bin", run, "W", None), ("bin", run, "S", None), ("bin", run, "N", MAIN)]
        out += [("bin", "drift", "C", None), ("bin", "drift", "Cb", None)]
    return out


def done_keys():
    if not os.path.exists(SERIES_CSV):
        return set()
    d = pd.read_csv(SERIES_CSV, usecols=["label", "run", "method", "cfg", "seed"]).drop_duplicates()
    return set(map(tuple, d.values.tolist()))


def append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def simulate(stage, seeds):
    dev = get_device()
    enc = FrozenBert(dev)
    # prime the CLS cache in the same fixed order as the other experiments (data_dpo first), then everything else
    B.Data(enc, dev, data_dir=os.path.join(ROOT, "data_dpo"))
    ctx = Ctx(enc, dev)
    streams = {}
    have = done_keys()
    for label, run, method, cfg in jobs(stage):
        for seed in seeds:
            key = (label, run, method, cfg_name(cfg), seed)
            if key in have:
                continue
            if (label, run) not in streams:
                streams[(label, run)] = Stream(ctx, label, run)
            st = streams[(label, run)]
            t0 = time.perf_counter()
            if method in ("W", "S"):
                series, rnds, ev = run_step(st, seed, method)
            elif method == "N":
                series, rnds, ev = run_new(st, seed, cfg)
            else:
                series, rnds, ev = run_tab(st, seed, online=(method == "C"))
            rows = [{"label": label, "run": run, "method": method, "cfg": cfg_name(cfg), "seed": seed, "i": i, **v}
                    for i, v in series.items() if v is not None]
            meta = {"label": label, "run": run, "method": method, "cfg": cfg_name(cfg), "seed": seed}
            append(ROUNDS_CSV, pd.DataFrame([{**meta, **r} for r in rnds]))
            if ev:
                append(EVENTS_CSV, pd.DataFrame([{**meta, **e} for e in ev]))
            append(SERIES_CSV, pd.DataFrame(rows))    # written last: marks the job as done
            print(f"[{label} {run} {method} {cfg_name(cfg)} seed={seed}] fs_new(750)={series[T_END]['fs_new']:.3f} grid_new(750)={series[T_END]['grid_new']:.3f} "
                  f"triggers={len(ev)} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("main", "variants", "bin", "all", "v2"))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        simulate(args.stage, [int(s) for s in args.seeds.split(",")])
    from tact.drift.drift_trigger_report import write_report
    write_report()
