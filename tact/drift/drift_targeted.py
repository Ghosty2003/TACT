"""Drift-triggered targeted data invalidation ("targeted reset") against the default warm start, windowed retraining and recency-weighted replay
(results/drift_targeted.md, results/fig_drift_methods.pdf; raw csv in results/v2/drift_targeted_*.csv).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.drift.drift_targeted            # simulations (resumable) + report
    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.drift.drift_targeted --report-only

Protocol = the habit drift of Appendix C (drift_trigger.py's `v2` stage, final architecture, continuous labels, threshold rule): 15 rounds x 50 interactions of data_drift15 (dense Saturday-fruit evidence), the
Saturday-fruit habit stops being true after round 3 (necessity 0.8 instead of 0.1 from interaction 151 on; the validation scenarios used for early stopping follow the drift, as in drift_trigger.py), 5 seeds,
`drift` run vs `ctrl` (NO-DRIFT) run on the same stream with the original labels. The deployed model is retrained at the end of every round; between round ends it is fixed unless a trigger fires.
Methods
  W    default: warm start on all data at every round end                     S    windowed: from scratch on the last 3 rounds (150 scenarios) at every round end (existing)
  R2 / R4  recency-weighted replay: warm start on all data, every scenario weighted 0.5 ** (age in rounds / half-life), half-life 2 / 4 rounds (age 0 = the round being added; validation loss unweighted)
  T    targeted reset: warm start at every round end as W, plus
       1. surprise on ACT decisions only: when the deployed model's decision for an interaction is "act" (r_hat(act) >= r_hat(ask)), the observed act label y_act is compared with r_hat(act, c);
          surprise if |y_act - r_hat(act, c)| > tau. Interactions executed as "ask" cannot be surprises. Round 1 (cold start: everything is asked, no model yet) has none.
       2. trigger when >= k surprises of the CURRENT round fall into the same cell. Cell = (question cluster, day, time): same day, same time of day, and cosine of the frozen-encoder
          (NLI-BERT mean pooling) question vectors >= s. The oracle topic is never used by the method (it is used only for the precision statistics below).
       3. on trigger: every stored interaction of that cell that is OLDER than the first surprise of the group is deleted from the replay buffer (the new surprising interactions, and everything
          newer, are kept); the reward model is retrained FROM SCRATCH on the remaining data (all interactions so far minus the deleted ones, incl. the current one) and deployed immediately.
          Interpretation: "delete all stored interactions in that cell, keep the new surprising ones" is read as "older than the first surprise of the group" (as in drift_trigger.py), so that
          newer non-surprising evidence of the cell is not thrown away either.
       Defaults tau = 0.3, k = 2, s = 0.8; one-at-a-time sweeps tau in {0.2, 0.3, 0.4}, k in {1, 2, 3}, s in {0.7, 0.8, 0.9} (7 configurations).
Metrics (grid of 1512 cells, clean truth of the moment; `fs` = the 18 Saturday-fruit cells = the drifted cells): adaptation = interactions after the drift until the regret on the drifted cells (new truth)
is < 0.05; retention = learning degree L (mean of bread-Sunday-morning, coffee-weekday-morning, snack-Friday-evening) of the final reward model's ask scores; triggers and final test regret
(75 test scenarios, truth of the moment) in the NO-DRIFT run; final test / grid regret after drift; precision of the deleted rows = share that belongs to the drifted habit (topic fruit, Saturday; evaluation only).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import time

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.drift import drift_trigger as D
from tact.drift import habit_drift as H
from tact.core.generate_dataset import availability
from tact.core.model import CONFIG, INPUT_COLS, RESULTS_DIR, ROOT, FrozenBert, get_device, make_tensors
from tact.core.plots import is_habit

PER, ROUNDS, T_END, DRIFT_I, HAB3 = D.PER, D.ROUNDS, D.T_END, D.PER * D.DRIFT_ROUND, D.HAB3
TARGET = 0.05
DEFAULT = dict(tau=0.3, k=2, s=0.8)
SWEEP = [DEFAULT] + [dict(DEFAULT, tau=t) for t in (0.2, 0.4)] + [dict(DEFAULT, k=k) for k in (1, 3)] + [dict(DEFAULT, s=s) for s in (0.7, 0.9)]
SERIES = os.path.join(RESULTS_DIR, "drift_targeted_series.csv")
ROUNDS_CSV = os.path.join(RESULTS_DIR, "drift_targeted_rounds.csv")
EVENTS = os.path.join(RESULTS_DIR, "drift_targeted_events.csv")
FINAL = os.path.join(RESULTS_DIR, "drift_targeted_final.csv")


def cfg_name(m, c):
    return "-" if m != "T" else f"tau{c['tau']:.1f}_k{c['k']}_s{c['s']:.1f}"


# ------------------------------------------------------------------ weighted warm-start reward model (recency replay)
class OursW(B.Ours):
    weights = None     # scenario id -> weight

    def fit_reward(self, n, rnd_seed, warm):
        d, dev = self.d, self.d.dev
        sc = d.sc_train.iloc[d.keep]
        rows = d.train_long[d.train_long.id.isin(set(sc.id))]
        tr = make_tensors(rows[INPUT_COLS + d.loc_cols], d.enc, dev)
        tr["w"] = torch.tensor(rows.id.map(self.weights).values, dtype=torch.float32, device=dev)
        if self.rm is None or not warm:
            self.rm = B.new_scorer(d.enc, dev, rnd_seed, sigmoid=True)

        def loss(m, b):
            p = m(b["cls"], b["emotion"], b["day"], b["time"], b.get("location"), b.get("extra"))
            w = b["w"] if "w" in b else torch.ones_like(b["y"])      # validation: unweighted
            return (w * (p - b["y"]) ** 2).sum() / w.sum()
        h = self._hp("reward")
        B.fit(self.rm, loss, tr, d.rm_val, lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed)


# ------------------------------------------------------------------ one simulation
class Sim:
    def __init__(self, ctx, st, seed, method, cfg):
        self.ctx, self.st, self.seed, self.method, self.cfg = ctx, st, seed, method, cfg
        d = st.D["old"]
        self.f_test = d.f_test["seen"]
        self.test = d.tests["seen"]
        self.base_cache = {}
        # test truth of the moment: after the drift the Saturday-fruit scenarios follow the new rule
        t = self.test
        nec = H.NEW_NEC
        ca = np.array([nec * CONFIG["valence"][e] * availability(dd, tt) if (tp == "fruit" and dd == 5) else a
                       for tp, e, dd, tt, a in zip(t.topic, t.emotion, t.day, t.time, t.c_ask)])
        cc = np.array([1.0 - nec if (tp == "fruit" and dd == 5) else c for tp, dd, c in zip(t.topic, t.day, t.c_act)])
        self.test_truth = {"old": (t.c_ask.values, t.c_act.values), "new": (ca, cc)}

    def metrics(self, ask):
        key = ask.tobytes()
        if key not in self.base_cache:
            self.base_cache[key] = self.ctx.metrics(ask)
        return self.base_cache[key]

    def fit(self, ours, rnd, keep, warm, rnd_seed):
        d = self.st.D[self.st.key(rnd)]
        d.prefix = D.pool_prefix(d, keep)
        d.keep = keep
        ours.d = d
        ours.fit_round(len(keep), rnd_seed, warm=warm)

    def grid(self, ours):
        a, pa = self.st.decide_grid(ours)
        return a, pa

    def final_test(self, ours, truth):
        a, _ = ours.decide(self.f_test)
        ca, cc = self.test_truth[truth]
        return float((np.maximum(ca, cc) - np.where(a, ca, cc)).mean())

    def run(self):
        st, ctx, m, seed = self.st, self.ctx, self.method, self.seed
        sb = seed * 1000
        ours = (OursW if m[0] == "R" else st.Ours)(st.D["old"], seed)
        deleted = np.zeros(T_END, bool)
        series, rounds, events = {}, [], []
        n_trig = 0
        dec = pa = None
        tau, k, s = (self.cfg["tau"], self.cfg["k"], self.cfg["s"]) if m == "T" else (None, None, None)
        for r in range(1, ROUNDS + 1):
            idx = np.arange(PER * (r - 1), PER * r)
            if m == "T" and dec is not None:
                f_r = B.feats(st.sc.iloc[idx].reset_index(drop=True), ctx.enc, ctx.dev)
                with torch.no_grad():
                    ra, rc = (x.cpu().numpy() for x in B.reward_scores(ours.rm, f_r))
                pending = []
            for j, gi in enumerate(idx):
                i = gi + 1
                if m == "T" and dec is not None:
                    if not ra[j] > rc[j]:                                      # executed decision = act
                        if abs(st.s_act[gi] - rc[j]) > tau:                    # surprise
                            pending.append((gi, st.day[gi], st.time[gi], st.Q[gi]))
                            grp = [p for p in pending if p[1] == st.day[gi] and p[2] == st.time[gi] and float(p[3] @ st.Q[gi]) >= s]
                            if len(grp) >= k:
                                t0 = min(p[0] for p in grp)
                                cell = (st.day == st.day[gi]) & (st.time == st.time[gi]) & (st.Q @ st.Q[gi] >= s)
                                new_del = cell & (np.arange(T_END) < t0) & ~deleted
                                deleted |= new_del
                                n_trig += 1
                                hab_other = np.array([is_habit(tp, dd, tt) and tp != "fruit" for tp, dd, tt in zip(st.topic, st.day, st.time)])
                                events.append({"i": int(i), "round": r, "day": int(st.day[gi]), "time": int(st.time[gi]), "topic": st.topic[gi], "true_drift": bool(st.fs_new[gi]),
                                               "group_n": len(grp), "group_drifted": int(sum(st.fs_new[p[0]] for p in grp)), "deleted": int(new_del.sum()),
                                               "deleted_drifted_habit": int((new_del & (st.topic == "fruit") & (st.day == 5)).sum()),
                                               "deleted_other_habit": int((new_del & hab_other).sum()), "deleted_other": int((new_del & ~((st.topic == "fruit") & (st.day == 5))).sum())})
                                gids = {p[0] for p in grp}
                                pending = [p for p in pending if p[0] not in gids]
                                keep = [x for x in range(gi + 1) if not deleted[x]]
                                self.fit(ours, r, keep, warm=False, rnd_seed=sb + 800 + 20 * r + n_trig)
                                dec, pa = self.grid(ours)
                                with torch.no_grad():
                                    ra, rc = (x.cpu().numpy() for x in B.reward_scores(ours.rm, f_r))
                if dec is not None and i >= PER:
                    series[i] = self.metrics(dec)
            # round end
            keep = [x for x in range(PER * r) if not deleted[x]]
            if m == "S":
                keep = list(range(PER * max(0, r - 3), PER * r))
                ours = st.Ours(st.D[st.key(r)], seed)
            if m[0] == "R":                                    # R<half-life>: R1, R2, R3, R4, R6 ...
                h = float(m[1:])
                ours.weights = {st.sc.id.values[x]: 0.5 ** ((r - 1 - x // PER) / h) for x in keep}
            self.fit(ours, r, keep, warm=(m != "S" and r > 1), rnd_seed=sb + 900 + r)
            dec, pa = self.grid(ours)
            series[PER * r] = self.metrics(dec)
            rounds.append({"round": r, "n_pool": len(keep), "n_deleted": int(deleted.sum()), "n_triggers": n_trig, **ctx.learning(pa)})
        truth = "new" if st.run == "drift" else "old"
        fin = {"test_regret": self.final_test(ours, truth), "test_regret_old_truth": self.final_test(ours, "old"), "n_triggers": n_trig, "n_deleted": int(deleted.sum()),
               "n_deleted_drifted": int((deleted & (st.topic == "fruit") & (st.day == 5)).sum())}
        return series, rounds, events, fin


# ------------------------------------------------------------------ driver
def jobs():
    out = []
    for run in ("drift", "ctrl"):
        out += [(run, "W", None), (run, "S", None), (run, "R2", None), (run, "R4", None)] + [(run, "T", c) for c in SWEEP]
    return out


def append(path, df):
    df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def simulate(seeds):
    dev = get_device()
    enc = FrozenBert(dev)
    B.Data(enc, dev, data_dir=os.path.join(ROOT, "data_dpo"))        # prime the encoder cache in the fixed order of the other experiments
    ctx = D.Ctx(enc, dev)
    have = set(map(tuple, pd.read_csv(FINAL)[["run", "method", "cfg", "seed"]].values.tolist())) if os.path.exists(FINAL) else set()
    streams = {}
    for run, m, cfg in jobs():
        for seed in seeds:
            if (run, m, cfg_name(m, cfg), seed) in have:
                continue
            if run not in streams:
                streams[run] = D.Stream(ctx, "cont", run)
            t0 = time.perf_counter()
            series, rounds, events, fin = Sim(ctx, streams[run], seed, m, cfg).run()
            meta = {"run": run, "method": m, "cfg": cfg_name(m, cfg), "seed": seed}
            append(ROUNDS_CSV, pd.DataFrame([{**meta, **x} for x in rounds]))
            if events:
                append(EVENTS, pd.DataFrame([{**meta, **x} for x in events]))
            append(SERIES, pd.DataFrame([{**meta, "i": i, **v} for i, v in series.items()]))
            append(FINAL, pd.DataFrame([{**meta, **fin}]))                   # written last: marks the job as done
            print(f"[{run} {m} {cfg_name(m, cfg)} seed={seed}] fs_new(end)={series[T_END]['fs_new']:.3f} triggers={fin['n_triggers']} test={fin['test_regret']:.3f} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--report-only", action="store_true")
    a = ap.parse_args()
    if not a.report_only:
        simulate([int(x) for x in a.seeds.split(",")])
    from tact.drift.drift_targeted_report import write_report
    write_report()
