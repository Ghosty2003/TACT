"""Rebuttal experiment 4: is the tabular baseline under-tuned? (results/v2/rebuttal/tabular_tuning.md, tabular_tuning_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_rebuttal_tabular

The tabular agent of the paper (budget.Tabular('topic')): Q[(topic, emotion, day, time, action)] with a first-visit initialisation to the observed reward, a constant learning rate 0.5, and a back-off chain
exact cell -> (topic, action) mean -> global action mean (per action). Here 12 variants (+ 12 extra ones with two finer back-off chains, see EXTRA_CHAINS below; they are NOT part of the specified search): learning rate in {0.1, 0.3, 0.5, 0.8, 1.0, decay} x chain in {full (current), no-topic-level (exact cell -> global action mean)};
`decay` = the rate 1 / (number of visits of that cell), i.e. the running mean of its observed rewards. The first visit still initialises Q to the observed reward; the visiting order is the seed-dependent shuffle
of the main experiment (budget.sweep).
Selection: for every N the variant with the lowest mean (over the 5 seeds) VALIDATION regret on the 75 validation scenarios, computed with the noise-free clean values of the validation scenarios (this is more
information than a deployed agent has, i.e. it favours the baseline; the validation regret on the observed labels is stored as well); ties go to the current configuration. A single configuration for all N (lowest validation regret averaged over the
N) is reported too. Test: regret on the 75 test scenarios (seen), agreement with the oracle, regret on the 1512-cell grid. For information the best variant on the TEST set at every N (an upper bound that no
validation-based choice can beat) and every variant's test regret are tabulated.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.drift.habit_drift import truth_grid
from tact.core.model import RESULTS_DIR, FrozenBert, get_device

OUT = os.path.join(RESULTS_DIR, "rebuttal")
LRS = [0.1, 0.3, 0.5, 0.8, 1.0, "decay"]
CHAINS = ["full", "notopic"]


# back-off levels between the exact cell and the global action mean: each is a function of a scenario row -> key
LEVELS = {"full": [lambda r: (r.topic,)], "notopic": [],                                              # the two chains asked for
          "tdt": [lambda r: (r.topic, r.day, r.time), lambda r: (r.topic,)],                            # extra: (topic, day, time) mean, then topic mean
          "te": [lambda r: (r.topic, r.emotion), lambda r: (r.topic,)]}                                 # extra: (topic, emotion) mean, then topic mean
SPEC_CHAINS = ["full", "notopic"]
EXTRA_CHAINS = ["tdt", "te"]
CHAINS = SPEC_CHAINS + EXTRA_CHAINS


class Tab:
    def __init__(self, lr, chain):
        self.lr, self.chain, self.visits, self.Q = lr, chain, {}, {}
        self.sum = [{} for _ in LEVELS[chain]]
        self.cnt = [{} for _ in LEVELS[chain]]
        self.gsum, self.gcnt = {"ask": 0.0, "act": 0.0}, {"ask": 0, "act": 0}

    def update(self, sc, order):
        for i in order:
            r = next(sc.iloc[[i]].itertuples())
            for a, val in (("ask", r.s_ask), ("act", r.s_act)):
                k = (r.topic, r.emotion, r.day, r.time, a)
                v = self.visits.get(k, 0) + 1
                self.visits[k] = v
                if k not in self.Q:
                    self.Q[k] = val
                else:
                    rate = 1.0 / v if self.lr == "decay" else self.lr
                    self.Q[k] += rate * (val - self.Q[k])
                for lv, fn in enumerate(LEVELS[self.chain]):
                    kk = (fn(r), a)
                    self.sum[lv][kk] = self.sum[lv].get(kk, 0.0) + val
                    self.cnt[lv][kk] = self.cnt[lv].get(kk, 0) + 1
                self.gsum[a] += val
                self.gcnt[a] += 1

    def revisit_share(self):
        n_upd = sum(self.visits.values())
        return (n_upd - len(self.visits)) / n_upd

    def decide(self, sc):
        out = []
        for r in sc.itertuples():
            vals = {}
            for a in ("ask", "act"):
                k = (r.topic, r.emotion, r.day, r.time, a)
                if k in self.Q:
                    vals[a] = self.Q[k]
                    continue
                for lv, fn in enumerate(LEVELS[self.chain]):
                    kk = (fn(r), a)
                    if kk in self.cnt[lv]:
                        vals[a] = self.sum[lv][kk] / self.cnt[lv][kk]
                        break
                else:
                    vals[a] = self.gsum[a] / max(self.gcnt[a], 1)
            out.append(vals["ask"] > vals["act"])
        return np.array(out)


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    g = truth_grid(False)
    oracle_g = (g.c_ask >= g.c_act).values
    key = list(zip(g.topic, g.day, g.time))
    tri = pd.DataFrame({"k": key, "o": oracle_g}).groupby("k").o.agg(["min", "max"])
    sens_tri = {k: bool(r["min"] != r["max"]) for k, r in tri.iterrows()}
    sets = {"seen": data.tests["seen"], "unseen": data.tests["unseen"], "grid": g, "val": data.sc_val}
    flag = {nm: np.array([sens_tri[(t, d, m)] for t, d, m in zip(sc.topic, sc.day, sc.time)]) for nm, sc in sets.items()}
    rows = []
    for n in B.NS:
        sc_tr = data.sc_train.iloc[:n]
        for seed in B.SEEDS:
            order = np.random.default_rng([seed, n]).permutation(n)
            for lr in LRS:
                for chain in CHAINS:
                    t = Tab(lr, chain)
                    t.update(sc_tr, order)
                    row = {"n": n, "seed": seed, "lr": str(lr), "chain": chain, "revisit_share": t.revisit_share()}
                    for nm, sc in sets.items():
                        d = t.decide(sc)
                        ca, cc = sc.c_ask.values, sc.c_act.values
                        reg = np.maximum(ca, cc) - np.where(d, ca, cc)
                        row[f"{nm}_regret"] = float(reg.mean())
                        row[f"{nm}_acc"] = float((d == (ca >= cc)).mean())
                        if nm in ("seen", "grid"):
                            row[f"{nm}_regret_sens"], row[f"{nm}_regret_insens"] = float(reg[flag[nm]].mean()), float(reg[~flag[nm]].mean())
                        if nm == "val":
                            obs = np.maximum(sc.s_ask.values, sc.s_act.values) - np.where(d, sc.s_ask.values, sc.s_act.values)
                            row["val_regret_obs"] = float(obs.mean())
                    rows.append(row)
        print(f"[tabular tuning] N={n} done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tabular_tuning_runs.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
