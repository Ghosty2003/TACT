"""Rebuttal experiments 1 and 2 (results/v2/rebuttal/cost_receptivity.md, cost_receptivity_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_rebuttal_cost

Final architecture (frozen NLI-BERT mean pooling + Context encoder + rank-16 bilinear head, threshold rule), data_dpo, N in {150, 350}, 5 seeds, seed `seed * 1000 + N`, early stopping on the validation scenarios.
Methods (all trained on the first N training scenarios; the reward models see the same texts / labels as TACT, only the inputs differ):
  tact           full TACT: r_hat(q, c) for ask and act, ask iff r_hat(ask) - r_hat(act) > 0 (the main model of the paper)
  tact_noemo     the context-ablation model: the emotion embedding replaced by zeros (everything else as tact)
  q_const        CONSTANT-COST baseline: r_hat(q) = the same network trained on the ask rows only with ALL context embeddings replaced by zeros (so it sees only the question text); the act side is one scalar
                 c = mean of the observed act labels of the N training scenarios; ask iff r_hat(q) > c
  q_bucket       BUCKETED-COST baseline: as q_const, but c(e) = mean observed act label of the training scenarios with emotion e (6 scalars)
  q_const_opt    (extra, not asked for) as q_const, but c is the scalar that minimises the TRAINING regret (observed labels, candidate values = midpoints of the sorted r_hat(q) of the training scenarios),
                 i.e. the strongest possible constant cost that is still fitted on training data only
  tact_ask_const (extra) full TACT's ask side r_hat(ask | q, c) against the scalar c of q_const: the context stays on the ask side, only the act side is made constant
Evaluation: regret and agreement with the oracle on (a) the 75 test scenarios (test_seen; test_unseen = phrasing 3 as well) and (b) the 1512-cell noise-free grid (first question text of each topic), the grid
regret is also reported normalised by min(always-ask, never-ask) on the grid. Subsets: a (topic, day, time) triple is "emotion-sensitive" when the oracle decision (ask iff clean ask >= clean act) is not the
same for all six emotions; test scenarios / grid cells are assigned by their triple (the topic is used here for the evaluation split only, never by a method).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.drift.habit_drift import truth_grid
from tact.core.model import INPUT_COLS, RESULTS_DIR, FrozenBert, get_device, make_tensors

OUT = os.path.join(RESULTS_DIR, "rebuttal")
NS = [150, 350]
METHODS = ["tact", "tact_noemo", "q_const", "q_bucket", "q_const_opt", "tact_ask_const"]
ALL_CTX = ("emotion", "day", "time")


def fit_model(data, seed, n, drop=(), ask_only=False):
    """Reward model of the main pipeline (B.Ours.fit_reward); ask_only trains on the ask rows of the N scenarios (and validates on the ask rows of the validation scenarios)."""
    B.MODEL_KW = {"drop_ctx": drop} if drop else {}
    ours = B.Ours(data, seed)
    try:
        if ask_only:
            ids = set(data.sc_train.iloc[:n].id)
            tr = data.train_long[data.train_long.id.isin(ids) & (data.train_long.cand == "ask")]
            va = data.val_long[data.val_long.cand == "ask"]
            old_val = data.rm_val
            data.prefix = lambda k: (None, make_tensors(tr[INPUT_COLS + data.loc_cols], data.enc, data.dev), None)
            data.rm_val = make_tensors(va[INPUT_COLS + data.loc_cols], data.enc, data.dev)
            try:
                ours.fit_reward(n, seed * 1000 + n, warm=False)
            finally:
                del data.prefix
                data.rm_val = old_val
        else:
            ours.fit_reward(n, seed * 1000 + n, warm=False)
    finally:
        B.MODEL_KW = {}
    return ours


def best_scalar(rq, s_ask, s_act):
    """c minimising the training regret (observed labels) of 'ask iff rq > c'."""
    u = np.unique(rq)
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    best = np.maximum(s_ask, s_act)
    reg = [float((best - np.where(rq > c, s_ask, s_act)).mean()) for c in cands]
    return float(cands[int(np.argmin(reg))])


def scores(ours, f):
    ra, rc = B.reward_scores(ours.rm, f)
    return ra.cpu().numpy(), rc.cpu().numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    dev = get_device()
    enc = FrozenBert(dev)
    data = B.Data(enc, dev)
    g = truth_grid(False)
    f_grid = B.feats(g, enc, dev)
    sets = {"seen": (data.tests["seen"], data.f_test["seen"]), "unseen": (data.tests["unseen"], data.f_test["unseen"]), "grid": (g, f_grid)}
    # emotion sensitivity of every (topic, day, time) triple, from the clean truth of the grid
    oracle_g = (g.c_ask >= g.c_act).values
    key = list(zip(g.topic, g.day, g.time))
    tri = pd.DataFrame({"k": key, "o": oracle_g}).groupby("k").o.agg(["min", "max"])
    sens_tri = {k: bool(r["min"] != r["max"]) for k, r in tri.iterrows()}
    flag = {name: np.array([sens_tri[(t, d, m)] for t, d, m in zip(sc.topic, sc.day, sc.time)]) for name, (sc, _) in sets.items()}
    # (2a) how much can emotion matter at all
    best_g = np.maximum(g.c_ask.values, g.c_act.values)
    triple_val = pd.DataFrame({"k": key, "a": g.c_ask.values, "b": g.c_act.values}).groupby("k")[["a", "b"]].transform("mean")
    blind_ask = (triple_val.a >= triple_val.b).values          # best emotion-independent decision of each triple
    blind_regret = float((best_g - np.where(blind_ask, g.c_ask.values, g.c_act.values)).mean())
    grid_base = {"always": float((best_g - g.c_ask.values).mean()), "never": float((best_g - g.c_act.values).mean())}
    info = {"n_triples": len(sens_tri), "n_sens_triples": int(sum(sens_tri.values())), "n_cells": len(g), "n_sens_cells": int(flag["grid"].sum()),
            "n_test": len(data.tests["seen"]), "n_sens_test": int(flag["seen"].sum()), "blind_oracle_grid_regret": blind_regret,
            "grid_always": grid_base["always"], "grid_never": grid_base["never"], "grid_oracle_act_rate": float((~oracle_g).mean())}
    by_topic = pd.DataFrame({"topic": [k[0] for k in key], "sens": [sens_tri[k] for k in key]}).groupby("topic").sens.mean()
    pd.DataFrame({"topic": by_topic.index, "share_of_cells_in_sensitive_triples": by_topic.values}).to_csv(os.path.join(OUT, "sensitive_share_by_topic.csv"), index=False)
    pd.Series(info).to_csv(os.path.join(OUT, "sensitivity_info.csv"), header=False)

    rows = []

    def record(n, seed, method, dec):
        for name, (sc, _) in sets.items():
            ca, cc = sc.c_ask.values, sc.c_act.values
            best = np.maximum(ca, cc)
            reg_vec = best - np.where(dec[name], ca, cc)
            acc_vec = (dec[name] == (ca >= cc)).astype(float)
            for sub, m in (("all", np.ones(len(sc), bool)), ("sens", flag[name]), ("insens", ~flag[name])):
                rows.append({"n": n, "seed": seed, "method": method, "eval": name, "subset": sub, "n_scen": int(m.sum()), "regret": float(reg_vec[m].mean()),
                             "acc": float(acc_vec[m].mean()), "ask_rate": float(dec[name][m].mean())})

    for n in NS:
        for seed in B.SEEDS:
            tact = fit_model(data, seed, n)
            noemo = fit_model(data, seed, n, drop=("emotion",))
            qm = fit_model(data, seed, n, drop=ALL_CTX, ask_only=True)
            sc_tr = data.sc_train.iloc[:n]
            s_ask, s_act = sc_tr.s_ask.values, sc_tr.s_act.values
            c_const = float(s_act.mean())
            c_emo = sc_tr.groupby("emotion").s_act.mean()
            f_tr = B.feats(sc_tr, enc, dev)
            rq_tr = scores(qm, f_tr)[0]
            c_opt = best_scalar(rq_tr, s_ask, s_act)
            dec = {m: {} for m in METHODS}
            for name, (sc, f) in sets.items():
                ra, rc = scores(tact, f)
                dec["tact"][name] = ra > rc
                ra2, rc2 = scores(noemo, f)
                dec["tact_noemo"][name] = ra2 > rc2
                rq = scores(qm, f)[0]
                dec["q_const"][name] = rq > c_const
                dec["q_bucket"][name] = rq > sc.emotion.map(lambda e: c_emo.get(e, c_const)).values
                dec["q_const_opt"][name] = rq > c_opt
                dec["tact_ask_const"][name] = ra > c_const
            for m in METHODS:
                record(n, seed, m, dec[m])
            r = [x for x in rows if x["n"] == n and x["seed"] == seed and x["eval"] == "seen" and x["subset"] == "all"]
            print(f"[N={n} seed={seed}] seen regret " + " ".join(f"{x['method']}={x['regret']:.3f}" for x in r) + f" | c={c_const:.3f} c_opt={c_opt:.3f}", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "cost_receptivity_runs.csv"), index=False)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cost_receptivity_runs.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
