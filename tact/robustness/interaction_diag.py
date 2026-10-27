"""Diagnostic: why is the 'activity x topic' rule not learned? (results/v2/interaction_diag.md, interaction_diag_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.robustness.interaction_diag [--report-only]

Nothing used by the paper is modified. Final architecture (TACT_V2=1), 5 seeds, N = 350 and 700 (first N scenarios of the 700 training scenarios of an `--n 1000` dataset, as in Appendix A).
'Learned' = learning degree L >= 0.5 (fixed beforehand). L = (1 - predicted ratio) / (1 - true ratio), the ratio being the mean predicted ASK score of the affected cells
(activity = cooking, topic in the rule's scope, non-habit) over that of the control cells (the other two activities, same topics), over the full grid.

Step 0  rule definition and a check that the labels of the training sets contain the rule (statistics on `data_ctx7_n1000`).
Step 1  upper-bound reference models on the same training data: one-hot inputs (topic, emotion, day, time, extra variables) PLUS the explicit topic x activity cross features: ridge regression
        (alpha = 1) and a small MLP (64-32, Adam 1e-3, batch 16, early stopping on the validation scenarios), both trained on the ask rows only.
Step 2  4 variables (3 basic + activity), rule variants built by relabelling the scenarios of `data_diag4_n1000` (`--extra-vars activity`): scope {fruit only, the 4 kitchen topics}
        x strength {necessity x 0.5, necessity set to 0.10 like a habit rule} x density {default sampling, ~20% of the scenarios affected: random scenarios of the scope with activity = cooking
        replace others, as the dense variant of Appendix C}; plus two references: the 7-variable rule of Appendix A and the 4-variable existing rule with the act score NOT changed by the rule.
Every variant: all labels re-derived from the scenario (clean ask = necessity' x receptiveness, clean act = 1 - necessity' [or 1 - the unchanged necessity], noise N(0, 0.05) from a generator seeded by
split and scenario id), so the variants differ only in the factor under test.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import itertools
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from tact.decision import budget as B
from tact.robustness import context_scaling as CS
from tact.core.generate_dataset import ACT_TEXT, DAYS, HABIT_NECESSITY, KITCHEN_TOPICS, QUESTIONS, TIMES, VALENCE, availability, habit_hit, necessity
from tact.core.grid_eval import TOPICS
from tact.core.model import CONFIG, EMOTIONS, EXTRA_SPEC, RESULTS_DIR, ROOT, FrozenBert, get_device
from tact.core.plots import is_habit

BASE4 = os.path.join(ROOT, "data_diag4_n1000")
OUT = os.path.join(ROOT, "data_diag")
RUNS_CSV = os.path.join(RESULTS_DIR, "interaction_diag_runs.csv")
NS = (350, 700)
SEEDS = B.SEEDS
BOOST = 0.20
SCOPES = {"fruit": {"fruit"}, "kitchen": set(KITCHEN_TOPICS)}


def cond_name(scope, strength, density, act="change"):
    return f"{scope}_x{strength:g}_{density}" + ("" if act == "change" else "_actfixed")


CONDS = {cond_name(s, st, d): dict(scope=s, strength=st, density=d, act="change") for s, st, d in itertools.product(("fruit", "kitchen"), (0.5, 0.1), ("default", "boost"))}
CONDS["kitchen_x0.5_default_actfixed"] = dict(scope="kitchen", strength=0.5, density="default", act="fixed")
# controls for the density factor: only the TRAINING scenarios are densified, the validation scenarios (early stopping) keep the default density
CONDS["kitchen_x0.5_boosttrain"] = dict(scope="kitchen", strength=0.5, density="boost_train", act="change")
CONDS["fruit_x0.5_boosttrain"] = dict(scope="fruit", strength=0.5, density="boost_train", act="change")
REF7 = "ref_7var_existing"


# ------------------------------------------------------------------ data construction (4 variables)
def ensure_base():
    if not os.path.exists(os.path.join(BASE4, "train.csv")):
        subprocess.run([sys.executable, "-m", "tact.core.generate_dataset", "--n", "1000", "--seed", "42", "--with-act", "--extra-vars", "activity", "--outdir", BASE4],
                       check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def nec_new(topic, day, time, activity, c):
    base = necessity(topic, day, time)
    if activity == "cooking" and topic in SCOPES[c["scope"]]:
        return base * 0.5 if c["strength"] == 0.5 else HABIT_NECESSITY, base
    return base, base


def relabel(df, c, split, rng_inject=None, n_blocks=None):
    """df: long-format csv of one split (ask + act rows per id). Optionally injects affected scenarios (density), then re-derives every label."""
    df = df.copy()
    ask_idx = {i: k for k, i in zip(df.index[df.cand == "ask"], df.id[df.cand == "ask"])}
    act_idx = {i: k for k, i in zip(df.index[df.cand == "act"], df.id[df.cand == "act"])}
    ids = list(df.id[df.cand == "ask"])
    if rng_inject is not None:
        scope = sorted(SCOPES[c["scope"]])
        affected = lambda i: df.at[ask_idx[i], "activity"] == "cooking" and df.at[ask_idx[i], "topic"] in SCOPES[c["scope"]]
        for blk in n_blocks:                      # blocks of scenario positions, each brought to ~BOOST affected scenarios
            blk_ids = [ids[p] for p in blk]
            need = int(round(BOOST * len(blk_ids))) - sum(affected(i) for i in blk_ids)
            cand = [i for i in blk_ids if not affected(i)]
            for i in rng_inject.choice(cand, max(need, 0), replace=False):
                topic = scope[rng_inject.integers(len(scope))]
                emo, day, tm, ph = EMOTIONS[rng_inject.integers(len(EMOTIONS))], int(rng_inject.integers(7)), int(rng_inject.integers(3)), int(rng_inject.integers(1, 3))
                hh = habit_hit(topic, day, tm)
                for cand_, k in (("ask", ask_idx[i]), ("act", act_idx[i])):
                    df.loc[k, ["topic", "emotion", "day", "day_name", "time", "time_name", "activity"]] = [topic, emo, day, DAYS[day], tm, TIMES[tm], "cooking"]
                    df.loc[k, "phrasing"] = ph if cand_ == "ask" else 0
                    df.loc[k, "question"] = QUESTIONS[topic][ph - 1] if cand_ == "ask" else ACT_TEXT
                    df.loc[k, "habit_hit"] = hh["desc"] if hh else np.nan
    for i in ids:
        a, b = ask_idx[i], act_idx[i]
        r = df.loc[a]
        nec, base = nec_new(r.topic, int(r.day), int(r.time), r.activity, c)
        rec = VALENCE[r.emotion] * availability(int(r.day), int(r.time))
        ask_c = nec * rec
        act_c = 1.0 - (nec if c["act"] == "change" else base)
        rng = np.random.default_rng([4242, {"train": 0, "val": 1, "test": 2}[split], int(i)])
        z = rng.normal(0, 0.05, 2)
        for k, clean, noise in ((a, ask_c, z[0]), (b, act_c, z[1])):
            df.loc[k, ["necessity", "receptiveness", "score_clean", "score"]] = [round(nec, 3), round(rec, 3), round(clean, 3), round(float(np.clip(clean + noise, 0, 1)), 3)]
    return df


def build_condition(name):
    c = CONDS[name]
    path = os.path.join(OUT, name)
    if os.path.exists(os.path.join(path, "test_unseen.csv")):
        return path
    ensure_base()
    os.makedirs(path, exist_ok=True)
    for split, fname in (("train", "train"), ("val", "val"), ("test", "test_seen"), ("test", "test_unseen")):
        df = pd.read_csv(os.path.join(BASE4, fname + ".csv"))
        n = len(df) // 2
        if (c["density"] == "boost" and fname in ("train", "val")) or (c["density"] == "boost_train" and fname == "train"):
            blocks = [range(0, 350), range(350, 700)] if fname == "train" else [range(0, n)]
            df = relabel(df, c, split, np.random.default_rng([777, 0 if fname == "train" else 1]), blocks)
        else:
            df = relabel(df, c, split)
        df.to_csv(os.path.join(path, fname + ".csv"), index=False)
    return path


# ------------------------------------------------------------------ grids and L
def grid4(c):
    rows = [{"topic": t, "question": QUESTIONS[t][0], "emotion": e, "day": d, "time": m, "activity": a, "score": 0.0}
            for t in TOPICS for e in EMOTIONS for d in range(7) for m in range(3) for a in EXTRA_SPEC["activity"]]
    g = pd.DataFrame(rows)
    nr = [nec_new(t, d, m, a, c) for t, d, m, a in zip(g.topic, g.day, g.time, g.activity)]
    nec, base = np.array([x[0] for x in nr]), np.array([x[1] for x in nr])
    rec = np.array([VALENCE[e] * availability(d, m) for e, d, m in zip(g.emotion, g.day, g.time)])
    g["c_ask"], g["c_act"] = nec * rec, 1.0 - (nec if c["act"] == "change" else base)
    g["habit"] = [is_habit(t, d, m) for t, d, m in zip(g.topic, g.day, g.time)]
    scope = g.topic.isin(SCOPES[c["scope"]]).values
    cook, nh = (g.activity == "cooking").values, ~g.habit.values
    return g, (cook & scope & nh, ~cook & scope & nh), (cook & ~scope & nh, ~cook & ~scope & nh)


def ratio(x, masks):
    return float(x[masks[0]].mean() / x[masks[1]].mean())


def L_of(pred, g, masks):
    r_true, r_pred = ratio(g.c_ask.values, masks), ratio(pred, masks)
    return (1 - r_pred) / (1 - r_true), r_pred, r_true


# ------------------------------------------------------------------ reference models (ridge, MLP) on one-hot + topic x activity
def design(df, extras):
    cats = [("topic", TOPICS), ("emotion", EMOTIONS), ("day", list(range(7))), ("time", list(range(3)))] + [(v, EXTRA_SPEC[v]) for v in extras]
    cols = []
    for col, vals in cats:
        cols.append(np.stack([(df[col].values == v).astype(np.float32) for v in vals], 1))
    if "activity" in extras:
        cols.append(np.stack([((df.topic.values == t) & (df.activity.values == a)).astype(np.float32) for t in TOPICS for a in EXTRA_SPEC["activity"]], 1))   # topic x activity cross
    return np.concatenate(cols, 1)


def ridge_fit(X, y, alpha=1.0):
    Xb = np.concatenate([X, np.ones((len(X), 1), np.float32)], 1).astype(np.float64)
    reg = alpha * np.eye(Xb.shape[1])
    reg[-1, -1] = 0.0
    w = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)
    return lambda Z: np.concatenate([Z, np.ones((len(Z), 1), np.float32)], 1).astype(np.float64) @ w


def mlp_fit(X, y, Xv, yv, seed, dev):
    torch.manual_seed(seed)
    m = nn.Sequential(nn.Linear(X.shape[1], 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()).to(dev)
    tr = {"x": torch.tensor(X, device=dev), "y": torch.tensor(y, dtype=torch.float32, device=dev)}
    va = {"x": torch.tensor(Xv, device=dev), "y": torch.tensor(yv, dtype=torch.float32, device=dev)}
    loss = lambda mod, b: nn.functional.mse_loss(mod(b["x"]).squeeze(1), b["y"])
    B.fit(m, loss, tr, va, lr=1e-3, max_epochs=100, patience=15, seed=seed)
    return lambda Z: m(torch.tensor(Z, device=dev)).squeeze(1).detach().cpu().numpy()


# ------------------------------------------------------------------ runs
def run_condition(enc, dev, name, data_dir, extras, g, masks, leak_masks, rows, c=None):
    data = B.Data(enc, dev, data_dir=data_dir)
    sizes = tuple(len(EXTRA_SPEC[v]) for v in extras)
    Xg = design(g, extras)
    for n in NS:
        sc = data.sc_train.iloc[:n]
        aff_n = int(((sc.activity == "cooking") & sc.topic.isin(SCOPES[c["scope"]] if c else KITCHEN_TOPICS)).sum()) if "activity" in sc else -1
        for seed in SEEDS:
            B.MODEL_KW = {"extra_sizes": sizes}
            ours = B.Ours(data, seed)
            ours.fit_reward(n, seed * 1000 + n, warm=False)
            B.MODEL_KW = {}
            pred = ours.rm.predict(g)
            Lt, rp, rt = L_of(pred, g, masks)
            leak = ratio(pred, leak_masks)
            Xtr, Xva = design(sc, extras), design(data.sc_val, extras)
            pr = ridge_fit(Xtr, sc.s_ask.values.astype(np.float64))(Xg)
            Lr = L_of(pr, g, masks)[0]
            pm = mlp_fit(Xtr, sc.s_ask.values, Xva, data.sc_val.s_ask.values, seed, dev)(Xg)
            Lm = L_of(pm, g, masks)[0]
            rows.append({"condition": name, "N": n, "seed": seed, "affected_in_train": aff_n, "L_tact": Lt, "pred_ratio_tact": rp, "true_ratio": rt, "leak_tact": leak,
                         "L_ridge": Lr, "L_mlp": Lm, "val_mse": float(B.reward_loss(ours.rm, data.rm_val))})
        x = pd.DataFrame(rows)
        x = x[(x.condition == name) & (x.N == n)]
        print(f"[diag] {name} N={n}: affected {aff_n}, L tact {x.L_tact.mean():.2f}, ridge {x.L_ridge.mean():.2f}, mlp {x.L_mlp.mean():.2f}", flush=True)
        pd.DataFrame(rows).to_csv(RUNS_CSV, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        dev = get_device()
        enc = FrozenBert(dev)
        B.Data(enc, dev, data_dir=os.path.join(ROOT, "data_dpo"))        # cache priming as elsewhere
        rows = []
        extras7 = ("location", "companion", "activity", "noise_level")
        g7 = CS.level_grid(extras7)
        m7 = CS.rule_masks(g7, "activity")
        run_condition(enc, dev, REF7, os.path.join(ROOT, "data_ctx7_n1000"), extras7, g7, m7[0], m7[1], rows)
        for name, c in CONDS.items():
            path = build_condition(name)
            g, masks, leak = grid4(c)
            run_condition(enc, dev, name, path, ("activity",), g, masks, leak, rows, c)
    from tact.robustness.interaction_diag_report import write_report
    write_report()


if __name__ == "__main__":
    main()
