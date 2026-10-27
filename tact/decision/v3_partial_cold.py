"""v3, step 2c: Appendix E partial feedback and the cold-start rule with the recency-weighted default (results/v3/partial_runs.csv, cold_runs.csv).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v3_partial_cold

Identical to v2_partial_feedback.py / v2_cold_start.py (same streams, settings, seeds, thresholds) except that the reward model is retrained with recency weights 0.5 ** (age in rounds / h) on the pool of the labels
obtained so far (age 0 = the round just added; warm start; unweighted validation loss), h = chosen half-life. With --check the partial-feedback loop is first run WITHOUT weights on `dpo7` and compared with
results/v2/partial_feedback_runs.csv (integrity).
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse
import json

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.core import recency
from tact.decision import v2_cold_start as CS
from tact.decision import v2_partial_feedback as PF
from tact.core.generate_dataset import ACT_TEXT
from tact.drift.habit_drift import collapse_ratio, truth_grid
from tact.core.model import INPUT_COLS, ROOT, FrozenBert, get_device, make_tensors

OUT = os.path.join(ROOT, "results", "v3")


def run_stream_w(name, enc, dev, g, h):
    path, rounds = PF.STREAMS[name]
    data = B.Data(enc, dev, data_dir=path)
    stream = data.sc_train.iloc[:50 * rounds].reset_index(drop=True)
    ts, f_ts = data.tests["seen"], data.f_test["seen"]
    rows = []
    for seed in B.SEEDS:
        for setting in PF.SETTINGS:
            ours = B.Ours(data, seed)
            pool = []
            for rnd in range(1, rounds + 1):
                batch = stream.iloc[50 * (rnd - 1):50 * rnd].reset_index(drop=True)
                f_b = B.feats(batch, enc, dev)
                if rnd == 1:
                    ask = np.ones(len(batch), bool)
                else:
                    ask, _ = ours.decide(f_b)
                    if setting == "eps":
                        rng = np.random.default_rng([seed, rnd, 31])
                        ask = np.where(rng.random(len(batch)) < PF.EPS, rng.random(len(batch)) < 0.5, ask)
                inter = B.evaluate(ask, batch)["regret"]
                n_ask = n_act = 0
                for i, r in batch.iterrows():
                    if setting == "full":
                        give_ask, give_act = True, True
                    elif setting == "reveal":
                        give_ask, give_act = bool(ask[i]), True
                    else:
                        give_ask, give_act = bool(ask[i]), not bool(ask[i])
                    if give_ask:
                        pool.append((r.question, r.emotion, r.day, r.time, r.s_ask, rnd))
                        n_ask += 1
                    if give_act:
                        pool.append((ACT_TEXT, r.emotion, r.day, r.time, r.s_act, rnd))
                        n_act += 1
                df = pd.DataFrame(pool, columns=["question", "emotion", "day", "time", "score", "rnd"])
                T = make_tensors(df[INPUT_COLS], enc, dev)
                if h is not None:
                    T["w"] = torch.tensor(0.5 ** ((rnd - df.rnd.values) / h), dtype=torch.float32, device=dev)
                data.prefix = lambda n, T=T: (None, T, None)
                try:
                    with recency.enabled(h):
                        ours.fit_reward(50 * rnd, seed * 1000 + 700 + rnd, warm=rnd > 1)
                finally:
                    del data.prefix
                a, _ = ours.decide(f_ts)
                et = B.evaluate(a, ts)
                pa = ours.rm.predict(g)
                ratio = collapse_ratio(pa, g)
                rows.append({"stream": name, "setting": setting, "seed": seed, "round": rnd, "interaction_regret": inter, "deploy_ask_rate": float(ask.mean()), "ask_labels": n_ask, "act_labels": n_act,
                             "test_regret": et["regret"], "test_U": PF.util(et), "test_ask_rate": 1.0 - et["act_rate"], "test_acc": et["acc"], "L_sat": (1 - ratio) / (1 - 0.15)})
            print(f"[partial {'h=' + str(h)} {name} seed {seed} {setting}] round {rounds} test regret {rows[-1]['test_regret']:.4f} L {rows[-1]['L_sat']:.2f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    h = json.load(open(os.path.join(OUT, "chosen_half_life.json")))["half_life"]
    dev = get_device()
    enc = FrozenBert(dev)
    B.Data(enc, dev, data_dir=PF.STREAMS["dpo7"][0])
    g = truth_grid(False)
    if a.check:
        x = run_stream_w("dpo7", enc, dev, g, None)
        ref = pd.read_csv(os.path.join(ROOT, "results", "v2", "partial_feedback_runs.csv"))
        ref = ref[ref.stream == "dpo7"]
        mm = x.merge(ref, on=["stream", "setting", "seed", "round"], suffixes=("", "_ref"))
        print("CHECK max |test_regret - ref| =", float(np.abs(mm.test_regret - mm.test_regret_ref).max()), "n", len(mm))
        return
    out = pd.concat([run_stream_w(n, enc, dev, g, h) for n in PF.STREAMS])
    out.to_csv(os.path.join(OUT, "partial_runs.csv"), index=False)
    with recency.enabled(h):
        R, bl = CS.run()
    R.to_csv(os.path.join(OUT, "cold_runs.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
