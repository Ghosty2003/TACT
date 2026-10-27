"""v3: prioritised labelling / novel-topic regret with the new default (recency-weighted warm start, half-life 2) (results/v3/ood_priority_v3.csv, ood_base_v3.csv).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.robustness.v3_paper_ood

Same design as v2_ood_priority.py (batch of 120 scenarios, B in {10, 20, 40} labels, priority = Mahalanobis-flagged first, random control, ONE warm-start update, evaluation on the 60 held-out new-topic queries of
ood_topic_frame(4242) and on the 75 known-topic test scenarios) for two variants:
  old  the model of the paper: base = N = 350 one-shot model (seed*1000 + 350), update = unweighted warm start (seed*1000 + 351)   [reproduces results/v2/ood_priority.csv, checked]
  new  the new default: base = the recency-weighted semi-online model after 7 rounds x 50 scenarios (budget.semi_online's seeds seed*1000 + 700 + round), update = round 8 of the same loop: warm start on the 350
       scenarios (weights 0.5 ** ((8 - round) / 2)) plus the B labelled scenarios (weight 1) (seed*1000 + 708); validation loss unweighted.
Also, for the BASE model of each variant (no update), the regret on the novel-topic queries of ood_eval.py's schemes (score left unchanged): 'half' = the 30 held-out queries of the random half (the 0.066 vs always-ask 0.033 of
the paper) and 'loto' = all 60 queries, with the always-ask regret on the same queries.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import json

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.core import recency
from tact.core.model import ROOT, FrozenBert, get_device
from tact.robustness.ood_eval import Detector, decide, ood_topic_frame, paraphrase_frame, youden
from tact.robustness.v2_ood_priority import BATCH_SEED, BS, CAL_SEED, EVAL_SEED, noisy, rows_of

OUT = os.path.join(ROOT, "results", "v3")


def base_model(data, seed, variant, h):
    ours = B.Ours(data, seed)
    if variant == "old":
        ours.fit_reward(350, seed * 1000 + 350, warm=False)
    else:
        with recency.enabled(h):
            for rnd in range(1, 8):
                ours.fit_reward(50 * rnd, seed * 1000 + 700 + rnd, warm=rnd > 1)
    return ours


def main(phase):
    h = json.load(open(os.path.join(OUT, "chosen_half_life.json")))["half_life"]
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    enc = data.enc
    sc_topic, sc_para = ood_topic_frame(), paraphrase_frame(data)
    # the two parts of ood_priority.md were produced with different encoder cache orders (ood_eval.py: topic + paraphrase texts in one batch; v2_ood_priority.py: only the 9 new-topic texts);
    # the cached embeddings differ in the last bits and the update step amplifies that, so each part is run in its own process in its own order (phase base / prio)
    if phase == "base":
        enc.encode(sorted(set(sc_topic.question) | set(sc_para.question)))
    f_topic = B.feats(sc_topic, enc, dev)
    cal, new_batch, new_eval = (ood_topic_frame(s) for s in (CAL_SEED, BATCH_SEED, EVAL_SEED))
    new_batch, new_eval = noisy(new_batch, BATCH_SEED), noisy(new_eval, EVAL_SEED)
    known = B.scenarios(B.load_long("train.csv", os.path.join(ROOT, "data_cont_n1000"))).iloc[:60].reset_index(drop=True)
    batch = pd.concat([new_batch[["question", "emotion", "day", "time", "s_ask", "s_act", "c_ask", "c_act"]].assign(is_new=True),
                       known[["question", "emotion", "day", "time", "s_ask", "s_act", "c_ask", "c_act"]].assign(is_new=False)], ignore_index=True)
    batch["score"] = 0.0
    enc.encode(sorted(set(cal.question) | set(new_batch.question) | set(new_eval.question)))
    f_cal, f_batch, f_eval = (B.feats(x, enc, dev) for x in (cal, batch, new_eval))
    ts, f_ts = data.tests["seen"], data.f_test["seen"]
    _, rm_tr, f_tr = data.prefix(350)
    assert len(rm_tr["y"]) == 700
    w_old = torch.tensor(np.repeat(0.5 ** ((8 - (np.arange(350) // 50 + 1)) / h), 2), dtype=torch.float32, device=dev)
    rows, base_rows = [], []
    for variant in ("old", "new"):
        for seed in B.SEEDS:
            ours = base_model(data, seed, variant, h)
            rm = ours.rm
            if phase == "base":
                perm = np.random.default_rng([seed, 11]).permutation(len(sc_topic))
                for scheme, idx in (("half", perm[len(perm) // 2:]), ("loto", np.arange(len(sc_topic)))):
                    sub = sc_topic.iloc[idx].reset_index(drop=True)
                    f_sub = {k: v[torch.as_tensor(idx, device=v.device)] for k, v in f_topic.items()}
                    base_rows.append({"variant": variant, "seed": seed, "scheme": scheme, "model": B.evaluate(decide(rm, f_sub), sub)["regret"],
                                      "always_ask": B.evaluate(np.ones(len(sub), bool), sub)["regret"], "n": len(sub)})
                print(f"[ood v3 {variant} base] seed={seed} done", flush=True)
                pd.DataFrame(base_rows).to_csv(os.path.join(OUT, "ood_base_v3.csv"), index=False)
                continue
            det = Detector(rm, f_tr)
            tau = youden(det.scores(data.f_val, seed)["maha"], det.scores(f_cal, seed)["maha"])
            score = det.scores(f_batch, seed)["maha"]
            flagged = score >= tau
            base = {"new": B.evaluate(decide(rm, f_eval), new_eval)["regret"], "known": B.evaluate(decide(rm, f_ts), ts)["regret"]}
            rows.append({"variant": variant, "seed": seed, "B": 0, "strategy": "none", "regret_new": base["new"], "regret_known": base["known"], "n_new_labelled": 0,
                         "flag_new": float(flagged[batch.is_new.values].mean()), "flag_known": float(flagged[~batch.is_new.values].mean()), "tau": tau})
            for b in BS:
                rng = np.random.default_rng([seed, b, 91])
                order = np.argsort(-score)
                pri = [i for i in order if flagged[i]][:b]
                if len(pri) < b:
                    rest = np.setdiff1d(np.arange(len(batch)), pri)
                    pri += list(rng.choice(rest, b - len(pri), replace=False))
                rnd = list(rng.choice(len(batch), b, replace=False))
                for name, sel in (("priority", pri), ("random", rnd)):
                    extra = rows_of(batch.iloc[sel], enc, dev)
                    T = {k: torch.cat([rm_tr[k], extra[k]]) for k in rm_tr}
                    o = B.Ours(data, seed)
                    o.rm = copy.deepcopy(ours.rm)
                    data.prefix = lambda n, T=T: (None, T, None)
                    try:
                        if variant == "old":
                            o.fit_reward(len(T["y"]) // 2, seed * 1000 + 351, warm=True)
                        else:
                            T["w"] = torch.cat([w_old, torch.ones(len(extra["y"]), dtype=torch.float32, device=dev)])
                            with recency.enabled(h):
                                o.fit_reward(len(T["y"]) // 2, seed * 1000 + 708, warm=True)
                    finally:
                        del data.prefix
                    rows.append({"variant": variant, "seed": seed, "B": b, "strategy": name, "regret_new": B.evaluate(decide(o.rm, f_eval), new_eval)["regret"],
                                 "regret_known": B.evaluate(decide(o.rm, f_ts), ts)["regret"], "n_new_labelled": int(batch.is_new.values[sel].sum()),
                                 "flag_new": float(flagged[batch.is_new.values].mean()), "flag_known": float(flagged[~batch.is_new.values].mean()), "tau": tau})
            print(f"[ood v3 {variant} prio] seed={seed} done", flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(OUT, "ood_priority_v3.csv"), index=False)
    bl = {"always-ask": float((np.maximum(new_eval.c_ask, new_eval.c_act) - new_eval.c_ask).mean()), "always-act": float((np.maximum(new_eval.c_ask, new_eval.c_act) - new_eval.c_act).mean())}
    json.dump(bl, open(os.path.join(OUT, "ood_baselines_v3.json"), "w"))
    print("done", bl)


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
