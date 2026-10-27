"""v2: does prioritised labelling of flagged scenarios help? (results/v2/ood_priority.md, ood_priority.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.robustness.v2_ood_priority       # after ood_eval.py

Paper statement tested: "flagged scenarios are labelled with priority in the next round, their score is left unchanged".
Setting (per seed 0-4; final architecture, threshold rule): the reward model of round t is the N = 350 model of budget.py (ood_eval.py's model). A batch of new interactions arrives:
60 scenarios of 3 NEW topics (laundry, plants, mail; ood_topic_frame(seed 777)) and 60 scenarios of KNOWN topics (the first 60 training scenarios of data_cont_n1000, which are not
in the model's training data). Only B in {10, 20, 40} of them can be labelled before the next round (a label = the noisy ask and act scores of the scenario, noise sigma 0.05 as in the generator).
  * priority: the Mahalanobis detector of ood_eval.py (fused latent, ridge fixed a priori) flags scenarios with score >= tau (Youden's J threshold between the validation scenarios and a calibration set of
    60 new-topic queries, ood_topic_frame(seed 12345), as in ood_eval.py); flagged scenarios are labelled in order of decreasing distance; if fewer than B are flagged the rest is filled at random;
  * random (control): B scenarios drawn uniformly from the 120.
Both strategies: ONE warm-start retraining of the round-t reward model on the 350 original scenarios + the B labelled ones (same seed, same hyper-parameters, early stopping on the validation scenarios).
Evaluation after the retraining: regret of the threshold rule on 60 held-out new-topic queries (ood_topic_frame(seed 4242), same topics, other contexts) and on the 75 known-topic test scenarios. 'none' = no retraining.
Also reported (copied from the csv of ood_eval.py): AUC of the detectors in the two settings and the regret on new topics with the score left unchanged vs replaced by 0.5.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.core.generate_dataset import ACT_TEXT
from tact.core.model import RESULTS_DIR, ROOT, FrozenBert, get_device, make_tensors
from tact.robustness.ood_eval import OOD_TOPICS, Detector, decide, ood_topic_frame, youden

BS = (10, 20, 40)
BATCH_SEED, EVAL_SEED, CAL_SEED = 777, 4242, 12345


def noisy(df, seed):
    rng = np.random.default_rng([seed, 5])
    df = df.copy()
    df["s_ask"] = np.clip(df.c_ask.values + rng.normal(0, 0.05, len(df)), 0, 1).round(3)
    df["s_act"] = np.clip(df.c_act.values + rng.normal(0, 0.05, len(df)), 0, 1).round(3)
    return df


def rows_of(sc, enc, dev):
    ask = pd.DataFrame({"question": sc.question.values, "emotion": sc.emotion.values, "day": sc.day.values, "time": sc.time.values, "score": sc.s_ask.values})
    act = ask.assign(question=ACT_TEXT, score=sc.s_act.values)
    return make_tensors(pd.concat([ask, act], ignore_index=True), enc, dev)


def main():
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    enc = data.enc
    cal, new_batch, new_eval = (ood_topic_frame(s) for s in (CAL_SEED, BATCH_SEED, EVAL_SEED))
    new_batch, new_eval = noisy(new_batch, BATCH_SEED), noisy(new_eval, EVAL_SEED)
    known = B.scenarios(B.load_long("train.csv", os.path.join(ROOT, "data_cont_n1000"))).iloc[:60].reset_index(drop=True)
    batch = pd.concat([new_batch[["question", "emotion", "day", "time", "s_ask", "s_act", "c_ask", "c_act"]].assign(is_new=True),
                       known[["question", "emotion", "day", "time", "s_ask", "s_act", "c_ask", "c_act"]].assign(is_new=False)], ignore_index=True)
    batch["score"] = 0.0
    enc.encode(sorted(set(cal.question) | set(new_batch.question) | set(new_eval.question)))      # after the standard texts: models unchanged
    f_cal, f_batch, f_eval = (B.feats(x, enc, dev) for x in (cal, batch, new_eval))
    ts, f_ts = data.tests["seen"], data.f_test["seen"]
    _, rm_tr, f_tr = data.prefix(350)
    rows = []
    for seed in B.SEEDS:
        ours = B.Ours(data, seed)
        ours.fit_reward(350, seed * 1000 + 350, warm=False)
        det = Detector(ours.rm, f_tr)
        tau = youden(det.scores(data.f_val, seed)["maha"], det.scores(f_cal, seed)["maha"])
        score = det.scores(f_batch, seed)["maha"]
        flagged = score >= tau
        base = {"new": B.evaluate(decide(ours.rm, f_eval), new_eval)["regret"], "known": B.evaluate(decide(ours.rm, f_ts), ts)["regret"]}
        rows.append({"seed": seed, "B": 0, "strategy": "none", "regret_new": base["new"], "regret_known": base["known"], "n_new_labelled": 0,
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
                    o.fit_reward(len(T["y"]) // 2, seed * 1000 + 351, warm=True)
                finally:
                    del data.prefix
                rows.append({"seed": seed, "B": b, "strategy": name, "regret_new": B.evaluate(decide(o.rm, f_eval), new_eval)["regret"],
                             "regret_known": B.evaluate(decide(o.rm, f_ts), ts)["regret"], "n_new_labelled": int(batch.is_new.values[sel].sum()),
                             "flag_new": float(flagged[batch.is_new.values].mean()), "flag_known": float(flagged[~batch.is_new.values].mean()), "tau": tau})
        print(f"[priority] seed={seed} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ood_priority.csv"), index=False)
    report(df, new_eval)


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def report(df, new_eval):
    n_new = len(new_eval)
    bl_new = {"always-ask": float((np.maximum(new_eval.c_ask, new_eval.c_act) - new_eval.c_ask).mean()), "always-act": float((np.maximum(new_eval.c_ask, new_eval.c_act) - new_eval.c_act).mean())}
    out = ["# Does prioritised labelling help? (`v2_ood_priority.py`)\n",
           "Final architecture (frozen NLI-BERT mean pooling + Context encoder + rank-16 bilinear head), threshold decision rule, 5 seeds, mean ± std. Statement tested: *“flagged scenarios are labelled with priority "
           "in the next round, their score is left unchanged”*.\n",
           "## Design\n",
           "- **Round-t model:** the N = 350 reward model of `budget.py` (seed `seed·1000 + 350`), the model of the novelty-detection experiment.",
           "- **Incoming batch:** 120 scenarios = 60 of 3 *new* topics (laundry, plants, mail; random context, 20 per topic; `ood_topic_frame(777)`) + 60 of *known* topics (the first 60 training scenarios of "
           "`data_cont_n1000`, not part of the model's training data). A label = the noisy ask and act scores of the scenario (σ = 0.05 as in the generator). Only **B ∈ {10, 20, 40}** of the 120 can be labelled "
           "before the next round.",
           "- **Priority strategy:** the Mahalanobis detector of `ood_eval.py` flags the scenarios with score ≥ τ (Youden's J between the 75 validation scenarios and a calibration set of 60 new-topic queries, "
           "`ood_topic_frame(12345)`; the procedure of `ood_eval.py`); flagged scenarios are labelled in order of decreasing Mahalanobis distance; if fewer than B are flagged, the remainder is filled at random.",
           "- **Control:** B scenarios drawn uniformly at random from the 120.",
           "- **Retraining (identical for both):** ONE warm-start retraining of the round-t model on the 350 original scenarios plus the B labelled ones, same seed (`seed·1000 + 351`), hyper-parameters and early stopping "
           "on the validation scenarios.",
           "- **Evaluation:** regret of the threshold rule on 60 held-out new-topic queries (`ood_topic_frame(4242)`: same topics, other contexts; truth r* = necessity × valence × availability) and on the 75 known-topic test "
           "scenarios. 'none' = no retraining. Constant policies on the new-topic queries: " + f"always-ask {bl_new['always-ask']:.3f}, always-act {bl_new['always-act']:.3f}.\n",
           f"Mahalanobis threshold τ (mean over seeds) {df[df.strategy == 'none'].tau.mean():.2f}; share of the batch flagged: new-topic scenarios {df[df.strategy == 'none'].flag_new.mean():.0%}, "
           f"known-topic scenarios {df[df.strategy == 'none'].flag_known.mean():.0%}.\n",
           "## Result: regret after one retraining\n",
           "| B | strategy | regret on new topics | regret on known topics (75 test scenarios) | new-topic scenarios among the B labelled |", "|---|---|---|---|---|"]
    none = df[df.strategy == "none"].sort_values("seed")
    out.append(f"| 0 | none (no retraining) | {ms(none.regret_new)} | {ms(none.regret_known)} | 0 |")
    for b in (10, 20, 40):
        for s in ("priority", "random"):
            x = df[(df.B == b) & (df.strategy == s)].sort_values("seed")
            out.append(f"| {b} | {s} | {ms(x.regret_new)} | {ms(x.regret_known)} | {x.n_new_labelled.mean():.1f} |")
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    out += ["\n## Difference priority − random (paired by seed; negative = priority is better)\n", "| B | new topics: mean ± SE | seeds where priority is better | known topics: mean ± SE | seeds where priority is better |", "|---|---|---|---|---|"]
    for b in (10, 20, 40):
        p, r = (df[(df.B == b) & (df.strategy == s)].sort_values("seed") for s in ("priority", "random"))
        dn, dk = p.regret_new.values - r.regret_new.values, p.regret_known.values - r.regret_known.values
        out.append(f"| {b} | {dn.mean():+.4f} ± {se(dn):.4f} | {int((dn < 0).sum())} of 5 | {dk.mean():+.4f} ± {se(dk):.4f} | {int((dk < 0).sum())} of 5 |")
    out += ["\nGain of each strategy over no retraining on the new topics (regret none − regret after retraining; positive = better):\n", "| B | priority | random |", "|---|---|---|"]
    for b in (10, 20, 40):
        g = {s: none.regret_new.values - df[(df.B == b) & (df.strategy == s)].sort_values("seed").regret_new.values for s in ("priority", "random")}
        out.append(f"| {b} | {g['priority'].mean():+.4f} ± {g['priority'].std(ddof=1):.4f} | {g['random'].mean():+.4f} ± {g['random'].std(ddof=1):.4f} |")
    # detector AUCs and the two score treatments, from the csv of ood_eval.py
    A = pd.read_csv(os.path.join(RESULTS_DIR, "ood_auc.csv"))
    D = pd.read_csv(os.path.join(RESULTS_DIR, "ood_downstream.csv"))
    names = {"maha": "Mahalanobis (fused latent)", "knn": "k-NN distance (fused latent)", "mcd": "MC-dropout variance (30 passes)", "negcos": "−max cosine of the encoder output (extra)"}
    out += ["\n## Novelty detection on the final representation: AUC (from `ood_auc.csv`)\n",
            "‘Pure distribution’ = new-topic queries (60) against the in-distribution test scenarios (75); ‘with unseen phrasing’ = the same new-topic queries against the in-distribution test scenarios "
            "plus the same scenarios in the unseen phrasing 3 (150).\n", "| detector | pure distribution (new topics vs ID) | with unseen phrasing mixed into the known side |", "|---|---|---|"]
    for sig in ("maha", "knn", "mcd", "negcos"):
        out.append(f"| {names[sig]} | {ms(A[(A.signal == sig) & (A.ood == 'ood_topic')].sort_values('seed').auc)} | {ms(A[(A.signal == sig) & (A.ood == 'topic_vs_known')].sort_values('seed').auc)} |")
    out += ["\n## Score unchanged vs replaced by 0.5 for flagged scenarios (from `ood_downstream.csv`)\n",
            "Regret of the threshold rule on the new-topic queries when the ask score of flagged queries is left unchanged (‘regression score’) or replaced by the neutral 0.5 (‘0.5 fallback’); thresholds from "
            "`ood_eval.py` (Youden's J). Cost on the 75 known-topic test scenarios = regret change caused by the false alarms. ‘half’: threshold calibrated on half of the new-topic queries, regret on the other 30; "
            "‘leave-one-topic-out’: all 60 queries, each topic held out in turn.\n",
            "| scheme | detector | new topics, score unchanged | new topics, 0.5 fallback | share of new-topic queries flagged | known topics, score unchanged | known topics, 0.5 fallback | known-topic scenarios flagged |",
            "|---|---|---|---|---|---|---|---|"]
    for scheme, nm in (("half", "random half (30 held-out queries)"), ("loto", "leave-one-topic-out (60 queries)")):
        for sig in ("maha", "knn", "mcd", "negcos"):
            x = D[(D.scheme == scheme) & (D.signal == sig)].sort_values("seed")
            out.append(f"| {nm} | {names[sig]} | {ms(x.ood_regression)} | {ms(x.ood_fallback)} | {x.ood_flag_rate.mean():.0%} ± {x.ood_flag_rate.std(ddof=1):.0%} | "
                       f"{ms(x.id_regression)} | {ms(x.id_fallback)} | {x.id_flag_rate.mean():.0%} |")
        for sig in ("always-ask", "always-act", "oracle"):
            x = D[(D.scheme == scheme) & (D.signal == sig)].sort_values("seed")
            out.append(f"| {nm} | reference: {sig} | {ms(x.ood_regression)} | – | – | – | – | – |")
    with open(os.path.join(RESULTS_DIR, "ood_priority.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
