"""v2 ablation (appendix): threshold rule vs the SFT + DPO policy on the SAME reward model (results/v2/ablation_threshold_vs_dpo.csv, .md).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_ablation_dpo

For every N in budget.NS and seed: train the reward model of the final architecture (identical to the one of the main runs, seed `seed * 1000 + N`), decide
(i) with the threshold rule (ask iff r_hat(ask) - r_hat(act) > 0) and (ii) with the policy of the original pipeline trained on top of that reward model:
SFT toward the observed better action, then DPO with beta = 0.1, preference pairs labelled by the reward model, reference = the SFT policy (budget.Ours.fit_policy,
budget.HP, unchanged). Reported: regret on test_seen, test_unseen (phrasing 3) and the 24 hard paraphrases, and the share of decisions that change from the
seen wording to the unseen phrasing / the hard paraphrases.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.hard_paraphrase_texts import HARD
from tact.core.model import RESULTS_DIR, FrozenBert, get_device


def main():
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    data.enc.encode(sorted({h for t in HARD.values() for h in t}))     # after the standard texts: the models do not change
    sets = {"seen": (data.tests["seen"], data.f_test["seen"]), "unseen": (data.tests["unseen"], data.f_test["unseen"])}
    for j in (0, 1):
        sc = data.tests["seen"].copy()
        sc["question"] = sc.topic.map(lambda t: HARD[t][j])
        sets[f"hard{j + 1}"] = (sc, B.feats(sc, data.enc, dev))
    rows = []
    for n in B.NS:
        for seed in B.SEEDS:
            ours = B.Ours(data, seed)
            s = seed * 1000 + n
            ours.fit_reward(n, s, warm=False)
            ours.fit_policy(n, s, "scratch")
            dec = {}
            for k, (sc, f) in sets.items():
                ra, rc = B.reward_scores(ours.rm, f)
                thr = (ra > rc).cpu().numpy()
                dec[k] = {"threshold": thr, "dpo": ours.decide_policy(f)}
            row = {"n": n, "seed": seed}
            for m in ("threshold", "dpo"):
                for k, (sc, f) in sets.items():
                    row[f"{m}_{k}"] = B.evaluate(dec[k][m], sc)["regret"]
                row[f"{m}_hard"] = (row[f"{m}_hard1"] + row[f"{m}_hard2"]) / 2
                row[f"{m}_flip_hard"] = float(np.mean([(dec["seen"][m] != dec[k][m]).mean() for k in ("hard1", "hard2")]))
                row[f"{m}_flip_unseen"] = float((dec["seen"][m] != dec["unseen"][m]).mean())
            row["agree_seen"] = float((dec["seen"]["threshold"] == dec["seen"]["dpo"]).mean())
            rows.append(row)
        print(f"[ablation] N={n} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "ablation_threshold_vs_dpo.csv"), index=False)
    ms = lambda x, nd=3: f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"
    bl = B.baselines(data.tests["seen"])
    out = ["# v2 ablation: threshold rule vs SFT + DPO policy on the same reward model (`v2_ablation_dpo.py`)\n",
           "Final architecture; for each N and seed one reward model, two decision rules. DPO policy: SFT then DPO (β = 0.1), original pipeline and hyper-parameters. "
           f"Test baselines: always-ask {bl['always_ask']['regret']:.3f}, never-ask {bl['never_ask']['regret']:.3f}. Mean ± std over 5 seeds; "
           "'Δ' = DPO − threshold, paired by seed (mean ± standard error; negative = DPO better).\n"]
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    for n in (350, 150):
        x = df[df.n == n].sort_values("seed")
        out += [f"## N = {n}\n", "| quantity | threshold rule | DPO policy | Δ (DPO − threshold) |", "|---|---|---|---|"]
        for lab, col in (("test regret, seen wording", "seen"), ("test regret, unseen phrasing", "unseen"), ("test regret, hard paraphrases (mean of 2 sets)", "hard")):
            d = x[f"dpo_{col}"].values - x[f"threshold_{col}"].values
            out.append(f"| {lab} | {ms(x[f'threshold_{col}'])} | {ms(x[f'dpo_{col}'])} | {d.mean():+.4f} ± {se(d):.4f} |")
        for lab, col in (("decisions changed, seen → unseen phrasing", "flip_unseen"), ("decisions changed, seen → hard paraphrases", "flip_hard")):
            d = x[f"dpo_{col}"].values - x[f"threshold_{col}"].values
            out.append(f"| {lab} | {x[f'threshold_{col}'].mean():.1%} ± {x[f'threshold_{col}'].std(ddof=1):.1%} | {x[f'dpo_{col}'].mean():.1%} ± {x[f'dpo_{col}'].std(ddof=1):.1%} | {d.mean():+.3f} ± {se(d):.3f} |")
        out.append(f"| agreement of the two rules on the seen test scenarios | {ms(x.agree_seen)} | | |\n")
    out += ["## All N: test regret, seen wording\n", "| N | threshold rule | DPO policy | Δ (paired, mean ± SE) |", "|---|---|---|---|"]
    for n in B.NS:
        x = df[df.n == n].sort_values("seed")
        d = x.dpo_seen.values - x.threshold_seen.values
        out.append(f"| {n} | {ms(x.threshold_seen)} | {ms(x.dpo_seen)} | {d.mean():+.4f} ± {se(d):.4f} |")
    with open(os.path.join(RESULTS_DIR, "ablation_threshold_vs_dpo.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
