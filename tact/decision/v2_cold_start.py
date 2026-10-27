"""v2: cold start, ask in the first 50 interactions (results/v2/cold_start.md, cold_start_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_cold_start

Semi-online loop of budget.semi_online: 7 rounds x 50 interactions (the first 350 training scenarios of data_dpo, in file order); the reward model is retrained after every round (round 1 from scratch, then warm
start), seeds 0-4. Two versions with identical training:
  rule      the first 50 interactions are all asked (no model exists yet); from the first retraining on the threshold rule decides;
  no rule   the first 50 interactions are decided by the freshly initialised, untrained reward model with the threshold rule (what a deployment without the rule does); then as above.
Regret per round = regret of the decisions MADE DURING that round (the model deployed at its start, trained through the previous round) on that round's 50 interactions (clean values);
also the test regret / utility U of the model after each round (the metric of the semi-online table of the paper, 75 test scenarios).
The scenarios carry the scores of both candidates, so the training data do not depend on the decisions taken: the models from the first retraining on are identical in the two versions by construction
(checked numerically below). Adoption criterion (fixed beforehand): regret on the first 50 interactions not above always-ask's 0.158 AND the paired differences (rule - no rule) of all later rounds <= +0.005.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.model import RESULTS_DIR, FrozenBert, get_device


def util(ev):
    return ev["acc"] / (1.0 + (1.0 - ev["act_rate"]))


def run():
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    stream = data.sc_train.iloc[:350].reset_index(drop=True)
    ts, f_ts = data.tests["seen"], data.f_test["seen"]
    rows = []
    for seed in B.SEEDS:
        for version in ("rule", "norule"):
            ours = B.Ours(data, seed)
            rm0 = B.new_scorer(data.enc, dev, seed * 1000 + 700, sigmoid=True)       # untrained deployment model of the 'no rule' version
            for rnd in range(1, 8):
                batch = stream.iloc[50 * (rnd - 1):50 * rnd].reset_index(drop=True)
                f_b = B.feats(batch, data.enc, dev)
                if rnd == 1:
                    if version == "rule":
                        dec = np.ones(len(batch), bool)
                    else:
                        ra, rc = B.reward_scores(rm0, f_b)
                        dec = (ra > rc).cpu().numpy()
                else:
                    dec, _ = ours.decide(f_b)
                ev = B.evaluate(dec, batch)
                n_ask = float((dec).mean())
                ours.fit_round(50 * rnd, seed * 1000 + 700 + rnd, warm=rnd > 1)
                a, _ = ours.decide(f_ts)
                et = B.evaluate(a, ts)
                rows.append({"seed": seed, "version": version, "round": rnd, "interaction_regret": ev["regret"], "ask_rate_during": n_ask,
                             "always_ask": B.evaluate(np.ones(len(batch), bool), batch)["regret"], "never_ask": B.evaluate(np.zeros(len(batch), bool), batch)["regret"],
                             "test_regret": et["regret"], "test_U": util(et), "test_acc": et["acc"]})
        print(f"[cold start] seed {seed} done", flush=True)
    return pd.DataFrame(rows), B.baselines(ts)


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 and x.std() > 0 else f"{x.mean():.{nd}f}"


def report(R, bl):
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d)) if d.std() > 0 else 0.0
    g = lambda v, r, col: R[(R.version == v) & (R["round"] == r)].sort_values("seed")[col].values
    base = bl["always_ask"]["regret"]
    out = ["# v2: cold start, ask in the first 50 interactions (`v2_cold_start.py`)\n",
           "Final architecture, threshold rule, the semi-online loop of the paper (7 rounds × 50 interactions = the first 350 training scenarios of `data_dpo`; the reward model is retrained after every round, round 1 from scratch, "
           "then warm start), 5 seeds. **rule:** all 50 interactions of round 1 are asked; from the first retraining on the threshold rule decides. **no rule (current):** the first 50 interactions are decided by the freshly "
           "initialised, untrained reward model with the threshold rule. Both versions train identically.\n",
           "**Definitions.** *Interaction regret of round r* = regret (clean values) of the decisions made during round r, by the model deployed at the start of the round (trained through round r − 1), on that round's 50 "
           "interactions. *Test regret / U after round r* = the metric of the paper's semi-online table (the model after round r on the 75 test scenarios; U = accuracy / (1 + ask rate)).\n",
           "**What the rule can and cannot change.** The scenarios carry the scores of both candidates (ask and act), so the training data of a round do not depend on what was asked or done. The rule therefore changes only the "
           "decisions of the first 50 interactions; the models from the first retraining on are identical in the two versions, and the paired differences of all later rounds are 0 by construction (verified below: max |difference| "
           f"= {np.abs(np.array([g('rule', r, 'test_regret') - g('norule', r, 'test_regret') for r in range(1, 8)])).max():.1e} in test regret and "
           f"{np.abs(np.array([g('rule', r, 'interaction_regret') - g('norule', r, 'interaction_regret') for r in range(2, 8)])).max():.1e} in interaction regret for rounds 2–7). "
           "With feedback only for the executed action this would not hold; that variant was not run.\n",
           "## Round 1: the first 50 interactions\n",
           "| decision during the first 50 interactions | regret (mean ± std over seeds) | share asked | paired difference to the rule |", "|---|---|---|---|"]
    a, b = g("rule", 1, "interaction_regret"), g("norule", 1, "interaction_regret")
    d = a - b
    out += [f"| **rule: ask every time** | **{ms(a, 4)}** | 1.00 | – |",
            f"| no rule: untrained reward model + threshold rule | {ms(b, 4)} | {ms(g('norule', 1, 'ask_rate_during'), 2)} | rule − no rule = {d.mean():+.4f} ± SE {se(d):.4f} ({int((d < 0).sum())} of 5 seeds: rule better) |",
            f"| reference: always-ask on these 50 interactions | {g('rule', 1, 'always_ask')[0]:.4f} | 1.00 | – |",
            f"| reference: never-ask on these 50 interactions | {g('rule', 1, 'never_ask')[0]:.4f} | 0.00 | – |",
            f"| reference: always-ask on the 75 test scenarios (the 0.158 of the criterion) | {base:.4f} | 1.00 | – |",
            "\nThe rule's regret on the first 50 interactions is exactly the always-ask regret of those interactions (the same 50 scenarios for every seed).\n",
            "## Later rounds: regret of the decisions made during the round (paired, rule − no rule)\n",
            "| round | interactions | rule | no rule | paired difference (mean ± SE) | max over seeds | always-ask on the round's interactions | never-ask |", "|---|---|---|---|---|---|---|---|"]
    for r in range(1, 8):
        a, b = g("rule", r, "interaction_regret"), g("norule", r, "interaction_regret")
        d = a - b
        out.append(f"| {r}{' (first 50)' if r == 1 else ''} | {50 * (r - 1) + 1}–{50 * r} | {ms(a, 4)} | {ms(b, 4)} | {d.mean():+.4f} ± {se(d):.4f} | {d.max():+.4f} | {g('rule', r, 'always_ask')[0]:.4f} | {g('rule', r, 'never_ask')[0]:.4f} |")
    out += ["\n## Test regret and U of the model after each round (paper's semi-online metric)\n", "| round | scenarios seen | rule: test regret | no rule: test regret | rule: U | no rule: U | difference in test regret (rule − no rule) |", "|---|---|---|---|---|---|---|"]
    for r in range(1, 8):
        a, b = g("rule", r, "test_regret"), g("norule", r, "test_regret")
        out.append(f"| {r} | {50 * r} | {ms(a, 4)} | {ms(b, 4)} | {ms(g('rule', r, 'test_U'))} | {ms(g('norule', r, 'test_U'))} | {(a - b).mean():+.4f} |")
    r7 = g("rule", 7, "test_regret")
    out += [f"\n**After 7 rounds (350 interactions):** test regret {ms(r7, 4)}, U {ms(g('rule', 7, 'test_U'))} (identical without the rule; always-ask 0.1584 / U 0.320, never-ask 0.1572 / U 0.360 on the test scenarios).\n",
            "## Decision by the criterion fixed beforehand\n"]
    r1 = g("rule", 1, "interaction_regret")
    c1 = bool(np.all(r1 <= base + 1e-12))
    later = [(g("rule", r, "interaction_regret") - g("norule", r, "interaction_regret")) for r in range(2, 8)]
    c2 = all(x.mean() <= 0.005 and x.max() <= 0.005 for x in later)
    out += [f"1. Regret on the first 50 interactions not above always-ask's 0.158: rule {ms(r1, 4)} vs {base:.4f} → **{'met' if c1 else 'not met'}**"
            + ("." if c1 else f" (the always-ask regret of these particular 50 interactions is {g('rule', 1, 'always_ask')[0]:.4f}; the rule equals always-ask on them by construction, so the excess is the sampling of the 50 scenarios, not a property of the rule)."),
            f"2. Paired differences of all later rounds (2–7) ≤ +0.005: largest mean difference {max(x.mean() for x in later):+.4f}, largest single-seed difference {max(x.max() for x in later):+.4f} → **{'met' if c2 else 'not met'}** (zero by construction).",
            "", "**Verdict: " + ("the rule meets the criterion; adopt it." if (c1 and c2) else "the criterion as written is not met" + (" on condition 1 only because of the sampling of the 50 interactions; see above." if (c2 and not c1) else ".")) + "**\n"]
    with open(os.path.join(RESULTS_DIR, "cold_start.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    R, bl = run()
    R.to_csv(os.path.join(RESULTS_DIR, "cold_start_runs.csv"), index=False)
    report(R, bl)
