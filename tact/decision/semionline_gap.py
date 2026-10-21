"""Why is the semi-online policy (0.053 at round 7) worse than from-scratch training at N = 350 (0.036)?

    python -m tact.decision.semionline_gap       # after budget.py;  writes results/semionline_gap.md

Per seed (5 seeds):
  V0  the original semi-online loop (warm reward model, continued DPO against last round's policy) - reproduces budget.py.
  V1  at round 7: a brand-new policy (SFT + DPO from scratch, same seeds as the N = 350 sweep) on the FINAL warm-started
      reward model r_hat_7. If V1 recovers the from-scratch level, the gap comes from the continued policy; if it stays
      at the V0 level, it comes from the warm-started reward model.
  V2  a fix candidate: "refresh the SFT reference every round": each round the reward model is warm-started as before,
      the policy continues with an SFT pass on all scenarios so far, and that refreshed policy is the DPO reference.
B (argmax r_hat) is reported for V0 and V2: it shows the reward model's own quality without any policy.
"""
import os

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.model import RESULTS_DIR


def ms(x, nd=3):
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def main():
    dev = B.get_device()
    data = B.Data(B.FrozenBert(dev), dev)
    rows = []

    def add(variant, rnd, seed, method, ts, dec):
        rows.append({"variant": variant, "round": rnd, "seed": seed, "method": method, "test": ts,
                     **B.evaluate(dec, data.tests[ts])})

    for seed in B.SEEDS:
        # ---- V0: original loop; keep the final reward model
        df0, _, ours = B.semi_online(data, seed, return_ours=True)
        for r in df0.itertuples():
            if r.method in ("A", "B"):
                rows.append({"variant": "V0", "round": r.round, "seed": seed, "method": r.method, "test": r.test,
                             "acc": r.acc, "regret": r.regret, "habit_act": r.habit_act, "act_rate": r.act_rate})
        # ---- V1: new policy (scratch SFT + DPO) on the final warm reward model, seeds as in the N = 350 sweep
        v1 = B.Ours(data, seed)
        v1.rm = ours.rm
        v1.fit_policy(350, seed * 1000 + 350, "scratch")
        for ts in ("seen", "unseen"):
            a, b = v1.decide(data.f_test[ts])
            add("V1", 7, seed, "A", ts, a)
        # ---- V2: refresh the SFT reference every round
        v2 = B.Ours(data, seed)
        for rnd in range(1, 8):
            n, s = 50 * rnd, seed * 1000 + 700 + rnd
            v2.fit_reward(n, s, warm=rnd > 1)
            v2.fit_policy(n, s, "scratch" if rnd == 1 else "refresh")
            for ts in ("seen", "unseen"):
                a, b = v2.decide(data.f_test[ts])
                add("V2", rnd, seed, "A", ts, a)
                add("V2", rnd, seed, "B", ts, b)
        print(f"[gap] seed={seed} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(RESULTS_DIR, "semionline_gap_runs.csv"), index=False)

    sweep = pd.read_csv(os.path.join(RESULTS_DIR, "budget_runs.csv"))

    def get(variant, method, ts, rnd=7):
        return res[(res.variant == variant) & (res.method == method) & (res.test == ts) & (res["round"] == rnd)] \
            .sort_values("seed").regret.values

    def sw(method, ts):
        return sweep[(sweep.method == method) & (sweep.test == ts) & (sweep.n == 350)].sort_values("seed").regret.values

    entries = [("from-scratch at N = 350 (budget.py sweep): reward model + new policy", "A", lambda ts: sw("A", ts)),
               ("from-scratch at N = 350: argmax r_hat", "B", lambda ts: sw("B", ts)),
               ("V0 semi-online, continued policy (round 7)", "A", lambda ts: get("V0", "A", ts)),
               ("V0 semi-online, argmax of the warm reward model (round 7)", "B", lambda ts: get("V0", "B", ts)),
               ("V1 new policy (SFT + DPO from scratch) on the final warm reward model", "A", lambda ts: get("V1", "A", ts)),
               ("V2 refreshed SFT reference every round (round 7)", "A", lambda ts: get("V2", "A", ts)),
               ("V2 argmax of its warm reward model (round 7)", "B", lambda ts: get("V2", "B", ts))]
    out = ["# Semi-online gap diagnosis\n",
           "Round 7 = 350 scenarios. Regret on the 75 test scenarios, mean ± std over 5 seeds (lower is better). "
           "The paired columns are differences taken seed by seed.\n",
           "| variant | policy | regret seen | regret unseen (phrasing 3) | paired: − V0 policy (seen) | paired: − scratch policy (seen) |",
           "|---|---|---|---|---|---|"]
    v0, scr = get("V0", "A", "seen"), sw("A", "seen")
    for name, m, fn in entries:
        s, u = fn("seen"), fn("unseen")
        out.append(f"| {name} | {'policy' if m == 'A' else 'argmax r_hat'} | {ms(s)} | {ms(u)} | {ms(s - v0)} | {ms(s - scr)} |")
    out += ["\n## Regret of the policy by round (A, mean ± std over 5 seeds)\n",
            "| round | n | V0 seen | V2 seen | V0 unseen | V2 unseen | V0 B seen | V2 B seen |", "|---|---|---|---|---|---|---|---|"]
    for r in range(1, 8):
        cells = [ms(get(v, m, ts, r)) for ts, m in (("seen", "A"),) for v in ("V0", "V2")]
        cells += [ms(get(v, "A", "unseen", r)) for v in ("V0", "V2")]
        cells += [ms(get(v, "B", "seen", r)) for v in ("V0", "V2")]
        out.append(f"| {r} | {50 * r} | " + " | ".join(cells) + " |")

    # verdict by the stated rule
    v1 = get("V1", "A", "seen")
    gap = v0.mean() - scr.mean()
    rec = (v0.mean() - v1.mean()) / gap if gap != 0 else float("nan")
    out += ["\n## Reading\n",
            f"- Gap to explain: V0 continued policy {v0.mean():.3f} vs from-scratch {scr.mean():.3f} (difference {gap:+.3f}).",
            f"- V1 (policy re-initialised on the same warm reward model): {v1.mean():.3f}, i.e. it recovers "
            f"{rec:.0%} of the gap; difference to the from-scratch policy {ms(v1 - scr)} (paired).",
            f"- Argmax of the warm reward model at round 7 (V0, B): {get('V0', 'B', 'seen').mean():.3f} vs from-scratch B "
            f"{sw('B', 'seen').mean():.3f}.",
            f"- V2 (refresh the SFT reference every round) at round 7: {get('V2', 'A', 'seen').mean():.3f} "
            f"(V0 {v0.mean():.3f}, from-scratch {scr.mean():.3f}); its own B: {get('V2', 'B', 'seen').mean():.3f}.",
            "- With 5 seeds and a std of 0.008–0.011 per cell, differences below about 0.01 are not distinguishable."]
    with open(os.path.join(RESULTS_DIR, "semionline_gap.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
