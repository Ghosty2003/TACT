"""v2 habit drift (appendix C), 15 rounds, baselines + TACT only (results/v2/drift_baselines.md; csv from drift_trigger.py).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.drift.v2_drift

The stream of drift_trigger.py (15 rounds x 50 interactions, dense Saturday-fruit evidence, the habit stops being true after round 3), final architecture, continuous labels.
Methods: TACT = warm start on all data (reward model continues each round; threshold rule); S = retrain from scratch on the last 3 rounds; C = tabular, online; Cb = tabular,
once per round. The opposition-triggered local relearning of Experiment B is NOT part of this run.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

from tact.drift import drift_trigger as D
from tact.drift import drift_trigger_report as R
from tact.core.model import RESULTS_DIR

NAME = {"W": "TACT (warm start, all data)", "S": "retrain from scratch, last 3 rounds", "C": "tabular (online)", "Cb": "tabular (per round)"}


def report():
    S, Rn = pd.read_csv(D.SERIES_CSV), pd.read_csv(D.ROUNDS_CSV)
    S = S[(S.label == "cont") & (S.run == "drift")]
    Rn = Rn[(Rn.label == "cont") & (Rn.run == "drift")]
    meths = ["W", "S", "C", "Cb"]
    out = ["# v2 habit drift, 15 rounds (`v2_drift.py`)\n",
           "Final architecture, threshold rule, 5 seeds. Stream: 15 rounds × 50 interactions; dense fruit-Saturday evidence (8 of every 50 scenarios); original rule in rounds 1–3, from round 4 "
           "(interaction 151) the Saturday-fruit rule no longer holds (necessity 0.8 instead of 0.1). Regret on the 18 Saturday-fruit grid cells against the NEW truth; full-grid regret against the truth of the moment. "
           "No local-relearning mechanism.\n",
           "## Adaptation: interactions after the drift until the Saturday-fruit regret is below 0.05\n",
           "| method | seeds reaching it | first (mean over seeds; never = 601) | stays below until the end (mean; never = 601) |", "|---|---|---|---|"]
    reach = {}
    for m in meths:
        df = S[S.method == m]
        f, st = R.reach(df)
        reach[m] = (f, st)
        out.append(f"| {NAME[m]} | {R.summarize_reach(f)} | {R.censored(f).mean():.0f} | {R.censored(st).mean():.0f} |")
    out += ["\n## Regret on the 18 Saturday-fruit cells (new truth) after k interactions of the drifted rule\n", "| k | " + " | ".join(NAME[m] for m in meths) + " |", "|---|" + "---|" * len(meths)]
    for k in (1, 50, 100, 150, 200, 300, 450, 600):
        out.append(f"| {k} | " + " | ".join(R.ms(R.round_end(S[S.method == m], "fs_new", D.DRIFT_ROUND * D.PER + k)) for m in meths) + " |")
    out += ["\n## Full-grid regret at the round ends (truth of the moment)\n", "| interaction | " + " | ".join(NAME[m] for m in meths) + " |", "|---|" + "---|" * len(meths)]
    for r in range(2, 16):
        i = r * D.PER
        col = "grid_old" if i <= D.DRIFT_ROUND * D.PER else "grid_new"
        out.append(f"| {i} (round {r}) | " + " | ".join(R.ms(R.round_end(S[S.method == m], col, i), 4) for m in meths) + " |")
    out += ["\n## Saturday-fruit collapse ratio of the reward model / Q table (Saturday over other days; truth 0.15 before, 1.20 after the drift)\n",
            "| round | " + " | ".join(NAME[m] for m in meths) + " |", "|---|" + "---|" * len(meths)]
    for r in range(1, 16):
        out.append(f"| {r}{' (drift starts next)' if r == D.DRIFT_ROUND else ''} | " + " | ".join(R.ms(Rn[(Rn.method == m) & (Rn['round'] == r)].sort_values('seed').collapse, 2) for m in meths) + " |")
    out += ["\n## Retention of the three unchanged habits at round 15 (bread Sunday mornings, coffee weekday mornings, snack Friday evenings)\n",
            "| method | regret on their cells | learning degree L (mean of the 3) |", "|---|---|---|"]
    for m in meths:
        end = S[(S.method == m) & (S.i == D.T_END)].sort_values("seed")
        L = Rn[(Rn.method == m) & (Rn['round'] == 15)].sort_values("seed")[[f"L_{t}" for t in D.HAB3]].mean(axis=1)
        out.append(f"| {NAME[m]} | {R.ms(end[[f'h_{t}' for t in D.HAB3]].mean(axis=1))} | {R.ms(L, 2)} |")
    with open(os.path.join(RESULTS_DIR, "drift_baselines.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    D.simulate("v2", D.SEEDS)
    report()
