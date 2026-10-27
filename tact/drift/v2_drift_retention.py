"""v2 habit drift, supplement: retention of the three unchanged habits (results/v2/drift_retention.md).

    python -m tact.drift.v2_drift_retention        # after v2_drift.py (reads drift_trigger_rounds.csv / drift_trigger_series.csv of results/v2)

For every baseline of the 15-round drift stream (TACT warm start, tabular online, tabular per round, retrain from scratch on the last 3 rounds): the learning degree L of each of the three habits that do
NOT change (bread on Sunday mornings, coffee on weekday mornings, snack on Friday evenings) before the drift (round 3) and after it (round 15, and at rounds 6, 9, 12), 5 seeds, mean ± std.
L = (1 - predicted ratio) / (1 - true ratio) of the habit's cells to the other cells of the same topic (reward model's ask scores; for the tabular agent its Q values); 1 = fully learned.
"""
import os

import numpy as np
import pandas as pd

from tact.core.model import RESULTS_DIR

HAB = ("bread", "coffee", "snack")
NAME = {"W": "TACT (warm start, all data)", "S": "retrain from scratch, last 3 rounds", "C": "tabular (online)", "Cb": "tabular (per round)"}
DESC = {"bread": "bread, Sunday mornings", "coffee": "coffee, weekday mornings", "snack": "snack, Friday evenings"}


def ms(x, nd=2):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def main():
    R = pd.read_csv(os.path.join(RESULTS_DIR, "drift_trigger_rounds.csv"))
    S = pd.read_csv(os.path.join(RESULTS_DIR, "drift_trigger_series.csv"))
    R, S = R[(R.label == "cont") & (R.run == "drift")], S[(S.label == "cont") & (S.run == "drift")]
    g = lambda m, r: R[(R.method == m) & (R["round"] == r)].sort_values("seed")
    out = ["# v2 habit drift, supplement: retention of the three unchanged habits (`v2_drift_retention.py`)\n",
           "Final architecture, threshold rule, 15 rounds × 50 interactions, 5 seeds (mean ± std over seeds); the Saturday-fruit habit stops being true after round 3 (interaction 150), the other three habits never change. "
           "L = learning degree of a habit (1 = learned as strongly as the truth, 0 = not at all; negative = the opposite effect), computed from the reward model's ask scores on the 1512-cell grid "
           "(for the tabular agent: from its Q values). Round 3 = the last round before the drift, round 15 = the end of the stream.\n",
           "## L of each unchanged habit at the end of the stream (round 15)\n",
           "| method | " + " | ".join(DESC[h] for h in HAB) + " | mean of the three |", "|---|" + "---|" * (len(HAB) + 1)]
    for m in NAME:
        x = g(m, 15)
        out.append(f"| {NAME[m]} | " + " | ".join(ms(x[f"L_{h}"]) for h in HAB) + f" | {ms(x[[f'L_{h}' for h in HAB]].mean(axis=1))} |")
    out += ["\n## Before the drift (round 3), and the change until round 15\n", "| method | L before the drift (mean of the three) | L at round 15 (mean of the three) | change round 15 − round 3 (paired by seed) |", "|---|---|---|---|"]
    for m in NAME:
        a, b = g(m, 3)[[f"L_{h}" for h in HAB]].mean(axis=1).values, g(m, 15)[[f"L_{h}" for h in HAB]].mean(axis=1).values
        out.append(f"| {NAME[m]} | {ms(a)} | {ms(b)} | {(b - a).mean():+.2f} ± {(b - a).std(ddof=1):.2f} |")
    out += ["\n## L (mean of the three habits) over the drifted rounds\n", "| round | " + " | ".join(NAME[m] for m in NAME) + " |", "|---|" + "---|" * len(NAME)]
    for r in (3, 4, 6, 9, 12, 15):
        out.append(f"| {r}{' (before the drift)' if r == 3 else ''} | " + " | ".join(ms(g(m, r)[[f'L_{h}' for h in HAB]].mean(axis=1)) for m in NAME) + " |")
    out += ["\n## Regret on the cells of the three habits at the end of the stream (decisions of the deployed policy; the truth does not change)\n",
            "| method | " + " | ".join(DESC[h] for h in HAB) + " | mean of the three | worst value of that mean at any interaction (mean over seeds) |", "|---|" + "---|" * (len(HAB) + 2)]
    for m in NAME:
        end = S[(S.method == m) & (S.i == 750)].sort_values("seed")
        allh = S[S.method == m].assign(h=lambda d: d[[f"h_{h}" for h in HAB]].mean(axis=1)).groupby("seed").h.max()
        out.append(f"| {NAME[m]} | " + " | ".join(ms(end[f"h_{h}"], 3) for h in HAB) + f" | {ms(end[[f'h_{h}' for h in HAB]].mean(axis=1), 3)} | {allh.mean():.3f} |")
    with open(os.path.join(RESULTS_DIR, "drift_retention.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
