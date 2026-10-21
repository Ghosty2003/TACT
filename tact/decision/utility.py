"""Clarification Utility U = Acc / (1 + AskRate), appended to results/budget_summary.md.

    python -m tact.decision.utility       # after budget.py, hard_paraphrase.py (idempotent: replaces its own section)

Acc = agreement of the decision with the oracle (ask iff the clean ask value exceeds the clean act value), AskRate = share of
scenarios in which the method asks. Both come from the stored per-run records (budget_runs.csv, hard_paraphrase_runs.csv,
dpo_semionline.csv: acc and act_rate, AskRate = 1 - act_rate); no model is re-trained. U is computed per seed, then
averaged (mean ± std over 5 seeds). Baselines (always-ask, never-ask, oracle) are computed from the test scenarios.
"""
import os

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.model import RESULTS_DIR

MARK = "## Clarification utility"
NAME = {"A": "A Ours", "B": "B Ours-direct", "C": "C Tabular", "Ct": "Ct Tabular (text key)"}


def U(df):
    return df.acc / (1.0 + (1.0 - df.act_rate))


def ms(x, nd=3):
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def main():
    r = lambda f: pd.read_csv(os.path.join(RESULTS_DIR, f))
    bud, hard, so = r("budget_runs.csv"), r("hard_paraphrase_runs.csv"), r("dpo_semionline.csv")
    for d in (bud, hard, so):
        d["ask_rate"] = 1.0 - d.act_rate
        d["U"] = U(d)
    bl = B.baselines(B.scenarios(B.load_long("test_seen.csv")))
    bu = {k: v["acc"] / (1 + (1 - v["act_rate"])) for k, v in bl.items()}

    out = [MARK + " U = Acc / (1 + AskRate)\n",
           "Added after the main experiments; computed from the stored decisions (acc and ask-rate of every run), nothing is "
           "re-trained. AskRate = share of test scenarios in which the method asks. U rewards correct decisions and penalises asking: "
           "at equal accuracy a policy that asks half as often scores higher. Mean ± std over 5 seeds of the per-seed U.\n",
           "### Baselines (75 test scenarios)\n", "| | accuracy | AskRate | U |", "|---|---|---|---|"]
    for k, nm in (("always_ask", "always-ask"), ("never_ask", "never-ask"), ("oracle", "oracle")):
        out.append(f"| {nm} | {bl[k]['acc']:.3f} | {1 - bl[k]['act_rate']:.3f} | {bu[k]:.3f} |")
    out.append("\nThe oracle's U is 0.61 and the trivial policies sit at 0.32 (always-ask) and 0.36 (never-ask); a method must beat "
               "0.36 to be better than never asking.\n")

    def table(df, tests, label, extra_rows=None):
        lines = []
        for ts in tests:
            lines += [f"\n### {label}: test_{ts}, U by N\n", "| N | " + " | ".join(NAME[m] for m in B.METHODS) + " |",
                      "|---|" + "---|" * len(B.METHODS)]
            for n in B.NS:
                lines.append(f"| {n} | " + " | ".join(
                    ms(df[(df.method == m) & (df.test == ts) & (df.n == n)].U) for m in B.METHODS) + " |")
        return lines

    out += table(bud, ("seen", "unseen"), "Budget sweep")
    out += ["\n### Budget sweep, N = 350: accuracy, AskRate and U on test_seen\n",
            "| method | accuracy | AskRate | U |", "|---|---|---|---|"]
    for m in B.METHODS:
        g = bud[(bud.method == m) & (bud.test == "seen") & (bud.n == 350)]
        out.append(f"| {NAME[m]} | {ms(g.acc, 2)} | {ms(g.ask_rate, 2)} | {ms(g.U)} |")
    for k, nm in (("always_ask", "always-ask"), ("never_ask", "never-ask"), ("oracle", "oracle")):
        out.append(f"| {nm} | {bl[k]['acc']:.2f} | {1 - bl[k]['act_rate']:.2f} | {bu[k]:.3f} |")

    # hard paraphrases (mean of hard1 and hard2 per seed), N = 150 and 350
    out += ["\n### Hard paraphrases (zero-shot): U on test_seen vs. the hard set\n",
            "Hard = mean of the two paraphrase sets per seed. Baselines do not depend on the wording (see above).\n",
            "| N | method | U seen | U hard (mean of hard1, hard2) | AskRate seen → hard |", "|---|---|---|---|---|"]
    for n in (150, 350):
        for m in B.METHODS:
            g = hard[(hard.method == m) & (hard.n == n)]
            seen = g[g.test == "seen"].sort_values("seed")
            hd = g[g.test.isin(["hard1", "hard2"])].groupby("seed")[["U", "ask_rate"]].mean()
            out.append(f"| {n} | {NAME[m]} | {ms(seen.U)} | {ms(hd.U)} | {seen.ask_rate.mean():.2f} → {hd.ask_rate.mean():.2f} |")

    # semi-online
    out += ["\n### Semi-online loop: U by round\n",
            "| round | n | A seen | C seen | A unseen | C unseen | B seen |", "|---|---|---|---|---|---|---|"]
    for rd in range(1, 8):
        cells = []
        for ts, m in (("seen", "A"), ("seen", "C"), ("unseen", "A"), ("unseen", "C"), ("seen", "B")):
            cells.append(ms(so[(so.method == m) & (so.test == ts) & (so["round"] == rd)].U))
        out.append(f"| {rd} | {50 * rd} | " + " | ".join(cells) + " |")

    path = os.path.join(RESULTS_DIR, "budget_summary.md")
    text = open(path).read()
    if MARK in text:
        text = text[:text.index(MARK)].rstrip("\n") + "\n"
    with open(path, "w") as f:
        f.write(text.rstrip("\n") + "\n\n" + "\n".join(out) + "\n")
    pd.concat([bud.assign(experiment="budget"), hard.assign(experiment="hard")]).to_csv(
        os.path.join(RESULTS_DIR, "utility_runs.csv"), index=False)
    print("\n".join(out))


if __name__ == "__main__":
    main()
