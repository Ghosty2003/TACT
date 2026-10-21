"""Sensitivity of the DPO policy to beta (results/beta_sweep.md).

    python -m tact.decision.beta_sweep

beta in {0.05, 0.1, 0.5}, 5 seeds, N = 150 and 350; data, seeds, hyper-parameters and the whole pipeline are those of the main DPO
experiment (hard_paraphrase.run: reward model -> SFT -> DPO, then seen / unseen phrasing / the two hard-paraphrase sets);
only budget.BETA is changed. beta = 0.1 must reproduce hard_paraphrase_runs.csv, and the argmax decision B does not depend on beta.
"""
import os

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.decision import hard_paraphrase as HP
from tact.decision.hard_paraphrase_texts import HARD, HARD_DRAFT1
from tact.core.model import RESULTS_DIR

BETAS = [0.05, 0.1, 0.5]


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def main():
    dev = B.get_device()
    data = B.Data(B.FrozenBert(dev), dev)
    data.enc.encode(sorted({h for t in HARD.values() for h in t} | {h for t in HARD_DRAFT1.values() for h in t}))
    frames, flips = [], []
    for beta in BETAS:
        B.BETA = beta
        df, store, sets = HP.run(data)
        frames.append(df.assign(beta=beta))
        for n in HP.NS_HARD:
            for m in ("A", "B"):
                for s in B.SEEDS:
                    f = np.mean([(store[(n, s, "seen", m)] != store[(n, s, h, m)]).mean() for h in ("hard1", "hard2")])
                    flips.append({"beta": beta, "n": n, "method": m, "seed": s, "flip_rate": f})
        print(f"[beta] {beta} done", flush=True)
    B.BETA = 0.1
    R = pd.concat(frames, ignore_index=True)
    F = pd.DataFrame(flips)
    R.to_csv(os.path.join(RESULTS_DIR, "beta_sweep_runs.csv"), index=False)

    # integrity: beta = 0.1 reproduces the main runs; B does not depend on beta
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "hard_paraphrase_runs.csv"))
    m1 = R[R.beta == 0.1].merge(ref, on=["n", "seed", "method", "test"], suffixes=("", "_ref"))
    d01 = float((m1.regret - m1.regret_ref).abs().max())
    bB = R[(R.method == "B")].pivot_table(index=["n", "seed", "test"], columns="beta", values="regret")
    dB = float((bB.max(axis=1) - bB.min(axis=1)).max())

    def col(beta, n, m, test):
        if test == "hard":
            return np.array([R[(R.beta == beta) & (R.n == n) & (R.method == m) & (R.test.isin(["hard1", "hard2"])) & (R.seed == s)].regret.mean()
                             for s in B.SEEDS])
        return R[(R.beta == beta) & (R.n == n) & (R.method == m) & (R.test == test)].sort_values("seed").regret.values

    out = ["# β sensitivity of the DPO policy (`beta_sweep.py`)\n",
           "β ∈ {0.05, 0.1, 0.5}, 5 seeds, N = 150 and 350 training scenarios. Data, seeds, learning rates, early stopping and the whole pipeline "
           "(reward model → SFT → DPO, dropout off, reference = SFT policy) are those of the main DPO experiment; only β changed. "
           f"Checks: β = 0.1 reproduces `hard_paraphrase_runs.csv` (max abs regret difference {d01:.1e}); the argmax decision B does not depend "
           f"on β (max difference across β {dB:.1e}). Regret on the clean values, mean ± std over seeds; hard = mean of the two hard-paraphrase "
           "sets per seed.\n"]
    for n in HP.NS_HARD:
        out += [f"## N = {n}\n",
                "| β | A regret, seen | A regret, unseen phrasing 3 | A regret, hard paraphrases | A: decisions changed seen → hard | "
                "A − B regret, seen | A − B regret, hard |", "|---|---|---|---|---|---|---|"]
        for beta in BETAS:
            fa = F[(F.beta == beta) & (F.n == n) & (F.method == "A")].sort_values("seed").flip_rate.values
            out.append(f"| {beta:g} | {ms(col(beta, n, 'A', 'seen'))} | {ms(col(beta, n, 'A', 'unseen'))} | {ms(col(beta, n, 'A', 'hard'))} | "
                       f"{100 * fa.mean():.1f}% ± {100 * fa.std(ddof=1):.1f}% | "
                       f"{ms(col(beta, n, 'A', 'seen') - col(beta, n, 'B', 'seen'))} | {ms(col(beta, n, 'A', 'hard') - col(beta, n, 'B', 'hard'))} |")
        out.append(f"| argmax B (all β) | {ms(col(0.1, n, 'B', 'seen'))} | {ms(col(0.1, n, 'B', 'unseen'))} | {ms(col(0.1, n, 'B', 'hard'))} | "
                   f"{100 * F[(F.beta == 0.1) & (F.n == n) & (F.method == 'B')].flip_rate.mean():.1f}% | – | – |")
        out.append("")
    out += ["## Is the result sensitive to β?\n",
            "Rule fixed beforehand: sensitive if some pair of β values differs in mean by more than the larger of their seed standard "
            "deviations. Largest difference between β values, and the corresponding seed std:\n",
            "| N | quantity | largest difference between β | larger seed std | verdict |", "|---|---|---|---|---|"]
    verdicts = []
    for n in HP.NS_HARD:
        for nm, test in (("regret, seen", "seen"), ("regret, unseen phrasing", "unseen"), ("regret, hard", "hard")):
            vals = {b: col(b, n, "A", test) for b in BETAS}
            means = {b: v.mean() for b, v in vals.items()}
            hi, lo = max(means, key=means.get), min(means, key=means.get)
            diff = means[hi] - means[lo]
            sd = max(vals[hi].std(ddof=1), vals[lo].std(ddof=1))
            v = "sensitive" if diff > sd else "within noise"
            verdicts.append(v)
            out.append(f"| {n} | A {nm} | {diff:.3f} (β = {hi:g} vs {lo:g}) | {sd:.3f} | {v} |")
    out.append(f"\n**Overall: {'the policy is sensitive to β in at least one quantity' if 'sensitive' in verdicts else 'no quantity differs by more than its seed std: the results are not sensitive to β in this range'}** "
               f"({verdicts.count('within noise')} of {len(verdicts)} comparisons within noise).")
    with open(os.path.join(RESULTS_DIR, "beta_sweep.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
