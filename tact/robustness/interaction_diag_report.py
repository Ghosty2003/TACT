"""Report of interaction_diag.py (results/v2/interaction_diag.md)."""
import os

import numpy as np
import pandas as pd

from tact.core.generate_dataset import KITCHEN_TOPICS, QUESTIONS
from tact.core.model import RESULTS_DIR, ROOT

RUNS = os.path.join(RESULTS_DIR, "interaction_diag_runs.csv")
REF7 = "ref_7var_existing"
FACT = ["fruit_x0.5_default", "fruit_x0.5_boost", "fruit_x0.1_default", "fruit_x0.1_boost", "kitchen_x0.5_default", "kitchen_x0.5_boost", "kitchen_x0.1_default", "kitchen_x0.1_boost"]
ACTFIX = "kitchen_x0.5_default_actfixed"
CTRL = ["kitchen_x0.5_boosttrain", "fruit_x0.5_boosttrain"]
LABEL = {REF7: "ref A: 7 variables, existing rule (kitchen group, ×0.5, default density)", ACTFIX: "ref B: 4 variables, existing rule, act score NOT changed by the rule"}


def ms(x, nd=2):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def desc(name):
    if name in LABEL:
        return LABEL[name]
    s, st, d = name.split("_")
    dd = {"default": "default sampling", "boost": "~20% affected (train and validation)", "boosttrain": "~20% affected in TRAINING only (validation default)"}[d]
    return f"{s} {'only (1 topic)' if s == 'fruit' else 'group (4 topics)'}, necessity {'× 0.5' if st == 'x0.5' else '→ 0.10'}, {dd}"


def step0():
    rng = np.random.default_rng(0)
    out = []
    for n in (350, 700):
        tr = pd.read_csv(os.path.join(ROOT, "data_ctx7_n1000", "train.csv"))
        ask = tr[tr.cand == "ask"].reset_index(drop=True).iloc[:n]
        act = tr[tr.cand == "act"].reset_index(drop=True).iloc[:n]
        kit, cook = ask.topic.isin(KITCHEN_TOPICS).values, (ask.activity == "cooking").values
        aff, ctl = kit & cook, kit & ~cook
        ia, ic = np.where(aff)[0], np.where(ctl)[0]
        boots = [ask.score.values[rng.choice(ia, len(ia))].mean() / ask.score.values[rng.choice(ic, len(ic))].mean() for _ in range(4000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append(f"| {n} | {int(kit.sum())} | {int(aff.sum())} ({aff.mean():.1%}) | {int(ctl.sum())} | {ask.score.values[aff].mean() / ask.score.values[ctl].mean():.3f} (95% CI {lo:.2f}–{hi:.2f}) | "
                   f"{ask.score_clean.values[aff].mean() / ask.score_clean.values[ctl].mean():.3f} | {act.score.values[aff].mean() / act.score.values[ctl].mean():.2f} (clean {act.score_clean.values[aff].mean() / act.score_clean.values[ctl].mean():.2f}) | "
                   f"{ask.necessity.values[aff].mean():.3f} / {ask.necessity.values[ctl].mean():.3f} |")
    return out


def write_report():
    R = pd.read_csv(RUNS)
    g = lambda c, n, col: R[(R.condition == c) & (R.N == n)].sort_values("seed")[col].values
    topics = list(QUESTIONS)
    out = ["# Diagnostic: why is the 'activity × topic' rule not learned? (`interaction_diag.py`)\n",
           "Final architecture (frozen NLI-BERT mean pooling + Context encoder + rank-16 bilinear head), 5 seeds, N = 350 and 700 (first N scenarios of the 700 training scenarios of an `--n 1000` dataset, as in "
           "Appendix A), mean ± std. **'Learned' = learning degree L ≥ 0.5 (fixed beforehand).** No file used by the paper was modified; L here is the reward model's L on the full factorial grid, computed as in "
           "`context_scaling.py`.\n"]
    # ---------------- step 0
    out += ["## Step 0: the rule and the check of the data\n",
            "**Definition in the generator (`generate_dataset.py`, `--extra-vars activity`).**\n",
            f"- 12 topics: {', '.join(topics)}. Kitchen group (`KITCHEN_TOPICS`): **{', '.join(sorted(KITCHEN_TOPICS))} = 4 of the 12 topics.**",
            "- `activity` ∈ {relaxing, working, cooking}, drawn uniformly per scenario, independently of everything else (its own random stream, `Random(seed + 5)`).",
            "- The rule fires for activity = cooking AND topic in the kitchen group: about (1/3) × (4/12) = 11% of the scenarios (measured below: 9–10%).",
            "- It changes the **necessity** (× 0.5), not the receptiveness. Consequences: ask value = necessity × receptiveness is halved; **the act value = 1 − necessity also changes** (e.g. plate: 0.25 → 0.625), "
            "so it is a two-variable rule that changes both candidates' scores and, in many cells, the oracle's decision.",
            "- L (as in `context_scaling.py`): over the full grid (12 topics × 6 emotions × 7 days × 3 times × all values of the extra variables), cells of the affected set = cooking ∧ kitchen topic ∧ non-habit "
            "(the kitchen topics have no habit rule), control set = the other two activities, same topics. ratio = mean predicted **ask score** of the affected cells / that of the control cells; the true ratio is "
            "exactly 0.5 (receptiveness and all other variables are identically distributed in the two sets). L = (1 − predicted ratio) / (1 − 0.5). Leak control: the same ratio on the other 8 topics (truth 1).\n",
            "**Does the training data contain the rule?** Training scenarios of `data_ctx7_n1000` (the 7-variable data of Appendix A), first N:\n",
            "| N | kitchen-topic scenarios | affected (cooking × kitchen) | control (not cooking × kitchen) | ratio of mean ask LABEL, affected / control | ratio of mean CLEAN ask on the same scenarios | ratio of mean act label (clean) | mean necessity affected / control |",
            "|---|---|---|---|---|---|---|---|"] + step0()
    out += ["\nThe rule is visible in the labels: the ask labels of the affected scenarios are about half of the controls' (truth 0.5; the clean ratio of the *sampled* scenarios is below 0.5 too, because the other variables are "
            "random), and the act labels are about 2.3 times larger. **Step 0 passes: the problem is not the data or the generator.** What is small is the evidence: 32 / 72 affected scenarios in the training sets, "
            "spread over 4 topics and over the 36 combinations of the other extra variables.\n"]
    # ---------------- step 1
    out += ["## Step 1: is there enough information in the data? Upper-bound reference models\n",
            "Same training data, same grid and same L. Reference models: one-hot inputs (topic, emotion, day, time, every extra variable) **plus the explicit topic × activity cross features** (36 one-hot columns), "
            "trained on the ask rows only: ridge regression (α = 1) and a small MLP (64-32, sigmoid output, Adam 10⁻³, batch 16, early stopping on the validation scenarios). They are given the structure the "
            "TACT model has to find by itself.\n",
            "| data | N | affected scenarios in training | **TACT L** | ridge + cross L | MLP + cross L | TACT: leak ratio (truth 1) |", "|---|---|---|---|---|---|---|"]
    for c in (REF7, "kitchen_x0.5_default"):
        for n in (350, 700):
            x = R[(R.condition == c) & (R.N == n)]
            out.append(f"| {desc(c)} | {n} | {int(x.affected_in_train.iloc[0])} | **{ms(x.L_tact)}** | {ms(x.L_ridge)} | {ms(x.L_mlp)} | {ms(x.leak_tact)} |")
    out.append("")
    # ---------------- step 2
    out += ["## Step 2: separating the candidate causes (3 basic variables + activity unless stated)\n",
            "Rule variants by relabelling the scenarios of one `--extra-vars activity` dataset: scope (only the fruit topic, or the 4 kitchen topics) × strength (necessity × 0.5, or set to 0.10 as a habit rule) × density "
            "(default sampling, or ~20% of the scenarios affected: scenarios of the scope with activity = cooking and random emotion/day/time/phrasing replace others, in the training and validation scenarios, as the dense "
            "variant of Appendix C). Every label is re-derived from the scenario (same noise stream for all variants), so the variants differ only in the factor under test. 'Control scenarios' = same topics, other activities "
            "(few for the single-topic scope).\n",
            "| condition | variables | affected / control scenarios in training (N = 350 · 700) | true ratio | **TACT L, N = 350** | **TACT L, N = 700** | ridge + cross L (350 · 700) | MLP + cross L (350 · 700) | TACT leak ratio (700) |",
            "|---|---|---|---|---|---|---|---|---|"]
    ctlcount = {}
    for c in FACT + [ACTFIX] + CTRL:
        path = os.path.join(ROOT, "data_diag", c)
        tr = pd.read_csv(os.path.join(path, "train.csv"))
        ask = tr[tr.cand == "ask"].reset_index(drop=True)
        scope = {"fruit"} if c.startswith("fruit") else set(KITCHEN_TOPICS)
        ctlcount[c] = [int((~(ask.iloc[:n].activity == "cooking") & ask.iloc[:n].topic.isin(scope)).sum()) for n in (350, 700)]
    rows = [(REF7, "7")] + [(c, "4") for c in FACT] + [(ACTFIX, "4")] + [(c, "4") for c in CTRL]
    for c, nv in rows:
        a350, a700 = R[(R.condition == c) & (R.N == 350)], R[(R.condition == c) & (R.N == 700)]
        mark = lambda v: " ✔" if np.mean(v) >= 0.5 else ""
        cnt = f"{int(a350.affected_in_train.iloc[0])} / {ctlcount[c][0]} · {int(a700.affected_in_train.iloc[0])} / {ctlcount[c][1]}" if c in ctlcount else f"{int(a350.affected_in_train.iloc[0])} · {int(a700.affected_in_train.iloc[0])}"
        out.append(f"| {desc(c)} | {nv} | {cnt} | {a700.true_ratio.mean():.3f} | {ms(a350.L_tact)}{mark(a350.L_tact)} | {ms(a700.L_tact)}{mark(a700.L_tact)} | "
                   f"{a350.L_ridge.mean():.2f} · {a700.L_ridge.mean():.2f} | {a350.L_mlp.mean():.2f} · {a700.L_mlp.mean():.2f} | {a700.leak_tact.mean():.2f} |")
    out.append("\n✔ = mean L ≥ 0.5 (learned).\n")
    # ---------------- factor effects
    out += ["### Effect of each factor on the TACT L (mean over the other factors; the 8-condition grid, act score changes with the rule)\n",
            "| factor | level | N = 350: mean L over its 4 conditions (conditions with L ≥ 0.5) | N = 700: mean L (conditions with L ≥ 0.5) |", "|---|---|---|---|"]
    def lvl(sel, n):
        v = [R[(R.condition == c) & (R.N == n)].L_tact.mean() for c in FACT if sel(c)]
        return f"{np.mean(v):.2f} ({sum(x >= 0.5 for x in v)} of {len(v)})"
    for fac, levels in (("scope", (("only the fruit topic", lambda c: c.startswith("fruit")), ("the 4 kitchen topics", lambda c: c.startswith("kitchen")))),
                        ("strength", (("necessity × 0.5", lambda c: "_x0.5_" in c), ("necessity → 0.10", lambda c: "_x0.1_" in c))),
                        ("density", (("default sampling", lambda c: c.endswith("default")), ("~20% affected", lambda c: c.endswith("boost"))))):
        for nm, sel in levels:
            out.append(f"| {fac} | {nm} | {lvl(sel, 350)} | {lvl(sel, 700)} |")
    out += ["", "Single contrasts (TACT L, mean ± std over 5 seeds):\n", "| contrast | N = 350 | N = 700 |", "|---|---|---|"]
    pairs = [("variable dilution: 7 variables → 4 variables (same existing rule)", REF7, "kitchen_x0.5_default"),
             ("act score: changes with the rule → stays fixed (4 variables, existing rule)", "kitchen_x0.5_default", ACTFIX)]
    for lab, a, b in pairs:
        out.append(f"| {lab} | {ms(g(a, 350, 'L_tact'))} → {ms(g(b, 350, 'L_tact'))} | {ms(g(a, 700, 'L_tact'))} → {ms(g(b, 700, 'L_tact'))} |")
    for lab, a, b in (("scope: kitchen group → fruit only (×0.5, default)", "kitchen_x0.5_default", "fruit_x0.5_default"), ("strength: ×0.5 → 0.10 (kitchen, default)", "kitchen_x0.5_default", "kitchen_x0.1_default"),
                      ("density: default → ~20% (kitchen, ×0.5)", "kitchen_x0.5_default", "kitchen_x0.5_boost"), ("density × strength (kitchen, 0.10, ~20%)", "kitchen_x0.5_default", "kitchen_x0.1_boost"),
                      ("everything at once (fruit only, 0.10, ~20%)", "kitchen_x0.5_default", "fruit_x0.1_boost")):
        out.append(f"| {lab} | {ms(g(a, 350, 'L_tact'))} → {ms(g(b, 350, 'L_tact'))} | {ms(g(a, 700, 'L_tact'))} → {ms(g(b, 700, 'L_tact'))} |")
    p = os.path.join(RESULTS_DIR, "interaction_diag_findings.md")
    if os.path.exists(p):
        out += [""] + open(p).read().rstrip().split("\n")
    with open(os.path.join(RESULTS_DIR, "interaction_diag.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
