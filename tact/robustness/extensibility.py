"""Extensibility check: a 4th context variable (location) with an interaction rule (results/extensibility.md).

    python -m tact.core.generate_dataset --n 500 --seed 42 --with-act --with-location --outdir data_ext
    python -m tact.robustness.extensibility          # after budget.py and ARCH=onehot python -m tact.decision.budget (their 3-variable curves are re-used)

data_ext = data_dpo + location in {kitchen, living_room, office}; in the office the receptiveness of every clarification is
multiplied by 0.3 (non-office labels are identical to data_dpo). Models are NOT changed, only given one more input: BertScorer a
location embedding of 8 dims (concatenated with the emotion/day/time embeddings before the context MLP), OneHotScorer 3 more
one-hot dimensions. Everything else (training scenarios, seeds, hyper-parameters, SFT + DPO) is as in budget.py. Evaluation:
  * reward model: the full 12 x 6 x 7 x 3 x 3 = 4536-cell grid against the clean formula (with the office factor), N = 350,
    compared with the same pipeline on the 3-variable data (1512 cells);
  * decisions: regret on test_seen for A (policy), B (argmax of the reward model) and the tabular agent with the key
    (topic, emotion, day, time) [C_noloc, ignores the location] and (topic, emotion, day, time, location) [C_loc, 126 -> 378 contexts per topic].
"""
import os

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.generate_dataset import QUESTIONS, availability
from tact.reward.grid_eval import TOPICS, build_grid, grid_metrics, pm
from tact.core.model import CONFIG, EMOTIONS, LOCATIONS, RESULTS_DIR, ROOT, FrozenBert, get_device
from tact.core.plots import is_habit, necessity

DATA_EXT = os.path.join(ROOT, "data_ext")
LOC_KW = {"bert": {"location_dim": 8}, "onehot": {"n_locations": 3}}
ARCH_NAME = {"bert": "BertScorer (+8-d location embedding)", "onehot": "OneHotScorer (+3-d one-hot location)"}
CTX4 = ("emotion", "day", "time", "location")
RESULT_DIR_3 = {"bert": os.path.join(ROOT, "results"), "onehot": os.path.join(ROOT, "results", "onehot_as_main")}


def build_grid4():
    rows, truth, habit, office = [], [], [], []
    for t in TOPICS:
        for e in EMOTIONS:
            for d in range(7):
                for m in range(3):
                    for loc in LOCATIONS:
                        rows.append({"question": QUESTIONS[t][0], "emotion": e, "day": d, "time": m, "location": loc})
                        f = 0.3 if loc == "office" else 1.0
                        truth.append(necessity(t, d, m) * CONFIG["valence"][e] * availability(d, m) * f)
                        habit.append(is_habit(t, d, m))
                        office.append(loc == "office")
    return pd.DataFrame(rows), np.array(truth), np.array(habit), np.array(office)


def office_ratio(pred, habit, office):
    """Mean prediction in the office over the mean prediction elsewhere, non-habit cells (truth: 0.3)."""
    nh = ~habit
    return float(pred[nh & office].mean() / pred[nh & ~office].mean())


def main():
    dev = get_device()
    enc = FrozenBert(dev)
    d3, d4 = B.Data(enc, dev), B.Data(enc, dev, data_dir=DATA_EXT)
    bl3, bl4 = B.baselines(d3.tests["seen"]), B.baselines(d4.tests["seen"])
    base3 = min(bl3["always_ask"]["regret"], bl3["never_ask"]["regret"])
    base4 = min(bl4["always_ask"]["regret"], bl4["never_ask"]["regret"])
    g3, t3, h3 = build_grid()
    g4, t4, h4, off4 = build_grid4()
    # parameter counts of the two input variants
    npar = {}
    for a in ("bert", "onehot"):
        B.ARCH = a
        B.MODEL_KW = {}
        p3 = sum(p.numel() for p in B.new_scorer(enc, dev, 0, True).parameters())
        B.MODEL_KW = LOC_KW[a]
        p4 = sum(p.numel() for p in B.new_scorer(enc, dev, 0, True).parameters())
        npar[a] = (p3, p4)
    B.MODEL_KW = {}
    office_test = (d4.tests["seen"].location == "office").values

    dec_rows, grid_rows = [], []
    for arch in ("bert", "onehot"):
        B.ARCH = arch
        for n in B.NS:
            for seed in B.SEEDS:
                s = seed * 1000 + n
                B.MODEL_KW = LOC_KW[arch]
                ours = B.Ours(d4, seed)
                ours.fit_round(n, s, warm=False)
                a, b = ours.decide(d4.f_test["seen"])
                tr_sc = d4.sc_train.iloc[:n]
                order = np.random.default_rng([seed, n]).permutation(n)
                tabs = {"C_noloc": B.Tabular("topic"), "C_loc": B.Tabular("topic", ctx_cols=CTX4)}
                for t in tabs.values():
                    t.update(tr_sc, order)
                dec = {"A": (a, {}), "B": (b, {})}
                for name, t in tabs.items():
                    d, lv = t.decide(d4.tests["seen"])
                    dec[name] = (d, {"lvl_exact": lv[0], "lvl_topic": lv[1], "lvl_global": lv[2]})
                for m, (d, extra) in dec.items():
                    dec_rows.append({"arch": arch, "n": n, "seed": seed, "method": m, **B.evaluate(d, d4.tests["seen"]),
                                     "act_office": float((~d[office_test]).mean()), "act_other": float((~d[~office_test]).mean()), **extra})
                if n == 350:     # reward model on the full 4-variable grid
                    pred = ours.rm.predict(g4)
                    gm = grid_metrics(pred, t4, h4)
                    grid_rows.append({"arch": arch, "variant": "4var", "seed": seed, **gm,
                                      "mse_office": float(np.mean((pred[off4] - t4[off4]) ** 2)),
                                      "mse_other": float(np.mean((pred[~off4] - t4[~off4]) ** 2)),
                                      "office_ratio": office_ratio(pred, h4, off4),
                                      "pred_office": float(pred[~h4 & off4].mean()), "pred_other": float(pred[~h4 & ~off4].mean())})
                    # the same metrics on the non-office cells only: the cell types of the 3-variable grid, hence comparable with it
                    grid_rows.append({"arch": arch, "variant": "4var_other", "seed": seed,
                                      **grid_metrics(pred[~off4], t4[~off4], h4[~off4])})
                    # 3-variable control: same pipeline on data_dpo (reward model only)
                    B.MODEL_KW = {}
                    c3 = B.Ours(d3, seed)
                    c3.fit_reward(350, s, warm=False)
                    grid_rows.append({"arch": arch, "variant": "3var", "seed": seed, **grid_metrics(c3.rm.predict(g3), t3, h3)})
            print(f"[ext] {arch} N={n} done", flush=True)
    B.MODEL_KW = {}
    D = pd.DataFrame(dec_rows)
    G = pd.DataFrame(grid_rows)
    D.to_csv(os.path.join(RESULTS_DIR, "extensibility_runs.csv"), index=False)
    G.to_csv(os.path.join(RESULTS_DIR, "extensibility_grid.csv"), index=False)

    # 3-variable decision curves from the budget runs of the same encodings
    br = {a: pd.read_csv(os.path.join(RESULT_DIR_3[a], "budget_runs.csv")) for a in ("bert", "onehot")}

    def curve3(a, m):
        g = br[a][(br[a].method == m) & (br[a].test == "seen")]
        return g.groupby("n").regret.mean()

    def curve4(a, m):
        g = D[(D.arch == a) & (D.method == m)]
        return g.groupby("n").regret.mean()

    def first(curve, base, frac):
        for n in B.NS:
            if curve[n] <= frac * base:
                return n
        return None

    out = ["# Extensibility: a fourth context variable (`extensibility.py`)\n",
           "**Data (`data_ext/`, `generate_dataset.py --n 500 --seed 42 --with-act --with-location`).** data_dpo plus `location` ∈ "
           "{kitchen, living_room, office} (uniform per scenario: 172 / 165 / 163 scenarios); in the office the receptiveness of every "
           "clarification is multiplied by 0.3 (ask labels re-derived for the office scenarios, everything else identical to data_dpo). "
           f"The oracle acts in 55.6% of the scenarios (92.6% in the office, 37–38% elsewhere; 37.4% overall without the location). "
           f"Test baselines: always-ask {bl4['always_ask']['regret']:.3f}, never-ask {bl4['never_ask']['regret']:.3f} "
           f"(data_dpo: {bl3['always_ask']['regret']:.3f} / {bl3['never_ask']['regret']:.3f}).\n",
           "**Models.** The architectures are unchanged except for one more input: BertScorer gets a location embedding of 8 dimensions "
           f"({npar['bert'][1] - npar['bert'][0]:+,} trainable parameters, {npar['bert'][1] / npar['bert'][0] - 1:+.2%}), OneHotScorer 3 more "
           f"one-hot dimensions ({npar['onehot'][1] - npar['onehot'][0]:+,}, {npar['onehot'][1] / npar['onehot'][0] - 1:+.2%}). Everything "
           "else as in budget.py; 5 seeds. The 3-variable reference numbers come from the same pipeline on data_dpo.\n",
           "## 1. Reward model on the full grid (N = 350)\n",
           "Grid: 12 topics × 6 emotions × 7 days × 3 times × 3 locations = 4536 cells (habit cells: 180), clean formula with the office "
           "factor. The 3-variable control uses the 1512-cell grid of grid_eval.py (habit cells: 60). **The all-cell MSE of the 4-variable "
           "grid is not comparable with the 3-variable one**: a third of its cells are office cells whose true values are multiplied by "
           "0.3 and are therefore small and easy, which lowers the average. The comparable row is the 4-variable model on the "
           "non-office cells only (the same cell types as the 3-variable grid, 1512 cells).\n",
           "| encoding | data | MSE | Spearman | pairwise | habit MSE | habit Spearman | non-habit MSE |", "|---|---|---|---|---|---|---|---|"]
    for a in ("bert", "onehot"):
        for v, nm in (("3var", "3 variables (data_dpo), 1512 cells"), ("4var_other", "4 variables, non-office cells only (comparable)"),
                      ("4var", "4 variables, all 4536 cells")):
            g = G[(G.arch == a) & (G.variant == v)]
            out.append(f"| {ARCH_NAME[a] if v == '4var' else a} | {nm} | {pm(g.mse, 4)} | {pm(g.spearman)} | {pm(g.pairwise_acc)} | "
                       f"{pm(g.habit_mse, 4)} | {pm(g.habit_spearman)} | {pm(g.nonhabit_mse, 4)} |")
    out += ["\n### Drop from adding the variable (4 variables on non-office cells − 3 variables, seed-paired difference; positive MSE = worse)\n",
            "| encoding | ΔMSE | ΔSpearman | Δpairwise | Δ habit MSE | Δ habit Spearman |", "|---|---|---|---|---|---|"]
    for a in ("bert", "onehot"):
        x = G[(G.arch == a) & (G.variant == "4var_other")].sort_values("seed")
        y = G[(G.arch == a) & (G.variant == "3var")].sort_values("seed")
        cells = [pm(x[c].values - y[c].values, nd) for c, nd in (("mse", 4), ("spearman", 3), ("pairwise_acc", 3), ("habit_mse", 4), ("habit_spearman", 3))]
        out.append(f"| {a} | " + " | ".join(cells) + " |")
    out += ["\n## 2. Is the office collapse learned? (reward model, N = 350)\n",
            "Mean predicted ask score in the office divided by the mean elsewhere over the non-habit cells (truth: exactly 0.30, "
            "because the factor multiplies every cell).\n",
            "| encoding | predicted office / other | mean prediction office | elsewhere | MSE office cells | MSE other cells |", "|---|---|---|---|---|---|"]
    for a in ("bert", "onehot"):
        g = G[(G.arch == a) & (G.variant == "4var")]
        out.append(f"| {a} | {pm(g.office_ratio)} | {pm(g.pred_office)} | {pm(g.pred_other)} | {pm(g.mse_office, 4)} | {pm(g.mse_other, 4)} |")
    out += ["\nAt decision level (test_seen, N = 350): share of scenarios in which the method acts (does not ask).\n",
            "| method | acts in the office | acts elsewhere |", "|---|---|---|"]
    ts = d4.tests["seen"]
    ora = ts.c_ask.values >= ts.c_act.values
    out.append(f"| oracle | {(~ora[office_test]).mean():.2f} | {(~ora[~office_test]).mean():.2f} |")
    for a in ("bert", "onehot"):
        for m in ("A", "B"):
            g = D[(D.arch == a) & (D.method == m) & (D.n == 350)]
            out.append(f"| {m} ({a}) | {pm(g.act_office, 2)} | {pm(g.act_other, 2)} |")
    for m, nm in (("C_noloc", "C, key without location"), ("C_loc", "C, key with location")):
        g = D[(D.arch == "bert") & (D.method == m) & (D.n == 350)]
        out.append(f"| {nm} | {pm(g.act_office, 2)} | {pm(g.act_other, 2)} |")
    out.append(f"\nOffice scenarios in the test set: {int(office_test.sum())} of {len(office_test)}.\n")

    out += ["## 3. Decision regret at N = 350 (test_seen) and the data cost of the extra variable\n",
            "Normalised regret = regret / min(always-ask, never-ask) of the same test set (the baselines differ between the two "
            f"datasets: {base3:.3f} for data_dpo, {base4:.3f} for data_ext). 3 variables: the budget.py runs; 4 variables: this script.\n",
            "| method | 3 variables: regret (normalised) | 4 variables: regret (normalised) | change in regret | change in normalised regret |",
            "|---|---|---|---|---|"]
    rows = [("A  policy, BertScorer", "bert", "A", "A"), ("B  argmax, BertScorer", "bert", "B", "B"),
            ("A  policy, OneHotScorer", "onehot", "A", "A"), ("B  argmax, OneHotScorer", "onehot", "B", "B"),
            ("C  tabular, key with all variables (3 → 4 variables)", "bert", "C", "C_loc"),
            ("C  tabular, key without location (4-variable data)", "bert", "C", "C_noloc")]
    for nm, a, m3, m4 in rows:
        v3 = br[a][(br[a].method == m3) & (br[a].test == "seen") & (br[a].n == 350)].sort_values("seed").regret.values
        v4 = D[(D.arch == a) & (D.method == m4) & (D.n == 350)].sort_values("seed").regret.values
        out.append(f"| {nm} | {pm(v3)} ({v3.mean() / base3:.2f}) | {pm(v4)} ({v4.mean() / base4:.2f}) | {v4.mean() - v3.mean():+.3f} | "
                   f"{v4.mean() / base4 - v3.mean() / base3:+.2f} |")
    out += ["\nTabular agent C: contexts per topic 126 → 378 (1512 → 4536 keys in all); expected training visits per key at N = 350: "
            f"{350 / (12 * 126):.2f} → {350 / (12 * 378):.3f}. Share of test scenarios whose exact key was seen (both actions) at N = 350: "
            f"3 variables {br['bert'][(br['bert'].method == 'C') & (br['bert'].test == 'seen') & (br['bert'].n == 350)].lvl_exact.mean():.2f}, "
            f"4 variables {D[(D.arch == 'bert') & (D.method == 'C_loc') & (D.n == 350)].lvl_exact.mean():.2f}.\n",
            "### Normalised regret by N, 3 variables → 4 variables (mean over 5 seeds)\n",
            "| N | " + " | ".join(f"{nm}" for nm, *_ in rows) + " |", "|---|" + "---|" * len(rows)]
    for n in B.NS:
        cells = []
        for nm, a, m3, m4 in rows:
            cells.append(f"{curve3(a, m3)[n] / base3:.2f} → {curve4(a, m4)[n] / base4:.2f}")
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    out += ["\n### Samples needed (smallest N on the grid)\n",
            "“–” = not reached by N = 350. Equivalent N = smallest 3-variable N whose mean normalised regret is at or below the "
            "4-variable value at N = 350 (how many 3-variable samples the 4-variable model's N = 350 is worth).\n",
            "| method | N for normalised regret ≤ 1: 3 → 4 variables | N for ≤ 0.5: 3 → 4 variables | normalised regret at N = 350: 3 → 4 | equivalent 3-variable N |",
            "|---|---|---|---|---|"]
    for nm, a, m3, m4 in rows[:5]:
        c3, c4 = curve3(a, m3), curve4(a, m4)
        eq = next((n for n in B.NS if c3[n] / base3 <= c4[350] / base4), None)
        f = lambda x: "–" if x is None else str(x)
        eq_txt = "no 3-variable N ≤ 350 reaches it (gaps of 0.01–0.03 are within noise)" if eq is None else str(eq)
        out.append(f"| {nm} | {f(first(c3, base3, 1.0))} → {f(first(c4, base4, 1.0))} | {f(first(c3, base3, 0.5))} → "
                   f"{f(first(c4, base4, 0.5))} | {c3[350] / base3:.2f} → {c4[350] / base4:.2f} | {eq_txt} |")
    out.append("\nThe equivalent N of the tabular agent is not very meaningful: its 3-variable curve is nearly flat (normalised "
               "regret 0.50–0.75 for all N), so many N values are 'equivalent'.")
    with open(os.path.join(RESULTS_DIR, "extensibility.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
