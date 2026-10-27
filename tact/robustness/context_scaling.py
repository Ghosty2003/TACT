"""Context-dimension scan: add categorical context variables, each with one multiplicative rule, and look for the limit.

    python -m tact.robustness.context_scaling     # generates data_ctx5/6/7 (+ data_ctx7_n1000) if missing; writes results/context_scaling.{md,png}

Levels (cumulative; every level regenerated with --n 500 --seed 42 on top of data_dpo):
  3 variables  data_dpo                                   (emotion, day, time)
  4            + location    (3)  office: receptiveness x 0.3          [data_ext]
  5            + companion   (2)  with_guests: receptiveness x 0.5
  6            + activity    (3)  cooking: necessity x 0.5 for the utensil topics {plate, knife, cup, bowl}
                                  (an interaction of two variables; it changes the ask value and the act value)
  7            + noise_level (2)  noisy: receptiveness x 0.7
Learned model = BertScorer (Context MLP) with one 8-d embedding per extra variable + SFT/DPO policy as in budget.py; tabular C keyed
by (topic, emotion, day, time, all extra variables). N = 350, 5 seeds. At level 7 an extra pair of runs on a --n 1000 dataset
(N = 350 and N = 700) separates "not enough data" from "capacity limit".

Definitions fixed beforehand:
  * collapse ratio of a rule = mean ask score where the rule fires / mean where it does not, over the non-habit cells of the full
    factorial grid (balanced, so the other coordinates cancel); the truth ratio is computed the same way from the clean formula;
  * rule learning degree L = (1 - predicted ratio) / (1 - true ratio): 1 = the whole collapse learned, 0 = none;
  * the limit = the first level at which the mean L of some rule falls below 0.5 (the learned collapse is less than half the true one,
    an error of more than a factor 2); if no level does, "limit not reached";
  * grid regret = mean over all grid cells of max(ask, act) - value(chosen) (clean values), normalised by min(always-ask, never-ask)
    on the same grid.
"""
import itertools
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core import plots
from tact.core.generate_dataset import KITCHEN_TOPICS, QUESTIONS, availability
from tact.core.grid_eval import TOPICS
from tact.core.model import CONFIG, EMOTIONS, EXTRA_SPEC, RESULTS_DIR, ROOT, FrozenBert, get_device
from tact.core.plots import is_habit, necessity

EXTRA_FACTOR = {"location": ("office", 0.3), "companion": ("with_guests", 0.5), "noise_level": ("noisy", 0.7)}
LEVELS = [  # name, n_vars, data dir, extra variables, generator arguments
    ("3", 3, "data_dpo", (), None),
    ("4", 4, "data_ext", ("location",), "--n 500 --with-location"),
    ("5", 5, "data_ctx5", ("location", "companion"), "--n 500 --extra-vars location,companion"),
    ("6", 6, "data_ctx6", ("location", "companion", "activity"), "--n 500 --extra-vars location,companion,activity"),
    ("7", 7, "data_ctx7", ("location", "companion", "activity", "noise_level"),
     "--n 500 --extra-vars location,companion,activity,noise_level"),
]
EXTRA7 = ("data_ctx7_n1000", ("location", "companion", "activity", "noise_level"),
          "--n 1000 --extra-vars location,companion,activity,noise_level")
RULES = {"location": "location: office × 0.3", "companion": "companion: with_guests × 0.5",
         "activity": "activity × topic: cooking, utensil topics × 0.5", "noise_level": "noise: noisy × 0.7"}


def ensure(ddir, gen_args):
    path = os.path.join(ROOT, ddir)
    if gen_args and not os.path.exists(os.path.join(path, "train.csv")):
        subprocess.run([sys.executable, "-m", "tact.core.generate_dataset", *gen_args.split(), "--seed", "42", "--with-act",
                        "--outdir", path], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    return path


def level_grid(extras):
    """Full factorial grid (12 topics x 6 emotions x 7 days x 3 times x extra values) with clean ask / act values."""
    combos = list(itertools.product(*[EXTRA_SPEC[v] for v in extras])) or [()]
    rows = []
    for t in TOPICS:
        for e in EMOTIONS:
            for d in range(7):
                for m in range(3):
                    for c in combos:
                        rows.append({"topic": t, "question": QUESTIONS[t][0], "emotion": e, "day": d, "time": m, **dict(zip(extras, c))})
    g = pd.DataFrame(rows)
    nec = np.array([necessity(t, d, m) for t, d, m in zip(g.topic, g.day, g.time)])
    if "activity" in g:
        nec = nec * np.where((g.activity == "cooking") & g.topic.isin(KITCHEN_TOPICS), 0.5, 1.0)
    rec = np.array([CONFIG["valence"][e] * availability(d, m) for e, d, m in zip(g.emotion, g.day, g.time)])
    for v, (val, f) in EXTRA_FACTOR.items():
        if v in g:
            rec = rec * np.where(g[v] == val, f, 1.0)
    g["habit"] = [is_habit(t, d, m) for t, d, m in zip(g.topic, g.day, g.time)]
    g["c_ask"], g["c_act"] = nec * rec, 1.0 - nec
    g["score"] = 0.0
    return g


def rule_masks(g, rule):
    """(fires, does-not-fire) boolean masks over the non-habit cells for a rule; the second pair is the leak control."""
    nh = ~g.habit.values
    if rule == "activity":
        k = g.topic.isin(KITCHEN_TOPICS).values
        cook = (g.activity == "cooking").values
        return [(cook & k & nh, ~cook & k & nh), (cook & ~k & nh, ~cook & ~k & nh)]
    val = EXTRA_FACTOR[rule][0]
    on = (g[rule] == val).values
    return [(on & nh, ~on & nh)]


def ratio(x, masks):
    a, b = masks
    return float(x[a].mean() / x[b].mean())


def decision_regret(ask, c_ask, c_act):
    best = np.maximum(c_ask, c_act)
    return float((best - np.where(ask, c_ask, c_act)).mean())


def run_level(enc, dev, name, nv, ddir, extras, n, seeds=B.SEEDS):
    data = B.Data(enc, dev, data_dir=ddir)
    g = level_grid(extras)
    f_grid = B.feats(g, enc, dev)
    ctx = ("emotion", "day", "time", *extras)
    reg_always = decision_regret(np.ones(len(g), bool), g.c_ask.values, g.c_act.values)
    reg_never = decision_regret(np.zeros(len(g), bool), g.c_ask.values, g.c_act.values)
    base = min(reg_always, reg_never)
    tsc = data.tests["seen"]
    tb = B.baselines(tsc)
    tbase = min(tb["always_ask"]["regret"], tb["never_ask"]["regret"])
    total_keys = 12 * 126 * int(np.prod([len(EXTRA_SPEC[v]) for v in extras] or [1]))
    rows = []
    B.MODEL_KW = {"extra_sizes": tuple(len(EXTRA_SPEC[v]) for v in extras)} if extras else {}
    for seed in seeds:
        s = seed * 1000 + n
        ours = B.Ours(data, seed)
        ours.fit_round(n, s, warm=False)
        sc_tr, rm_tr, f_tr = data.prefix(n)
        rm = ours.rm.eval()
        with B.torch.no_grad():
            tr_mse, va_mse = float(B.reward_loss(rm, rm_tr)), float(B.reward_loss(rm, data.rm_val))
            ra_tr, rc_tr = (x.cpu().numpy() for x in B.reward_scores(rm, f_tr))
        # the act text carries no topic, so the act rows have an irreducible error: report ask and act rows separately
        tr_ask = float(np.mean((ra_tr - sc_tr.s_ask.values) ** 2))
        tr_act = float(np.mean((rc_tr - sc_tr.s_act.values) ** 2))
        a, b = ours.decide(f_grid)
        tab = B.Tabular("topic", ctx_cols=ctx)
        tab.update(sc_tr, np.random.default_rng([seed, n]).permutation(n))
        c, _ = tab.decide(g)
        row = {"level": name, "n_vars": nv, "data": ddir, "n": n, "seed": seed, "base_grid": base, "total_keys": total_keys,
               "coverage": len({(r.topic, r.emotion, r.day, r.time, *[getattr(r, v) for v in extras]) for r in sc_tr.itertuples()}) / total_keys,
               "train_mse": tr_mse, "val_mse": va_mse, "train_mse_ask": tr_ask, "train_mse_act": tr_act}
        for m_, d_ in (("A", a), ("B", b), ("C", c)):
            row[f"grid_regret_{m_}"] = decision_regret(d_, g.c_ask.values, g.c_act.values)
            row[f"grid_nr_{m_}"] = row[f"grid_regret_{m_}"] / base
        fa, fb = ours.decide(data.f_test["seen"])
        ct, lv = tab.decide(tsc)
        for m_, d_ in (("A", fa), ("B", fb), ("C", ct)):
            row[f"test_nr_{m_}"] = B.evaluate(d_, tsc)["regret"] / tbase
        row["c_exact_hit_test"] = float(lv[0])
        # rule learning from the reward model's ask scores on the grid
        pred = rm.predict(g)
        for v in extras:
            for j, masks in enumerate(rule_masks(g, v)):
                r_true, r_pred = ratio(g.c_ask.values, masks), ratio(pred, masks)
                if j == 0:
                    row[f"{v}_true"], row[f"{v}_pred"] = r_true, r_pred
                    row[f"{v}_L"] = (1 - r_pred) / (1 - r_true)
                else:
                    row[f"{v}_leak_pred"] = r_pred    # truth: exactly 1
        rows.append(row)
    B.MODEL_KW = {}
    return pd.DataFrame(rows), base


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def make_figure(R, M):
    """The two-panel figure (rule learning degree and normalised regret vs. number of variables)."""
    COL = {"location": "#2a78d6", "companion": "#eb6834", "activity": "#1baf7a", "noise_level": "#6250d6"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for v in ("location", "companion", "activity", "noise_level"):
        pts = [(nv, M[M.level == name][f"{v}_L"]) for name, nv, _, extras, _ in LEVELS if v in extras]
        ax.errorbar([p[0] for p in pts], [p[1].mean() for p in pts], [p[1].std(ddof=1) for p in pts], color=COL[v], marker="o", lw=2,
                    capsize=3, label=RULES[v])
        for nn_, mk in ((350, "s"), (700, "D")):
            x = R[(R.level == "7 (n=1000 data)") & (R.n == nn_)][f"{v}_L"]
            ax.scatter([7.15 + (0.1 if nn_ == 700 else 0)], [x.mean()], marker=mk, facecolors="none", edgecolors=COL[v], s=45, zorder=4)
    ax.axhline(1.0, color=plots.INK2, ls=":", lw=1)
    ax.axhline(0.5, color=plots.INK2, ls="--", lw=1.2)
    ax.text(3.85, 0.51, "limit: L < 0.5", fontsize=8, color=plots.INK2, va="bottom")
    ax.set_xticks([4, 5, 6, 7])
    ax.set_xlim(3.8, 7.5)
    ax.set_xlabel("number of context variables")
    ax.set_ylabel("rule learning degree L (1 = true collapse learned)")
    ax.set_title("Learned model: how much of each rule is learned", fontsize=10, loc="left", color=plots.INK)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax = axes[1]
    LC = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a"}
    LN = {"A": "A  policy (Context MLP)", "B": "B  argmax of reward model", "C": "C  tabular, key over all variables"}
    for m in ("A", "B", "C"):
        pts = [(nv, M[M.level == name][f"grid_nr_{m}"]) for name, nv, *_ in LEVELS]
        ax.errorbar([p[0] for p in pts], [p[1].mean() for p in pts], [p[1].std(ddof=1) for p in pts], color=LC[m], marker="o", lw=2,
                    capsize=3, label=LN[m])
        for nn_, mk in ((350, "s"), (700, "D")):
            x = R[(R.level == "7 (n=1000 data)") & (R.n == nn_)][f"grid_nr_{m}"]
            ax.scatter([7.15 + (0.1 if nn_ == 700 else 0)], [x.mean()], marker=mk, facecolors="none", edgecolors=LC[m], s=45, zorder=4)
    ax.axhline(1.0, color=plots.INK2, ls="--", lw=1.2)
    ax.text(7.45, 1.015, "min(always-ask, never-ask) = 1", fontsize=8, color=plots.INK2, va="bottom", ha="right")
    ax.set_xticks([3, 4, 5, 6, 7])
    ax.set_xlim(2.8, 7.5)
    ax.set_xlabel("number of context variables")
    ax.set_ylabel("grid regret / min(always-ask, never-ask)")
    ax.set_title("Decision regret on the full grid, N = 350", fontsize=10, loc="left", color=plots.INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("Context-dimension scan (mean ± std over 5 seeds). Hollow squares / diamonds at 7: --n 1000 data, N = 350 / 700",
                 x=0.01, ha="left", fontsize=10.5, color=plots.INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "context_scaling.png"), dpi=170)
    fig.savefig(os.path.join(RESULTS_DIR, "context_scaling.pdf"))
    plt.close(fig)



def main():
    if "--plot-only" in sys.argv:       # redraw the figure from the saved runs (no training)
        R = pd.read_csv(os.path.join(RESULTS_DIR, "context_scaling_runs.csv"))
        make_figure(R, R[R.level.isin([l[0] for l in LEVELS])])
        return
    dev = get_device()
    enc = FrozenBert(dev)
    frames, bases = [], {}
    for name, nv, ddir, extras, gen in LEVELS:
        path = ensure(ddir, gen)
        df, base = run_level(enc, dev, name, nv, path, extras, 350)
        frames.append(df)
        bases[name] = base
        print(f"[ctx] level {name} done", flush=True)
    path7 = ensure(EXTRA7[0], EXTRA7[2])
    for n in (350, 700):
        df, base = run_level(enc, dev, "7 (n=1000 data)", 7, path7, EXTRA7[1], n)
        frames.append(df)
        bases[f"7b{n}"] = base
        print(f"[ctx] level 7 on n=1000 data, N={n} done", flush=True)
    R = pd.concat(frames, ignore_index=True)
    R.to_csv(os.path.join(RESULTS_DIR, "context_scaling_runs.csv"), index=False)

    main_levels = [l[0] for l in LEVELS]
    M = R[R.level.isin(main_levels)]

    # ------------------------------------------------ limit
    lim = None
    for name, nv, ddir, extras, gen in LEVELS:
        if any(M[(M.level == name)][f"{v}_L"].mean() < 0.5 for v in extras):
            lim = name
            break

    make_figure(R, M)

    # ------------------------------------------------ report
    out = ["# Context-dimension scan (`context_scaling.py`)\n",
           "Variables are added one by one on top of data_dpo, each with one multiplicative rule: 4 = location (3 values; office × 0.3 on "
           "the receptiveness, the existing data_ext), 5 = companion (alone / with_guests; × 0.5), 6 = activity (relaxing / working / "
           "cooking; cooking halves the necessity for the utensil topics plate, knife, cup, bowl, an interaction rule that changes both "
           "the ask and the act value), 7 = noise level (quiet / noisy; receptiveness × 0.7). Each level is a new `--n 500 --seed 42` "
           "dataset; learned model = BertScorer (Context MLP) with one 8-d embedding per extra variable, SFT + DPO policy as in budget.py; "
           "tabular C keyed by (topic, emotion, day, time, all extra variables). N = 350, 5 seeds, mean ± std. No hyper-parameter was changed.\n",
           "**Definitions (fixed before the runs).** Collapse ratio of a rule = mean predicted ask score where the rule fires / where it does "
           "not, over the non-habit cells of the full factorial grid; the truth ratio is computed in the same way from the clean formula. "
           "Learning degree L = (1 − predicted ratio) / (1 − true ratio) (1 = the whole collapse learned, 0 = none). **Limit** = the first level "
           "at which the mean L of some rule is below 0.5, i.e. the learned collapse is less than half the true one (an error of more than a "
           "factor 2). Grid regret = mean over all grid cells of max(ask, act) − value of the chosen action, normalised by min(always-ask, "
           "never-ask) of the same grid.\n",
           f"## Result: {'limit at level ' + lim + ' variables' if lim else 'limit not reached within 7 variables'}\n"]
    out += ["## 1. Size of the problem\n",
            "| variables | contexts per topic | keys in all (12 topics) | training coverage of the keys (N = 350) | C: test scenarios with an exact key | "
            "grid baseline regret min(always, never) |", "|---|---|---|---|---|---|"]
    for name, nv, ddir, extras, gen in LEVELS:
        x = M[M.level == name]
        out.append(f"| {nv} | {126 * int(np.prod([len(EXTRA_SPEC[v]) for v in extras] or [1]))} | {int(x.total_keys.iloc[0]):,} | "
                   f"{x.coverage.mean():.1%} | {x.c_exact_hit_test.mean():.0%} | {x.base_grid.iloc[0]:.3f} |")
    out += ["\nTraining coverage = distinct (topic, context) keys among the 350 training scenarios / all keys. The oracle acts more and more "
            "often as rules are added (37% of the scenarios with 3 variables, 74% with 7).\n",
            "## 2. How much of each rule is learned (reward model, ask scores on the full grid)\n",
            "Cells: learning degree L (predicted ratio / true ratio). Truth ratios: location 0.30, companion 0.50, activity × topic 0.50 "
            "(utensil topics), noise 0.70. Last column: the leak control of the activity rule, the predicted cooking / not-cooking ratio on the "
            "other topics, where the truth is exactly 1.\n",
            "| variables | location | companion | activity × topic | noise | activity rule on other topics (truth 1.00) |",
            "|---|---|---|---|---|---|"]
    for name, nv, ddir, extras, gen in LEVELS[1:]:
        x = M[M.level == name]
        cells = []
        for v in ("location", "companion", "activity", "noise_level"):
            cells.append(f"{ms(x[f'{v}_L'])} ({x[f'{v}_pred'].mean():.2f} / {x[f'{v}_true'].mean():.2f})" if v in extras else "–")
        leak = f"{ms(x['activity_leak_pred'])}" if "activity" in extras else "–"
        out.append(f"| {nv} | " + " | ".join(cells) + f" | {leak} |")
    out += ["\n## 3. Decision regret on the full grid, N = 350 (normalised)\n",
            "| variables | A policy | B argmax | C tabular | A on the 75 test scenarios | C on the 75 test scenarios |", "|---|---|---|---|---|---|"]
    for name, nv, *_ in LEVELS:
        x = M[M.level == name]
        out.append(f"| {nv} | {ms(x.grid_nr_A)} | {ms(x.grid_nr_B)} | {ms(x.grid_nr_C)} | {ms(x.test_nr_A)} | {ms(x.test_nr_C)} |")
    out += ["\nAbsolute grid regret (the baseline min(always, never) itself changes with the level, so the normalised numbers above move "
            "partly because of the denominator):\n",
            "| variables | baseline | A policy | B argmax | C tabular | A / C |", "|---|---|---|---|---|---|"]
    for name, nv, *_ in LEVELS:
        x = M[M.level == name]
        out.append(f"| {nv} | {x.base_grid.iloc[0]:.3f} | {ms(x.grid_regret_A, 4)} | {ms(x.grid_regret_B, 4)} | {ms(x.grid_regret_C, 4)} | "
                   f"{(x.grid_regret_A.values / x.grid_regret_C.values).mean():.2f} |")
    out += ["\n## 4. Seven variables: data or capacity? (`--n 1000` data, N = 350 vs. N = 700)\n",
            "The n = 500 data have only 350 training scenarios, so the N = 700 run uses a separately generated `--n 1000` dataset (700 training "
            "scenarios). Both N values are run on that dataset, so the comparison is not confounded by the data draw.\n",
            "| N | location L | companion L | activity × topic L | noise L | A normalised grid regret | C normalised grid regret | "
            "train MSE, ask rows | train MSE, act rows | validation MSE (all rows) |", "|---|---|---|---|---|---|---|---|---|---|"]
    S = R[R.level == "7 (n=1000 data)"]
    for nn_ in (350, 700):
        x = S[S.n == nn_].sort_values("seed")
        out.append(f"| {nn_} | " + " | ".join(ms(x[f"{v}_L"]) for v in ("location", "companion", "activity", "noise_level")) +
                   f" | {ms(x.grid_nr_A)} | {ms(x.grid_nr_C)} | {x.train_mse_ask.mean():.4f} | {x.train_mse_act.mean():.4f} | "
                   f"{x.val_mse.mean():.4f} |")
    x3, x7 = S[S.n == 350].sort_values("seed"), S[S.n == 700].sort_values("seed")
    out += ["\nPaired change when doubling N (700 − 350, seed by seed):\n", "| quantity | change |", "|---|---|"]
    for v, nm in (("location_L", "location L"), ("companion_L", "companion L"), ("activity_L", "activity × topic L"),
                  ("noise_level_L", "noise L"), ("grid_nr_A", "A normalised regret"), ("grid_nr_C", "C normalised regret")):
        d = x7[v].values - x3[v].values
        out.append(f"| {nm} | {d.mean():+.3f} ± {d.std(ddof=1):.3f} |")
    out.append("\nReading rule (fixed beforehand): a change larger than its seed std when N doubles points to data limits; no change with an "
               "already low training MSE on the ask rows points to a capacity or architecture limit. (The act rows' MSE is dominated by the "
               "topic-blind act text and cannot fall; it is shown only for completeness.)")
    with open(os.path.join(RESULTS_DIR, "context_scaling.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
