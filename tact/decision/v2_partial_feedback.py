"""v2: partial feedback in the semi-online loop (results/v2/partial_feedback.md, partial_feedback_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.decision.v2_partial_feedback

Semi-online loop (50 interactions per round, reward model retrained after every round: round 1 from scratch, then warm start, seed `seed * 1000 + 700 + round`, as budget.semi_online), final architecture,
threshold rule, cold start (all 50 interactions of round 1 are asked), 5 seeds. The decisions of a round are made by the model deployed at its start; the labels the round then yields depend on the setting:
  full     both labels (ask and act) of every interaction (the setting of every other experiment);
  reveal   asking reveals the counterfactual: ask -> both labels; act -> only the act label;
  bandit   only the label of the action actually taken;
  eps      bandit + epsilon-greedy: from round 2 on, with probability 0.1 a random action (uniform over {ask, act}), otherwise the threshold rule (round 1 is asked, as in all settings).
Streams: `dpo7` = the first 350 training scenarios of data_dpo (7 rounds, the paper's semi-online setting; the adoption criterion is judged here) and `drift15` = the 750 training scenarios of the 15-round stream of the
habit-drift experiment WITHOUT drift (data_drift15/cont_ctrl, original labels; 15 rounds). Early stopping always uses the full-label validation scenarios of the stream's dataset (an assumption: a validation set with
partial labels was not built). After every round: test regret / U / ask rate of the deployed decision rule on the test scenarios, learning degree L of the Saturday-fruit habit (collapse ratio of the afternoon fruit
heatmap on the 1512-cell grid, L = (1 - ratio) / (1 - 0.15)), and the numbers of ask / act labels obtained.
Adoption criterion (fixed beforehand): 'reveal' at round 7 (dpo7) has a test regret at most 0.005 above 'full' AND a Saturday-fruit L >= 0.5. 'bandit' and 'eps' are references.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd

from tact.decision import budget as B
from tact.core.generate_dataset import ACT_TEXT
from tact.drift.habit_drift import collapse_ratio, truth_grid
from tact.core.model import INPUT_COLS, RESULTS_DIR, ROOT, FrozenBert, get_device, make_tensors

EPS = 0.1
SETTINGS = ["full", "reveal", "bandit", "eps"]
NAME = {"full": "1. full feedback (current)", "reveal": "2. asking reveals the counterfactual", "bandit": "3. strict bandit", "eps": "3b. strict bandit + ε-greedy (ε = 0.1)"}
STREAMS = {"dpo7": (os.path.join(ROOT, "data_dpo"), 7), "drift15": (os.path.join(ROOT, "data_drift15", "cont_ctrl", "old_val"), 15)}


def util(ev):
    return ev["acc"] / (1.0 + (1.0 - ev["act_rate"]))


def run_stream(name, enc, dev, g):
    path, rounds = STREAMS[name]
    data = B.Data(enc, dev, data_dir=path)
    stream = data.sc_train.iloc[:50 * rounds].reset_index(drop=True)
    ts, f_ts = data.tests["seen"], data.f_test["seen"]
    rows = []
    for seed in B.SEEDS:
        for setting in SETTINGS:
            ours = B.Ours(data, seed)
            pool = []
            for rnd in range(1, rounds + 1):
                batch = stream.iloc[50 * (rnd - 1):50 * rnd].reset_index(drop=True)
                f_b = B.feats(batch, enc, dev)
                if rnd == 1:
                    ask = np.ones(len(batch), bool)                              # cold start: ask
                else:
                    ask, _ = ours.decide(f_b)
                    if setting == "eps":
                        rng = np.random.default_rng([seed, rnd, 31])
                        ask = np.where(rng.random(len(batch)) < EPS, rng.random(len(batch)) < 0.5, ask)
                inter = B.evaluate(ask, batch)["regret"]
                n_ask = n_act = 0
                for i, r in batch.iterrows():
                    if setting == "full":
                        give_ask, give_act = True, True
                    elif setting == "reveal":                                      # asked -> both labels; acted -> the act label only
                        give_ask, give_act = bool(ask[i]), True
                    else:                                                           # bandit / eps: the label of the action taken
                        give_ask, give_act = bool(ask[i]), not bool(ask[i])
                    if give_ask:
                        pool.append((r.question, r.emotion, r.day, r.time, r.s_ask))
                        n_ask += 1
                    if give_act:
                        pool.append((ACT_TEXT, r.emotion, r.day, r.time, r.s_act))
                        n_act += 1
                df = pd.DataFrame(pool, columns=["question", "emotion", "day", "time", "score"])
                T = make_tensors(df[INPUT_COLS], enc, dev)
                data.prefix = lambda n, T=T: (None, T, None)
                try:
                    ours.fit_reward(50 * rnd, seed * 1000 + 700 + rnd, warm=rnd > 1)
                finally:
                    del data.prefix
                a, _ = ours.decide(f_ts)
                et = B.evaluate(a, ts)
                pa = ours.rm.predict(g)
                ratio = collapse_ratio(pa, g)
                rows.append({"stream": name, "setting": setting, "seed": seed, "round": rnd, "interaction_regret": inter, "deploy_ask_rate": float(ask.mean()),
                             "ask_labels": n_ask, "act_labels": n_act, "test_regret": et["regret"], "test_U": util(et), "test_ask_rate": 1.0 - et["act_rate"], "test_acc": et["acc"],
                             "L_sat": (1 - ratio) / (1 - 0.15)})
            print(f"[partial] {name} seed {seed} {setting} done: round {rounds} test regret {rows[-1]['test_regret']:.4f}, L {rows[-1]['L_sat']:.2f}", flush=True)
    bl = B.baselines(ts)
    return pd.DataFrame(rows), bl


def main():
    dev = get_device()
    enc = FrozenBert(dev)
    B.Data(enc, dev, data_dir=STREAMS["dpo7"][0])
    g = truth_grid(False)
    out, bls = [], {}
    for name in STREAMS:
        df, bls[name] = run_stream(name, enc, dev, g)
        out.append(df)
        pd.concat(out).to_csv(os.path.join(RESULTS_DIR, "partial_feedback_runs.csv"), index=False)
    report(pd.concat(out), bls)


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if x.std() > 0 else f"{x.mean():.{nd}f}"


def report(R, bls):
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    g = lambda st, s, r, col: R[(R.stream == st) & (R.setting == s) & (R["round"] == r)].sort_values("seed")[col].values
    out = ["# v2: partial feedback in the semi-online loop (`v2_partial_feedback.py`)\n",
           "Final architecture (frozen NLI-BERT mean pooling + Context encoder + rank-16 bilinear head), threshold rule, cold start (the 50 interactions of round 1 are asked), 5 seeds, mean ± std. The reward model is retrained after every "
           "round of 50 interactions (round 1 from scratch, then warm start); the decisions of a round are made by the model deployed at its start.\n",
           "## Settings and assumptions\n",
           "| setting | labels obtained per interaction |", "|---|---|",
           "| 1. full feedback (current) | ask label and act label |", "| 2. asking reveals the counterfactual | asked → ask label and act label; acted → act label only |",
           "| 3. strict bandit | only the label of the action taken |", "| 3b. strict bandit + ε-greedy | as 3, but from round 2 on a random action with probability ε = 0.1 (round 1 is asked) |",
           "\n- **Streams.** `dpo7`: the first 350 training scenarios of `data_dpo` (7 rounds; the paper's semi-online setting; **the adoption criterion is judged on this stream**). `drift15`: the 750 training scenarios of the 15-round stream of the "
           "habit-drift experiment *without* drift (`data_drift15/cont_ctrl`: original labels; denser Saturday-fruit evidence, 8 of every 50 scenarios, than `data_dpo`) for the 15-round numbers; its test set is the 165 test scenarios of the "
           "`--n 1100` data.",
           "- **Validation scenarios** (early stopping) keep both labels in all settings (a validation set with partial labels was not built): an assumption that favours the partial-feedback settings a little.",
           "- **Test metrics** after every round: test regret, U = accuracy / (1 + ask rate) and the ask rate of the deployed decision rule on the test scenarios; **L** = learning degree of the Saturday-fruit habit "
           "(collapse ratio of the afternoon fruit heatmap, 1512-cell grid, L = (1 − ratio)/(1 − 0.15)).",
           f"- Test baselines: `dpo7`: always-ask {bls['dpo7']['always_ask']['regret']:.3f}, never-ask {bls['dpo7']['never_ask']['regret']:.3f}; `drift15`: always-ask {bls['drift15']['always_ask']['regret']:.3f}, never-ask {bls['drift15']['never_ask']['regret']:.3f}.\n"]
    # integrity
    ref = pd.read_csv(os.path.join(RESULTS_DIR, "dpo_semionline.csv"))
    ref = ref[(ref.method == "A") & (ref.test == "seen") & (ref["round"] == 7)].sort_values("seed").regret.values
    mine = g("dpo7", "full", 7, "test_regret")
    out += [f"**Integrity check.** Setting 1 on `dpo7` reproduces the semi-online loop of the paper (`dpo_semionline.csv`, round 7): max abs difference of the per-seed test regret = {np.abs(mine - ref).max():.1e}.\n"]
    # criterion
    d = g("dpo7", "reveal", 7, "test_regret") - g("dpo7", "full", 7, "test_regret")
    L2 = g("dpo7", "reveal", 7, "L_sat")
    c1, c2 = d.mean() <= 0.005, L2.mean() >= 0.5
    out += ["## Decision by the criterion fixed beforehand (setting 2, `dpo7`, round 7)\n",
            f"1. Test regret of setting 2 at most 0.005 above setting 1: {ms(g('dpo7', 'reveal', 7, 'test_regret'), 4)} vs {ms(g('dpo7', 'full', 7, 'test_regret'), 4)}; paired difference {d.mean():+.4f} ± SE {se(d):.4f} → **{'met' if c1 else 'not met'}**.",
            f"2. Saturday-fruit L of setting 2 ≥ 0.5: {ms(L2, 2)} → **{'met' if c2 else 'not met'}**.",
            "", "**" + ("Both conditions hold: the paper can use setting 2 (asking reveals the counterfactual) as the deployment assumption." if c1 and c2 else "The criterion is not met; the gap is reported below.") + "**\n"]
    # summary table 7 and 15 rounds
    for st, rounds_list, ttl in (("dpo7", (7,), "7 rounds (`dpo7`, the paper's semi-online stream)"), ("drift15", (7, 15), "`drift15` stream (15 rounds)")):
        out += [f"## Test metrics after the last rounds: {ttl}\n",
                "| setting | round | test regret | test U | test ask rate | Saturday-fruit L | paired Δ test regret vs setting 1 (mean ± SE) | paired Δ L vs setting 1 |", "|---|---|---|---|---|---|---|---|"]
        for r in rounds_list:
            for s in SETTINGS:
                x = g(st, s, r, "test_regret")
                dd = x - g(st, "full", r, "test_regret")
                dl = g(st, s, r, "L_sat") - g(st, "full", r, "L_sat")
                out.append(f"| {NAME[s]} | {r} | {ms(x, 4)} | {ms(g(st, s, r, 'test_U'))} | {ms(g(st, s, r, 'test_ask_rate'), 2)} | {ms(g(st, s, r, 'L_sat'), 2)} | "
                           f"{'–' if s == 'full' else f'{dd.mean():+.4f} ± {se(dd):.4f}'} | {'–' if s == 'full' else f'{dl.mean():+.2f} ± {se(dl):.2f}'} |")
        out.append("")
    # per round tables
    for st, nr in (("dpo7", 7), ("drift15", 15)):
        out += [f"## Per round: `{st}` (mean over 5 seeds)\n",
                "| round | " + " | ".join(f"{s}: test regret / L" for s in SETTINGS) + " | " + " | ".join(f"{s}: ask / act labels this round" for s in SETTINGS[1:]) + " |", "|---|" + "---|" * (len(SETTINGS) * 2 - 1)]
        for r in range(1, nr + 1):
            cells = [f"{g(st, s, r, 'test_regret').mean():.4f} / {g(st, s, r, 'L_sat').mean():.2f}" for s in SETTINGS]
            lab = [f"{g(st, s, r, 'ask_labels').mean():.1f} / {g(st, s, r, 'act_labels').mean():.1f}" for s in SETTINGS[1:]]
            out.append(f"| {r} | " + " | ".join(cells) + " | " + " | ".join(lab) + " |")
        out.append("")
    # label counts, cumulative
    out += ["## Labels actually obtained (cumulative over the loop, mean ± std over seeds; full feedback = 50 + 50 per round)\n",
            "| stream | setting | ask labels | act labels | share of act labels | share of interactions asked (deployment) |", "|---|---|---|---|---|---|"]
    for st, nr in (("dpo7", 7), ("drift15", 15)):
        for s in SETTINGS:
            x = R[(R.stream == st) & (R.setting == s)]
            a, b = x.groupby("seed").ask_labels.sum().values, x.groupby("seed").act_labels.sum().values
            asked = x.groupby("seed").deploy_ask_rate.mean().values
            out.append(f"| {st} ({nr} rounds) | {NAME[s]} | {ms(a, 0)} | {ms(b, 0)} | {ms(b / (a + b), 2)} | {ms(asked, 2)} |")
    out.append("")
    with open(os.path.join(RESULTS_DIR, "partial_feedback.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
