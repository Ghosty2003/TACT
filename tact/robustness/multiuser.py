"""Multi-user personalisation test (results/multiuser.md, results/multiuser_matrix.png).

    python -m tact.robustness.multiuser      # generates data_users/user_01..10 if missing (generate_dataset.py --with-act --user-seed k)

Ten simulated users, each defined by its user seed alone: valence table (default +-0.15, clipped to [0.1, 1.0]; users 9 and 10 are
NON-MONOTONE: sad = 0.85, angry kept low), 4 habit rules drawn from a pool of 10, weekday-morning availability in [0.4, 0.7], weekend fully
open with probability 0.5, an independent 500-scenario draw (stratified over the user's own habits, 70/15/15 split).
Per user the main pipeline is trained independently (Context MLP reward model + SFT + DPO policy, N = 350 scenarios), 5 seeds.
Evaluation on the common 12 x 6 x 7 x 3 = 1512-cell grid under each user's own truth (necessity with the user's habits x the user's
valence x the user's availability; act value 1 - necessity), decision regret normalised by min(always-ask, never-ask) of that truth.
Criteria fixed beforehand (see the md).
"""
import os
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from tact.decision import budget as B
from tact.core.generate_dataset import QUESTIONS
from tact.core.grid_eval import TOPICS
from tact.core.model import EMOTIONS, RESULTS_DIR, ROOT, FrozenBert, get_device
import json

USERS = list(range(1, 11))
NONMONO = (9, 10)
UDIR = lambda k: os.path.join(ROOT, "data_users", f"user_{k:02d}")


def ensure(k):
    if not os.path.exists(os.path.join(UDIR(k), "train.csv")):
        subprocess.run([sys.executable, "-m", "tact.core.generate_dataset", "--n", "500", "--with-act", "--user-seed", str(k),
                        "--outdir", UDIR(k)], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


class UserTruth:
    """Clean ask / act values of one user from its config.json."""

    def __init__(self, k):
        cfg = json.load(open(os.path.join(UDIR(k), "config.json")))
        self.cfg, self.u = cfg, cfg["user"]
        self.val = self.u["valence"]
        self.habits = self.u["habits"]
        self.base = cfg["base_necessity"]
        self.hab_nec = cfg["habit_necessity"]

    def hit(self, h, topic, day, time):
        if h["topic"] != topic:
            return False
        if h.get("weekday_only") and day >= 5:
            return False
        if h.get("weekend_only") and day < 5:
            return False
        if h["day"] is not None and h["day"] != day:
            return False
        return h["time"] is None or h["time"] == time

    def habit_idx(self, topic, day, time):
        for i, h in enumerate(self.habits):
            if self.hit(h, topic, day, time):
                return i
        return -1

    def avail(self, day, time):
        a = self.u
        if day >= 5 and a["weekend_open"]:
            return 1.0
        if time == 0:
            return a["weekday_morning_availability"]
        return 0.80 if time == 1 else 0.90

    def grid(self, g):
        hi = np.array([self.habit_idx(t, d, m) for t, d, m in zip(g.topic, g.day, g.time)])
        nec = np.where(hi >= 0, self.hab_nec, [self.base[t] for t in g.topic])
        rec = np.array([self.val[e] * self.avail(d, m) for e, d, m in zip(g.emotion, g.day, g.time)])
        return nec * rec, 1.0 - nec, hi


def make_grid():
    rows = [{"topic": t, "question": QUESTIONS[t][0], "emotion": e, "day": d, "time": m, "score": 0.0}
            for t in TOPICS for e in EMOTIONS for d in range(7) for m in range(3)]
    return pd.DataFrame(rows)


def regret(ask, ca, cc):
    return float((np.maximum(ca, cc) - np.where(ask, ca, cc)).mean())


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def main():
    for k in USERS:
        ensure(k)
    dev = get_device()
    enc = FrozenBert(dev)
    g = make_grid()
    f_grid = B.feats(g, enc, dev)
    T = {k: UserTruth(k) for k in USERS}
    truth = {k: T[k].grid(g) for k in USERS}                       # (c_ask, c_act, habit index)
    # verification: the truth formula reproduces the clean labels of every user's generated dataset
    label_diff = {}
    for k in USERS:
        ds = pd.read_csv(os.path.join(UDIR(k), "dataset.csv"))
        ask_rows, act_rows = ds[ds.cand == "ask"].reset_index(drop=True), ds[ds.cand == "act"].reset_index(drop=True)
        ca_, cc_, _ = T[k].grid(ask_rows)
        label_diff[k] = max(float(np.abs(ca_ - ask_rows.score_clean.values).max()), float(np.abs(cc_ - act_rows.score_clean.values).max()))
        assert label_diff[k] < 1.1e-3, (k, label_diff[k])
    base = {k: min(regret(np.ones(len(g), bool), *truth[k][:2]), regret(np.zeros(len(g), bool), *truth[k][:2])) for k in USERS}
    oracle = {k: truth[k][0] >= truth[k][1] for k in USERS}
    seeds = B.SEEDS

    # ------------------------------------------------ train every (user, seed)
    pred, decA, decB, test_nr = {}, {}, {}, {}
    for k in USERS:
        data = B.Data(enc, dev, data_dir=UDIR(k))
        ts = data.tests["seen"]
        tb = B.baselines(ts)
        tbase = min(tb["always_ask"]["regret"], tb["never_ask"]["regret"])
        for s in seeds:
            ours = B.Ours(data, s)
            ours.fit_round(350, s * 1000 + 350, warm=False)
            pred[(k, s)] = ours.rm.predict(g)
            decA[(k, s)], decB[(k, s)] = ours.decide(f_grid)
            fa, fb = ours.decide(data.f_test["seen"])
            test_nr[(k, s)] = (B.evaluate(fa, ts)["regret"] / tbase, B.evaluate(fb, ts)["regret"] / tbase)
        print(f"[users] user {k} done", flush=True)

    # ------------------------------------------------ cross matrix: model of user i under the truth of user j
    MA = np.array([[np.mean([regret(decA[(i, s)], *truth[j][:2]) / base[j] for s in seeds]) for j in USERS] for i in USERS])
    MB = np.array([[np.mean([regret(decB[(i, s)], *truth[j][:2]) / base[j] for s in seeds]) for j in USERS] for i in USERS])
    off = ~np.eye(10, dtype=bool)
    diag, offd = np.diag(MA), MA[off]
    crit1a = diag.mean() + diag.std(ddof=1) < offd.mean()

    # ------------------------------------------------ self evaluation
    rows = []
    for k in USERS:
        ca, cc, hi = truth[k]
        nh = hi < 0
        sp = [spearmanr(pred[(k, s)], ca).statistic for s in seeds]
        ratio_p, ratio_t, L = [], [], []
        for r_i in range(len(T[k].habits)):
            cells = hi == r_i
            same_topic_nh = np.array([(t == T[k].habits[r_i]["topic"]) for t in g.topic]) & nh
            rt = ca[cells].mean() / ca[same_topic_nh].mean()
            rp = np.mean([pred[(k, s)][cells].mean() / pred[(k, s)][same_topic_nh].mean() for s in seeds])
            ratio_t.append(rt)
            ratio_p.append(rp)
            L.append((1 - rp) / (1 - rt))
        rows.append({"user": k, "spearman": np.mean(sp), "spearman_sd": np.std(sp, ddof=1),
                     "grid_nr_A": np.mean([regret(decA[(k, s)], ca, cc) / base[k] for s in seeds]),
                     "grid_nr_B": np.mean([regret(decB[(k, s)], ca, cc) / base[k] for s in seeds]),
                     "test_nr_A": np.mean([test_nr[(k, s)][0] for s in seeds]), "test_nr_B": np.mean([test_nr[(k, s)][1] for s in seeds]),
                     "ratio_pred": np.mean(ratio_p), "ratio_true": np.mean(ratio_t), "L": np.mean(L), "base": base[k],
                     "oracle_act": float((~oracle[k]).mean())})
    S = pd.DataFrame(rows)
    S.to_csv(os.path.join(RESULTS_DIR, "multiuser_self.csv"), index=False)
    pd.DataFrame(MA, index=USERS, columns=USERS).to_csv(os.path.join(RESULTS_DIR, "multiuser_matrix_A.csv"))
    pd.DataFrame(MB, index=USERS, columns=USERS).to_csv(os.path.join(RESULTS_DIR, "multiuser_matrix_B.csv"))
    n_sp = int((S.spearman > 0.75).sum())
    crit1 = bool(crit1a and n_sp >= 8)

    # ------------------------------------------------ non-monotone users
    nm_rows = []
    for k in USERS:
        ca, cc, hi = truth[k]
        nh = hi < 0
        em = np.array(g.emotion)
        P = np.mean([pred[(k, s)] for s in seeds], axis=0)
        sad = nh & (em == "sad")
        sp_sad = [spearmanr(pred[(k, s)][sad], ca[sad]).statistic for s in seeds]
        mean_p = {e: P[nh & (em == e)].mean() for e in EMOTIONS}
        sp_em = spearmanr([mean_p[e] for e in EMOTIONS], [T[k].val[e] for e in EMOTIONS]).statistic
        pairs = [("sad", "neutral"), ("sad", "angry"), ("neutral", "angry")]
        ok_pairs = [(mean_p[a] > mean_p[b]) == (T[k].val[a] > T[k].val[b]) for a, b in pairs]
        # per-seed, per-cell agreement of the sad > neutral / sad > angry ordering
        agree = {}
        for b in ("neutral", "angry"):
            vals = []
            for s in seeds:
                p = pred[(k, s)].reshape(len(TOPICS), 6, 7, 3)
                c = ca.reshape(len(TOPICS), 6, 7, 3)
                ie, ib = EMOTIONS.index("sad"), EMOTIONS.index(b)
                m = (hi.reshape(len(TOPICS), 6, 7, 3)[:, ie] < 0)
                vals.append(float(((p[:, ie] > p[:, ib]) == (c[:, ie] > c[:, ib]))[m].mean()))
            agree[b] = np.mean(vals)
        nm_rows.append({"user": k, "sad_cell_spearman": np.mean(sp_sad), "sad_cell_spearman_sd": np.std(sp_sad, ddof=1), "emotion_spearman": sp_em,
                        "sad_pred": mean_p["sad"], "neutral_pred": mean_p["neutral"], "angry_pred": mean_p["angry"],
                        "pairs_ok": int(sum(ok_pairs)), "cell_agree_neutral": agree["neutral"], "cell_agree_angry": agree["angry"],
                        "sad_val": T[k].val["sad"], "neutral_val": T[k].val["neutral"], "angry_val": T[k].val["angry"]})
    NM = pd.DataFrame(nm_rows)
    NM.to_csv(os.path.join(RESULTS_DIR, "multiuser_nonmonotone.csv"), index=False)
    nm = NM[NM.user.isin(NONMONO)]
    crit2a = bool((nm.sad_cell_spearman > 0.5).all())
    crit2b = bool((nm.pairs_ok == 3).all())
    crit2 = crit2a and crit2b

    # ------------------------------------------------ do the learned policies differ where the users differ?
    ora_dis = np.array([[float((oracle[i] != oracle[j]).mean()) for j in USERS] for i in USERS])
    pol_dis = np.array([[np.mean([(decA[(i, s)] != decA[(j, s)]).mean() for s in seeds]) for j in USERS] for i in USERS])
    iu = np.triu_indices(10, 1)
    rho_pairs = spearmanr(ora_dis[iu], pol_dis[iu]).statistic

    # ------------------------------------------------ figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, M, nm_ in ((axes[0], MA, "TACT policy A"), (axes[1], MB, "reward argmax B")):
        im = ax.imshow(M, cmap="viridis_r", vmin=0, vmax=max(MA.max(), MB.max()))
        ax.set_xticks(range(10), [str(u) for u in USERS])
        ax.set_yticks(range(10), [str(u) for u in USERS])
        for i in range(10):
            for j in range(10):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if M[i, j] > 0.55 * max(MA.max(), MB.max()) else "black", fontweight="bold" if i == j else "normal")
            ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.5))
        ax.set_xlabel("user whose truth is used")
        ax.set_ylabel("user whose model is evaluated")
        ax.set_title(f"{nm_}: normalised grid regret (diagonal = own truth)", fontsize=10, loc="left")
        for u in NONMONO:
            ax.get_xticklabels()[u - 1].set_color("#b22")
            ax.get_yticklabels()[u - 1].set_color("#b22")
    fig.colorbar(im, ax=axes, shrink=0.85, label="regret / min(always-ask, never-ask)")
    fig.suptitle("Cross-user evaluation (mean over 5 seeds); users 9 and 10 (red) have non-monotone affect", x=0.01, ha="left", fontsize=11)
    fig.savefig(os.path.join(RESULTS_DIR, "multiuser_matrix.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------ report
    out = ["# Multi-user personalisation test (`multiuser.py`)\n",
           "**Setting.** Ten simulated users, each derived from its user seed alone (`generate_dataset.py --with-act --user-seed k`): valence table = "
           "default ±0.15 uniform, clipped to [0.1, 1.0]; **users 9 and 10 are non-monotone** (sad = 0.85, angry kept ≤ 0.4); 4 habit rules drawn from a "
           "pool of 10 (the 4 default ones + 6 new); weekday-morning availability U[0.4, 0.7]; weekends fully open with probability 0.5 (otherwise "
           "weekends follow the weekday pattern); an independent 500-scenario draw per user (stratified over the user's own habits), 70/15/15 split. "
           "Each user's main pipeline (Context MLP reward model + SFT + DPO, N = 350 scenarios) is trained independently, 5 seeds. Evaluation on the common "
           "1512-cell grid under each user's own truth (the truth formula reproduces the clean labels of every user's dataset, max abs difference "
           f"{max(label_diff.values()):.4f}), regret normalised by min(always-ask, never-ask) of that truth.\n",
           "**Criteria fixed beforehand.** (1) Personalisation holds if the mean of the diagonal of the cross matrix (each user's model under its own truth) "
           "+ 1 std (across the 10 users) is below the mean of the off-diagonal entries, **and** at least 8 of the 10 users have a self-evaluation grid "
           "Spearman > 0.75. (2) The non-monotone mapping is learnable if, for users 9 and 10, (a) the Spearman between predicted and true ask scores over the "
           "sad cells is > 0.5, and (b) the model reproduces the true ordering of the mean ask score among sad, neutral and angry (all three pairs; the "
           "model has no monotonicity prior). (3) If a criterion is not met, report which one and by how much.\n",
           "## The ten users\n",
           "| user | valence (neutral / calm / happy / sad / angry / fearful) | habits (pool index) | weekday-morning availability | weekend open | oracle acts on the grid |",
           "|---|---|---|---|---|---|"]
    for k in USERS:
        u = T[k].u
        v = u["valence"]
        out.append(f"| {k}{' (non-monotone)' if k in NONMONO else ''} | " + " / ".join(f"{v[e]:.2f}" for e in EMOTIONS) +
                   f" | {', '.join(str(i + 1) for i in u['habit_pool_indices'])} | {u['weekday_morning_availability']:.2f} | "
                   f"{'yes' if u['weekend_open'] else 'no'} | {S.loc[S.user == k, 'oracle_act'].iloc[0]:.1%} |")
    out += ["\n## Self-evaluation (each user's model on its own grid and test set; mean over 5 seeds)\n",
            "| user | grid Spearman | grid regret, policy A (normalised) | grid regret, argmax B | test regret A (normalised) | test regret B | "
            "habit collapse ratio predicted / true | learning degree L of the habit collapses |", "|---|---|---|---|---|---|---|---|"]
    for _, r in S.iterrows():
        out.append(f"| {int(r.user)} | {r.spearman:.3f} ± {r.spearman_sd:.3f} | {r.grid_nr_A:.3f} | {r.grid_nr_B:.3f} | {r.test_nr_A:.3f} | {r.test_nr_B:.3f} | "
                   f"{r.ratio_pred:.2f} / {r.ratio_true:.2f} | {r.L:.2f} |")
    out += ["\nL = (1 − predicted ratio) / (1 − true ratio) of the mean ask score on a habit's cells over the other non-habit cells of the same topic, averaged "
            "over the user's 4 rules (1 = the whole collapse learned).\n",
            "### Robustness across users (mean ± std across the 10 users)\n",
            "| quantity | mean ± std over users |", "|---|---|",
            f"| grid Spearman | {ms(S.spearman)} |", f"| grid regret, policy A (normalised) | {ms(S.grid_nr_A)} |",
            f"| grid regret, argmax B (normalised) | {ms(S.grid_nr_B)} |", f"| test regret, policy A (normalised) | {ms(S.test_nr_A)} |",
            f"| test regret, argmax B (normalised) | {ms(S.test_nr_B)} |", f"| habit-collapse learning degree L | {ms(S.L)} |",
            f"\nUsers with grid Spearman > 0.75: **{n_sp} of 10**.\n",
            "## Cross evaluation (`multiuser_matrix.png`)\n",
            "Entry (i, j): normalised grid regret of the policy A of user i evaluated under the truth of user j (mean over 5 seeds). Matrix of A "
            "(rows = model, columns = truth):\n",
            "| model \\ truth | " + " | ".join(str(u) for u in USERS) + " |", "|---|" + "---|" * 10]
    for i in USERS:
        out.append(f"| {i} | " + " | ".join((f"**{MA[i - 1, j - 1]:.2f}**" if i == j else f"{MA[i - 1, j - 1]:.2f}") for j in USERS) + " |")
    ratio_A, ratio_B = offd.mean() / diag.mean(), MB[off].mean() / np.diag(MB).mean()
    out += [f"\n| | diagonal (own truth) | off-diagonal (other users' truth) | off / diag |", "|---|---|---|---|",
            f"| policy A | {diag.mean():.3f} (std over users {diag.std(ddof=1):.3f}) | {offd.mean():.3f} (std {offd.std(ddof=1):.3f}) | {ratio_A:.2f} |",
            f"| argmax B | {np.diag(MB).mean():.3f} (std {np.diag(MB).std(ddof=1):.3f}) | {MB[off].mean():.3f} (std {MB[off].std(ddof=1):.3f}) | {ratio_B:.2f} |",
            f"\nOwn model better than the average other model for {int(sum(MA[i, i] < np.delete(MA[i], i).mean() for i in range(10)))} of 10 users "
            f"(row-wise) and own truth better matched than the average other model on {int(sum(MA[j, j] < np.delete(MA[:, j], j).mean() for j in range(10)))} of 10 "
            "truths (column-wise).\n",
            "**Do the learned policies differ where the users differ?** The share of grid cells in which the oracle decisions of two users differ "
            f"averages {ora_dis[iu].mean():.1%} over the 45 user pairs; for the learned policies (same seed) the disagreement averages {pol_dis[iu].mean():.1%}. "
            f"Across the 45 pairs the two disagreements correlate with a Spearman of {rho_pairs:.2f}.\n",
            "## Non-monotone users (9 and 10)\n",
            "| user | sad-cell Spearman (pred vs true), mean over seeds | emotion-level Spearman (mean prediction per emotion vs valence) | mean predicted ask score: sad / neutral / angry | "
            "true valence: sad / neutral / angry | correct pairs (sad>neutral, sad>angry, neutral>angry) | cells where sad vs neutral / sad vs angry order is right |",
            "|---|---|---|---|---|---|---|"]
    for _, r in NM.iterrows():
        if int(r.user) in NONMONO or int(r.user) in (1, 2):
            out.append(f"| {int(r.user)}{' (non-monotone)' if int(r.user) in NONMONO else ' (monotone reference)'} | {r.sad_cell_spearman:.3f} ± {r.sad_cell_spearman_sd:.3f} | "
                       f"{r.emotion_spearman:.2f} | {r.sad_pred:.3f} / {r.neutral_pred:.3f} / {r.angry_pred:.3f} | {r.sad_val:.2f} / {r.neutral_val:.2f} / {r.angry_val:.2f} | "
                       f"{int(r.pairs_ok)} of 3 | {r.cell_agree_neutral:.0%} / {r.cell_agree_angry:.0%} |")
    out += ["\n(The two monotone users 1 and 2 are shown as a reference; the emotion-level Spearman of all users is in `multiuser_nonmonotone.csv`.)\n",
            "## Verdict by the criteria fixed beforehand\n",
            f"1. **Personalisation: {'holds' if crit1 else 'does not hold'}.** Diagonal mean + 1 std = {diag.mean() + diag.std(ddof=1):.3f} versus off-diagonal mean "
            f"{offd.mean():.3f} → {'satisfied' if crit1a else f'not satisfied (short by {diag.mean() + diag.std(ddof=1) - offd.mean():.3f})'}; users with Spearman > 0.75: "
            f"{n_sp} of 10 → {'satisfied' if n_sp >= 8 else 'not satisfied'}.",
            f"2. **Non-monotone mapping learnable: {'yes' if crit2 else 'no'}.** (a) sad-cell Spearman > 0.5: users 9 / 10 = "
            + " / ".join(f"{x:.3f}" for x in nm.sad_cell_spearman) + f" → {'satisfied' if crit2a else 'not satisfied'}; (b) ordering of sad, neutral and angry reproduced "
            f"(3 pairs): " + " / ".join(f"{int(x)} of 3" for x in nm.pairs_ok) + f" → {'satisfied' if crit2b else 'not satisfied'}.",
            "\nLimits: users differ only through the valence table, the habit rules and the availability (the necessity of the topics and the act value "
            "are shared); the cross matrix is normalised by the baselines of the evaluated truth; 5 seeds per user; one data draw per user."]
    with open(os.path.join(RESULTS_DIR, "multiuser.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
