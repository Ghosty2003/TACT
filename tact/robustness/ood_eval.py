"""OOD / novelty detection for the reward model (results/ood_eval.md, results/ood_roc.png).

    python -m tact.robustness.ood_eval            # after generate_dataset.py --with-act (data_dpo/)

Model: the default BertScorer trained on N = 350 scenarios of data_dpo (ask + act rows), from scratch, seeds 0-4 - the
reward model of budget.py (same seeds), because the downstream check needs both an ask and an act score.
In-distribution (ID): the 75 test scenarios with their training-phrasings (test_seen = the scenarios of test.csv).
OOD sets: (a) OOD-topic: 3 topics that never occur in training (laundry, plants, mail; 3 phrasings each, 20 queries per
topic with uniformly random context = 60 queries); (b) OOD-paraphrase: the 24 hard paraphrases, each paired with the contexts
of the test scenarios of its topic (150 queries); reference only: (c) unseen phrasing 3 of the training topics (75 queries).
Signals (higher = more novel): knn (distance to nearest training latent, L2-normalised fused latent before the head),
maha (Mahalanobis distance on the fused latent, shared covariance + diagonal ridge), negcos (minus max cosine of the
frozen-BERT CLS to the training texts; on the fused latent it would be rank-identical to knn), mcd (variance of 30
MC-dropout predictions). Thresholds: Youden's J on validation + OOD-topic calibration queries.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from scipy.stats import rankdata

from tact.decision import budget as B
from tact.core import plots
from tact.core.generate_dataset import availability
from tact.decision.hard_paraphrase_texts import HARD
from tact.core.model import CONFIG, EMOTIONS, RESULTS_DIR

SIGNALS = ["knn", "maha", "negcos", "mcd"]
SIGNAME = {"knn": "k-NN distance (fused latent)", "maha": "Mahalanobis (fused latent)",
           "negcos": "−max cosine (frozen-BERT CLS)", "mcd": "MC-dropout variance (30 passes)"}
RIDGE_REL = 0.01          # Mahalanobis ridge = RIDGE_REL * mean(diag(cov)); fixed a priori (sensitivity table below)
MC_PASSES = 30
# necessity of the new topics: fixed a priori (no habit rules); r* = necessity * valence * availability as in config.json
OOD_TOPICS = {
    "laundry": (0.75, ["Which cycle should I run for the laundry?", "Which type of fabric softener should I add to the wash?",
                       "Which basket of clothes should I wash first?"]),
    "plants": (0.70, ["Which plant should I water first?", "Which type of fertilizer should I use for the herbs?",
                      "Which pot should I move to the balcony?"]),
    "mail": (0.80, ["Which package should I open first?", "Which letter should I put on your desk?",
                    "Which type of stamp should I use for the postcard?"]),
}
N_PER_OOD_TOPIC = 20


def ood_topic_frame(seed=12345):
    rng = np.random.default_rng(seed)
    rows = []
    for t, (nec, phr) in OOD_TOPICS.items():
        for i in range(N_PER_OOD_TOPIC):
            e, d, m = EMOTIONS[rng.integers(6)], int(rng.integers(7)), int(rng.integers(3))
            rows.append({"topic": t, "phrasing": i % 3 + 1, "question": phr[i % 3], "emotion": e, "day": d, "time": m,
                         "score": 0.0, "c_ask": nec * CONFIG["valence"][e] * availability(d, m), "c_act": 1.0 - nec,
                         "habit": False})
    return pd.DataFrame(rows)


def paraphrase_frame(data):
    parts = []
    for j in (0, 1):
        sc = data.tests["seen"].copy()
        sc["question"] = sc.topic.map(lambda t: HARD[t][j])
        parts.append(sc)
    return pd.concat(parts, ignore_index=True)


def auc(id_s, ood_s):
    """P(score_ood > score_id) + 0.5 P(tie) (Mann-Whitney)."""
    r = rankdata(np.concatenate([id_s, ood_s]))
    n1, n2 = len(id_s), len(ood_s)
    return float((r[n1:].sum() - n2 * (n2 + 1) / 2) / (n1 * n2))


def roc(id_s, ood_s, grid=np.linspace(0, 1, 101)):
    th = np.unique(np.concatenate([id_s, ood_s, [np.inf]]))[::-1]
    tpr = np.array([(ood_s >= t).mean() for t in th])
    fpr = np.array([(id_s >= t).mean() for t in th])
    return np.interp(grid, fpr, tpr)


def youden(id_s, ood_s):
    th = np.unique(np.concatenate([id_s, ood_s]))
    j = np.array([(ood_s >= t).mean() - (id_s >= t).mean() for t in th])
    return float(th[int(np.argmax(j))])           # first maximum = smallest threshold among ties


@torch.no_grad()
def latent(rm, f):
    rm.eval()
    return torch.cat([f["cls_ask"], rm.context_features(f["emotion"], f["day"], f["time"])], dim=1).double().cpu().numpy()


@torch.no_grad()
def mc_var(rm, f, seed):
    rm.train()                                      # dropout active
    torch.manual_seed(seed)
    p = torch.stack([rm(f["cls_ask"], f["emotion"], f["day"], f["time"]) for _ in range(MC_PASSES)])
    rm.eval()
    return p.var(0).cpu().numpy()


class Detector:
    def __init__(self, rm, f_tr, ridge_rel=RIDGE_REL):
        self.rm = rm
        self.lib = latent(rm, f_tr)
        self.lib_n = self.lib / np.linalg.norm(self.lib, axis=1, keepdims=True)
        self.mu = self.lib.mean(0)
        cov = np.cov(self.lib.T)
        cov = cov + ridge_rel * np.mean(np.diag(cov)) * np.eye(cov.shape[0])
        self.prec = np.linalg.inv(cov)
        c = f_tr["cls_ask"].double().cpu().numpy()
        self.lib_cls = c / np.linalg.norm(c, axis=1, keepdims=True)

    def scores(self, f, seed):
        z = latent(self.rm, f)
        zn = z / np.linalg.norm(z, axis=1, keepdims=True)
        d = z - self.mu
        c = f["cls_ask"].double().cpu().numpy()
        c = c / np.linalg.norm(c, axis=1, keepdims=True)
        return {"knn": cdist(zn, self.lib_n).min(1), "maha": np.sqrt(np.einsum("ij,jk,ik->i", d, self.prec, d)),
                "negcos": -(c @ self.lib_cls.T).max(1), "mcd": mc_var(self.rm, f, seed)}


def decide(rm, f, flagged=None):
    """ask? Argmax of the reward model; where flagged the ask score is replaced by the neutral 0.5."""
    ra, rc = B.reward_scores(rm, f)
    ra = ra.cpu().numpy()
    if flagged is not None:
        ra = np.where(flagged, 0.5, ra)
    return ra > rc.cpu().numpy()


def pm(x, nd=3):
    x = np.asarray(x, float)
    return f"{np.nanmean(x):.{nd}f} ± {np.nanstd(x, ddof=1):.{nd}f}"


def main():
    dev = B.get_device()
    data = B.Data(B.FrozenBert(dev), dev)
    sc_topic, sc_para = ood_topic_frame(), paraphrase_frame(data)
    # new texts are encoded AFTER the standard ones: the cached features of the training texts, and so the models,
    # are identical to those of budget.py
    data.enc.encode(sorted(set(sc_topic.question) | set(sc_para.question)))
    sets = {"id_test": (data.tests["seen"], data.f_test["seen"]), "id_val": (data.sc_val, data.f_val),
            "ood_topic": (sc_topic, B.feats(sc_topic, data.enc, dev)),
            "ood_para": (sc_para, B.feats(sc_para, data.enc, dev)),
            "unseen_phr": (data.tests["unseen"], data.f_test["unseen"])}
    _, _, f_tr = data.prefix(350)

    auc_rows, thr_rows, down_rows, sens_rows, roc_store = [], [], [], [], {}
    for seed in B.SEEDS:
        ours = B.Ours(data, seed)
        ours.fit_reward(350, seed * 1000 + 350, warm=False)
        rm = ours.rm
        det = Detector(rm, f_tr)
        S = {k: det.scores(v[1], seed) for k, v in sets.items()}
        for sig in SIGNALS:
            for oname in ("ood_topic", "ood_para", "unseen_phr"):
                auc_rows.append({"seed": seed, "signal": sig, "ood": oname, "auc": auc(S["id_test"][sig], S[oname][sig])})
                roc_store.setdefault((sig, oname), []).append(roc(S["id_test"][sig], S[oname][sig]))
            # stricter task: new topics vs. everything with a known topic (ID test + unseen phrasing of known topics)
            auc_rows.append({"seed": seed, "signal": sig, "ood": "topic_vs_known",
                             "auc": auc(np.concatenate([S["id_test"][sig], S["unseen_phr"][sig]]), S["ood_topic"][sig])})
        # Mahalanobis ridge sensitivity (reported, not used)
        for rr in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
            dd = Detector(rm, f_tr, ridge_rel=rr)
            sc = {k: dd.scores(sets[k][1], seed)["maha"] for k in ("id_test", "ood_topic", "ood_para", "unseen_phr")}
            for oname in ("ood_topic", "ood_para", "unseen_phr"):
                sens_rows.append({"seed": seed, "ridge": rr, "ood": oname, "auc": auc(sc["id_test"], sc[oname])})

        # ---- thresholds and downstream regret
        topics = sc_topic.topic.values
        rng = np.random.default_rng([seed, 11])
        perm = rng.permutation(len(sc_topic))
        cal_a, held_a = perm[:len(perm) // 2], perm[len(perm) // 2:]
        folds = {"half": [(cal_a, held_a)],
                 "loto": [(np.where(topics != t)[0], np.where(topics == t)[0]) for t in OOD_TOPICS]}
        ra_ood, rc_ood = (x.cpu().numpy() for x in B.reward_scores(rm, sets["ood_topic"][1]))
        for scheme, fl in folds.items():
            for sig in SIGNALS:
                tp, fp, tp_p, tp_u, share_id, regs = [], [], [], [], [], []
                flagged_all = np.zeros(len(sc_topic), bool)
                for cal, held in fl:
                    tau = youden(np.concatenate([S["id_val"][sig]]), S["ood_topic"][sig][cal])
                    tp.append((S["ood_topic"][sig][held] >= tau).mean())
                    fp.append((S["id_test"][sig] >= tau).mean())
                    tp_p.append((S["ood_para"][sig] >= tau).mean())
                    tp_u.append((S["unseen_phr"][sig] >= tau).mean())
                    flagged_all[held] = S["ood_topic"][sig][held] >= tau
                thr_rows.append({"seed": seed, "scheme": scheme, "signal": sig, "tpr_ood_topic": np.mean(tp),
                                 "fpr_id": np.mean(fp), "tpr_para": np.mean(tp_p), "tpr_unseen_phr": np.mean(tp_u)})
                # downstream: held-out OOD-topic queries (all 60 for loto, the held half for half)
                held_all = np.concatenate([h for _, h in fl])
                sub = sc_topic.iloc[held_all].reset_index(drop=True)
                f_sub = {k: v[torch.as_tensor(held_all, device=v.device)] for k, v in sets["ood_topic"][1].items()}
                fb = flagged_all[held_all]
                d_reg = B.evaluate(decide(rm, f_sub), sub)["regret"]
                d_fb = B.evaluate(decide(rm, f_sub, fb), sub)["regret"]
                # false-alarm cost on ID test with the (mean) threshold of the folds
                taus = [youden(S["id_val"][sig], S["ood_topic"][sig][cal]) for cal, _ in fl]
                fl_id = np.mean([S["id_test"][sig] >= t for t in taus], axis=0) > 0.5 if len(taus) > 1 else S["id_test"][sig] >= taus[0]
                id_reg = B.evaluate(decide(rm, sets["id_test"][1]), sets["id_test"][0])["regret"]
                id_fb = B.evaluate(decide(rm, sets["id_test"][1], fl_id), sets["id_test"][0])["regret"]
                down_rows.append({"seed": seed, "scheme": scheme, "signal": sig, "ood_regression": d_reg, "ood_fallback": d_fb,
                                  "ood_flag_rate": fb.mean(), "id_regression": id_reg, "id_fallback": id_fb,
                                  "id_flag_rate": fl_id.mean()})
            # references on the same OOD queries
            held_all = np.concatenate([h for _, h in fl])
            sub = sc_topic.iloc[held_all].reset_index(drop=True)
            f_sub = {k: v[torch.as_tensor(held_all, device=v.device)] for k, v in sets["ood_topic"][1].items()}
            n = len(sub)
            for name, dec in (("always-ask", np.ones(n, bool)), ("always-act", np.zeros(n, bool)),
                              ("oracle", (sub.c_ask >= sub.c_act).values)):
                down_rows.append({"seed": seed, "scheme": scheme, "signal": name,
                                  "ood_regression": B.evaluate(dec, sub)["regret"], "ood_fallback": np.nan,
                                  "ood_flag_rate": np.nan, "id_regression": np.nan, "id_fallback": np.nan, "id_flag_rate": np.nan})
        print(f"[ood] seed={seed} done", flush=True)

    A = pd.DataFrame(auc_rows)
    T = pd.DataFrame(thr_rows)
    D = pd.DataFrame(down_rows)
    A.to_csv(os.path.join(RESULTS_DIR, "ood_auc.csv"), index=False)
    T.to_csv(os.path.join(RESULTS_DIR, "ood_threshold.csv"), index=False)
    D.to_csv(os.path.join(RESULTS_DIR, "ood_downstream.csv"), index=False)

    # ------------------------------------------------------------ figure
    COL = {"knn": "#2a78d6", "maha": "#eb6834", "negcos": "#1baf7a", "mcd": "#6250d6"}
    grid = np.linspace(0, 1, 101)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharey=True)
    for ax, (o, ttl) in zip(axes, (("ood_topic", "OOD-topic (60 queries, 3 new topics)"),
                                   ("ood_para", "OOD-paraphrase (150 queries, 24 hard paraphrases)"),
                                   ("unseen_phr", "reference: unseen phrasing 3 (75 queries)"))):
        for sig in SIGNALS:
            curves = np.array(roc_store[(sig, o)])
            a = A[(A.signal == sig) & (A.ood == o)].auc
            ax.plot(grid, curves.mean(0), color=COL[sig], lw=2, label=f"{SIGNAME[sig]}  AUC {a.mean():.2f}")
            ax.fill_between(grid, curves.min(0), curves.max(0), color=COL[sig], alpha=0.10, lw=0)
        ax.plot([0, 1], [0, 1], color=plots.INK2, ls="--", lw=1)
        ax.set_xlabel("false-positive rate (ID test scenarios flagged)")
        ax.set_title(ttl, fontsize=10, loc="left", color=plots.INK)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("true-positive rate (OOD queries flagged)")
    fig.suptitle("ROC of four novelty signals (mean over 5 seeds, band = min–max); ID = 75 test scenarios", x=0.01, ha="left",
                 fontsize=11, color=plots.INK2)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "ood_roc.png"), dpi=170)
    plt.close(fig)

    # ------------------------------------------------------------ report
    out = ["# OOD / novelty detection (`ood_eval.py`)\n",
           "Model: default BertScorer, N = 350 scenarios of data_dpo (ask and act rows), from scratch, 5 seeds (the budget.py reward "
           "models). ID = the 75 test scenarios with their trained phrasings (test_seen; the scenarios of test.csv). All numbers: "
           "mean ± std over 5 seeds.\n",
           "- OOD-topic: laundry, plants, mail (3 phrasings each, 20 queries per topic, uniformly random context, 60 queries). Their "
           "ground truth uses the config formula with necessity 0.75 / 0.70 / 0.80 (chosen a priori, no habit rule).\n"
           "- OOD-paraphrase: the 24 hard paraphrases, each with the contexts of the test scenarios of its topic (150 queries).\n"
           "- Reference (not asked for): unseen phrasing 3 of the training topics (75 queries). It shows whether a detector "
           "measures *topic* novelty or merely *text* novelty. Accordingly the AUC table has one more column: OOD-topic versus "
           "all queries with a known topic (ID test + unseen phrasing).\n",
           "## AUC (ID = 0, OOD = 1)\n", "| signal | OOD-topic | OOD-paraphrase | reference: unseen phrasing 3 | OOD-topic vs ID + unseen phrasing |", "|---|---|---|---|---|"]
    for sig in SIGNALS:
        out.append(f"| {SIGNAME[sig]} | " + " | ".join(pm(A[(A.signal == sig) & (A.ood == o)].auc)
                                                       for o in ("ood_topic", "ood_para", "unseen_phr", "topic_vs_known")) + " |")
    out.append("\nThe frozen-BERT-CLS signal has no seed dependence (std 0). Every ID query uses one of the 24 training texts "
               "verbatim, so any signal that sees the text can separate ID from *any* new text: a high AUC on the reference "
               "column means the detector would also fire on harmless rephrasings.\n")
    out += ["## Automatic threshold (Youden's J)\n",
            "τ maximises TPR − FPR on the 75 validation scenarios (negatives) plus the calibration OOD-topic queries (positives). "
            "“half”: random half of the 60 OOD-topic queries for calibration, the other half held out (topics are shared between the "
            "halves). “leave-one-topic-out”: calibrate on two new topics, test on the third (mean over the three folds; closer to "
            "meeting a really new topic). FPR is measured on the 75 ID test scenarios. The paraphrase and unseen-phrasing columns "
            "apply the same τ (no calibration on them).\n",
            "| scheme | signal | TPR OOD-topic (held out) | FPR ID test | TPR OOD-paraphrase | TPR unseen phrasing 3 |", "|---|---|---|---|---|---|"]
    for scheme, nm in (("half", "random half"), ("loto", "leave-one-topic-out")):
        for sig in SIGNALS:
            g = T[(T.scheme == scheme) & (T.signal == sig)]
            out.append(f"| {nm} | {SIGNAME[sig]} | {pm(g.tpr_ood_topic)} | {pm(g.fpr_id)} | {pm(g.tpr_para)} | {pm(g.tpr_unseen_phr)} |")
    out += ["\n## Downstream: neutral fallback vs. using the regression score\n",
            "Decision = ask iff the ask score exceeds the act score of the same reward model. “Fallback”: if the detector flags a "
            "query, its ask score is replaced by the neutral 0.5 (then ask iff 0.5 > act score). Regret uses the true r* of the "
            "OOD topics (necessity × valence × availability; act = 1 − necessity). ID columns show the cost of false alarms on the 75 "
            "ID test scenarios (flag threshold from the same calibration; for leave-one-topic-out a scenario is flagged if most of the "
            "three fold thresholds flag it).\n",
            "| scheme | strategy | regret on held-out OOD-topic queries | flag rate on OOD | regret on ID test | flag rate on ID |",
            "|---|---|---|---|---|---|"]
    for scheme, nm in (("half", "random half (30 held-out queries)"), ("loto", "leave-one-topic-out (60 queries)")):
        g0 = D[(D.scheme == scheme) & (D.signal == "knn")]
        out.append(f"| {nm} | regression score (no fallback) | {pm(g0.ood_regression)} | – | {pm(g0.id_regression)} | – |")
        for sig in SIGNALS:
            g = D[(D.scheme == scheme) & (D.signal == sig)]
            out.append(f"| {nm} | fallback, {SIGNAME[sig]} | {pm(g.ood_fallback)} | {pm(g.ood_flag_rate, 2)} | "
                       f"{pm(g.id_fallback)} | {pm(g.id_flag_rate, 2)} |")
        for name in ("always-ask", "always-act", "oracle"):
            g = D[(D.scheme == scheme) & (D.signal == name)]
            out.append(f"| {nm} | {name} | {pm(g.ood_regression)} | – | – | – |")
    out += ["\n## Mahalanobis ridge sensitivity (AUC; reported only, the main table uses ridge = 0.01 × mean variance, fixed a priori)\n",
            "| ridge (× mean variance) | OOD-topic | OOD-paraphrase | unseen phrasing 3 |", "|---|---|---|---|"]
    Sdf = pd.DataFrame(sens_rows)
    for rr in sorted(Sdf.ridge.unique()):
        out.append(f"| {rr:g} | " + " | ".join(pm(Sdf[(Sdf.ridge == rr) & (Sdf.ood == o)].auc) for o in ("ood_topic", "ood_para", "unseen_phr")) + " |")
    out.append("\nThe AUC is 1.000 at every ridge value: the 350 training latents live in 832 dimensions (the text part takes "
               "only 24 distinct values), so any new text lies far outside the training span whatever the ridge, and the "
               "ridge cannot change the ranking of ID against any new text.")
    with open(os.path.join(RESULTS_DIR, "ood_eval.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
