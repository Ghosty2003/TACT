"""v2: what to do with flagged new-topic scenarios, leave the score (A) or ask (B)? (results/v2/ood_ask.md, ood_ask_runs.csv)

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.robustness.v2_ood_ask

Reward model: the N = 350 model of budget.py / ood_eval.py (final architecture, threshold rule), seeds 0-4. Detectors: Mahalanobis (fused latent) and -max cosine of the encoder output (ood_eval.Detector).
 A (current): a flagged scenario is decided as usual (ask iff r_hat(ask) > r_hat(act)), its score unchanged.
 B: a flagged scenario is simply asked; the question asked is the candidate question with the highest r_hat(ask) (for a new topic the three phrasings of the topic; the true ask value does not depend
    on the wording in this generator, so the choice of the question does not change the regret and B is, on flagged scenarios, "ask").
Threshold (re-calibrated; Youden's J): negatives = the 75 validation scenarios + the same scenarios with the unseen phrasing 3 (150); positives = the calibration queries ood_topic_frame(12345) (60, 3 new
topics). The hard paraphrases take no part in the calibration. Evaluation:
 * new topics: (i) 'random half': calibrate on a random half of the 60 queries (positives), evaluate on the other 30; (ii) 'leave-one-topic-out': calibrate on two topics, evaluate on the third (60 queries
   over the three folds); (iii) ood_topic_frame(4242): 60 other queries of the same topics, threshold calibrated on all 60 calibration queries;
 * known topics, at the threshold calibrated on all 60 calibration queries: the 75 test scenarios in the original wording, in the unseen phrasing 3, and with the 24 hard paraphrases (2 × 75 queries).
References: always-ask, never-ask.
Adoption criterion (fixed beforehand): B is adopted for a detector if (a) B's regret on the new topics is lower than A's with a paired difference (B - A, over the 5 seeds) larger than 1 standard error, in each of the
three new-topic evaluations, and (b) on the known topics, B's regret exceeds A's by at most 0.005 in the original wording, the unseen phrasing and the hard paraphrases (mean of B - A over seeds). If both detectors
qualify, the one with the lower flagged share on the validation scenarios (original + unseen wording) is chosen; the detector is chosen on calibration/validation data only. If none qualifies, A is kept.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from tact.decision import budget as B
from tact.core.generate_dataset import QUESTIONS
from tact.core.hard_paraphrase_texts import HARD
from tact.core.model import RESULTS_DIR, FrozenBert, get_device
from tact.robustness.ood_eval import OOD_TOPICS, Detector, decide, ood_topic_frame, paraphrase_frame, youden

DET = {"maha": "Mahalanobis (fused latent)", "negcos": "−max cosine (encoder output)"}
SETS_KNOWN = ["seen", "unseen", "hard"]


def run():
    dev = get_device()
    data = B.Data(FrozenBert(dev), dev)
    enc = data.enc
    cal, ev = ood_topic_frame(12345), ood_topic_frame(4242)
    para = paraphrase_frame(data)
    enc.encode(sorted(set(cal.question) | set(ev.question) | set(para.question) | {h for t in HARD.values() for h in t}))
    val_un = data.sc_val.copy()
    val_un["question"] = val_un.topic.map(lambda t: QUESTIONS[t][2])
    F = lambda sc: B.feats(sc, enc, dev)
    f_cal, f_ev, f_para, f_valun = F(cal), F(ev), F(para), F(val_un)
    _, _, f_tr = data.prefix(350)
    known = {"seen": (data.tests["seen"], data.f_test["seen"]), "unseen": (data.tests["unseen"], data.f_test["unseen"]), "hard": (para, f_para)}
    topics = cal.topic.values
    rows = []
    for seed in B.SEEDS:
        ours = B.Ours(data, seed)
        ours.fit_reward(350, seed * 1000 + 350, warm=False)
        rm = ours.rm
        det = Detector(rm, f_tr)
        S = {k: det.scores(f, seed) for k, f in (("val", data.f_val), ("valun", f_valun), ("cal", f_cal), ("ev", f_ev), ("seen", known["seen"][1]), ("unseen", known["unseen"][1]), ("hard", f_para))}
        rng = np.random.default_rng([seed, 11])
        perm = rng.permutation(len(cal))
        folds = {"half": [(perm[:30], perm[30:])], "loto": [(np.where(topics != t)[0], np.where(topics == t)[0]) for t in OOD_TOPICS]}

        def regrets(sc, f, flagged):
            a = decide(rm, f)
            b = np.where(flagged, True, a)
            return B.evaluate(a, sc)["regret"], B.evaluate(b, sc)["regret"]

        for sig in DET:
            neg = np.concatenate([S["val"][sig], S["valun"][sig]])
            tau = youden(neg, S["cal"][sig])
            base = {"seed": seed, "detector": sig, "tau": tau, "val_flag": float((neg >= tau).mean()), "cal_tpr": float((S["cal"][sig] >= tau).mean())}
            # new topics: random half and leave-one-topic-out
            for scheme, fl in folds.items():
                held_all = np.concatenate([h for _, h in fl])
                flagged = np.zeros(len(cal), bool)
                for c_idx, h_idx in fl:
                    t = youden(neg, S["cal"][sig][c_idx])
                    flagged[h_idx] = S["cal"][sig][h_idx] >= t
                sub = cal.iloc[held_all].reset_index(drop=True)
                f_sub = {k: v[torch.as_tensor(held_all, device=v.device)] for k, v in f_cal.items()}
                ra, rb = regrets(sub, f_sub, flagged[held_all])
                n = len(sub)
                rows.append({**base, "set": f"new:{scheme}", "regret_A": ra, "regret_B": rb, "flag": float(flagged[held_all].mean()),
                             "ask_all": B.evaluate(np.ones(n, bool), sub)["regret"], "never_ask": B.evaluate(np.zeros(n, bool), sub)["regret"]})
            # new topics, held-out frame 4242
            fl = S["ev"][sig] >= tau
            ra, rb = regrets(ev, f_ev, fl)
            rows.append({**base, "set": "new:4242", "regret_A": ra, "regret_B": rb, "flag": float(fl.mean()), "ask_all": B.evaluate(np.ones(len(ev), bool), ev)["regret"],
                         "never_ask": B.evaluate(np.zeros(len(ev), bool), ev)["regret"]})
            # known topics
            for k, (sc, f) in known.items():
                fl = S[k][sig] >= tau
                ra, rb = regrets(sc, f, fl)
                rows.append({**base, "set": f"known:{k}", "regret_A": ra, "regret_B": rb, "flag": float(fl.mean()), "ask_all": B.evaluate(np.ones(len(sc), bool), sc)["regret"],
                             "never_ask": B.evaluate(np.zeros(len(sc), bool), sc)["regret"]})
        print(f"[ood_ask] seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}"


def report(R):
    se = lambda d: d.std(ddof=1) / np.sqrt(len(d))
    g = lambda sig, s, col: R[(R.detector == sig) & (R.set == s)].sort_values("seed")[col].values
    NEW = [("new:half", "new topics, random half (30 held-out queries)"), ("new:loto", "new topics, leave-one-topic-out (60 queries)"), ("new:4242", "new topics, held-out frame `ood_topic_frame(4242)` (60 queries)")]
    KN = [("known:seen", "known topics, original wording (75 test scenarios)"), ("known:unseen", "known topics, unseen phrasing 3 (75)"), ("known:hard", "known topics, hard paraphrases (2 × 75 queries)")]
    out = ["# v2: leave the score of flagged scenarios (A) or ask (B)? (`v2_ood_ask.py`)\n",
           "Final architecture, threshold rule, N = 350 reward model, 5 seeds, mean ± std. **A** = a flagged scenario is decided as usual (score unchanged); **B** = a flagged scenario is asked (the question with the highest "
           "predicted ask score; in this generator the true ask value does not depend on the wording, so the choice of the question cannot change the regret and B = “ask” on flagged scenarios).\n",
           "## Design\n",
           "- **Detectors:** Mahalanobis on the fused latent and −max cosine of the encoder output, each with its own threshold (Youden's J).",
           "- **Calibration (re-done):** negatives = the 75 validation scenarios + the same scenarios with the unseen phrasing 3 (150 queries); positives = the calibration queries `ood_topic_frame(12345)` (60, three new topics). "
           "The 24 hard paraphrases take no part in the calibration.",
           "- **New-topic evaluations:** random half (calibrate on a random half of the calibration positives, evaluate on the other 30), leave-one-topic-out (calibrate on two topics, evaluate on the third, 60 queries in total) and "
           "the held-out frame `ood_topic_frame(4242)` (60 queries of the same topics, threshold from all 60 calibration queries).",
           "- **Known-topic evaluations** (threshold from all 60 calibration queries): the 75 test scenarios in the original wording, in the unseen phrasing 3, and with the 24 hard paraphrases (150 queries).",
           "- **Adoption criterion (fixed beforehand):** for a detector, B is adopted if (a) on the new topics B's regret is lower than A's with a paired difference (over seeds) larger than one standard error, in **each** of the three new-topic "
           "evaluations (my reading of the criterion), and (b) on the known topics B's regret exceeds A's by at most 0.005 in the original wording, in the unseen phrasing and on the hard paraphrases. If both detectors qualify, the one "
           "with the lower flagged share on the validation scenarios is chosen (selection on calibration/validation data only). Otherwise A is kept.\n",
           "## Thresholds and calibration quality\n", "| detector | threshold τ (mean over seeds) | share of the 150 validation negatives flagged | share of the 60 calibration positives flagged |", "|---|---|---|---|"]
    for sig, nm in DET.items():
        x = R[(R.detector == sig) & (R.set == "known:seen")].sort_values("seed")
        out.append(f"| {nm} | {x.tau.mean():.3f} | {x.val_flag.mean():.1%} ± {x.val_flag.std(ddof=1):.1%} | {x.cal_tpr.mean():.1%} ± {x.cal_tpr.std(ddof=1):.1%} |")
    out += ["\n## New topics: regret of A and B, paired difference (B − A; negative = B better)\n",
            "| evaluation | detector | share flagged | A (score unchanged) | B (ask) | B − A (mean ± SE) | seeds where B is better | always-ask | never-ask |", "|---|---|---|---|---|---|---|---|---|"]
    for s, nm in NEW:
        for sig, dn in DET.items():
            a, b = g(sig, s, "regret_A"), g(sig, s, "regret_B")
            d = b - a
            out.append(f"| {nm} | {dn} | {ms(g(sig, s, 'flag'), 2)} | {ms(a)} | {ms(b)} | {d.mean():+.4f} ± {se(d):.4f} | {int((d < 0).sum())} of 5 | {ms(g(sig, s, 'ask_all'))} | {ms(g(sig, s, 'never_ask'))} |")
    out += ["\n## Known topics: flagged share and regret of A and B (B − A: positive = B worse)\n",
            "| evaluation | detector | share flagged | A (score unchanged) | B (ask) | B − A (mean ± SE) | always-ask | never-ask |", "|---|---|---|---|---|---|---|---|"]
    for s, nm in KN:
        for sig, dn in DET.items():
            a, b = g(sig, s, "regret_A"), g(sig, s, "regret_B")
            d = b - a
            out.append(f"| {nm} | {dn} | {ms(g(sig, s, 'flag'), 2)} | {ms(a)} | {ms(b)} | {d.mean():+.4f} ± {se(d):.4f} | {ms(g(sig, s, 'ask_all'))} | {ms(g(sig, s, 'never_ask'))} |")
    out += ["\n## Decision by the criterion fixed beforehand\n", "| detector | (a) B better on new topics by > 1 SE in all three evaluations | (b) B − A ≤ 0.005 on seen / unseen / hard | validation flagged share | adopt B? |", "|---|---|---|---|---|"]
    ok = {}
    for sig, dn in DET.items():
        ca = []
        for s, _ in NEW:
            d = g(sig, s, "regret_B") - g(sig, s, "regret_A")
            ca.append(d.mean() < 0 and abs(d.mean()) > se(d))
        cb_vals = [(g(sig, s, "regret_B") - g(sig, s, "regret_A")).mean() for s, _ in KN]
        cb = all(v <= 0.005 for v in cb_vals)
        vf = R[(R.detector == sig) & (R.set == "known:seen")].val_flag.mean()
        ok[sig] = (all(ca), cb, vf)
        out.append(f"| {dn} | {'yes' if all(ca) else 'no'} ({', '.join('yes' if c else 'no' for c in ca)}) | {'yes' if cb else 'no'} ({', '.join(f'{v:+.3f}' for v in cb_vals)}) | {vf:.1%} | **{'yes' if all(ca) and cb else 'no'}** |")
    qual = [s for s, (a, b, _) in ok.items() if a and b]
    if qual:
        pick = min(qual, key=lambda s: ok[s][2])
        verdict = f"B is adopted with the detector **{DET[pick]}** (lower validation flagged share among the qualifying detectors)."
    else:
        verdict = "**No detector satisfies the criterion: A (leave the score unchanged) is kept.**"
    out += ["", verdict, ""]
    with open(os.path.join(RESULTS_DIR, "ood_ask.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    R = run()
    R.to_csv(os.path.join(RESULTS_DIR, "ood_ask_runs.csv"), index=False)
    report(R)
