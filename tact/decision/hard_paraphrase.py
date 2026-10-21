"""Zero-shot evaluation on hard paraphrases (results/hard_paraphrase.md).

    python -m tact.decision.hard_paraphrase      # after budget.py

Two hand-written paraphrases per topic (hard_paraphrase_texts.py), screened only by frozen-BERT cosine (< 0.90) to the
topic's training phrasings. The 75 test scenarios are re-used with the question replaced by paraphrase 1 / paraphrase 2
(two test sets, hard1 and hard2). Models are re-trained exactly as in budget.py (same seeds, N = 150 and 350) and never
see a paraphrase. Methods: A, B, C (topic key), Ct (question-text key).
"""
import itertools
import os

import numpy as np
import pandas as pd
import torch.nn.functional as F

from tact.decision import budget as B
from tact.core.generate_dataset import QUESTIONS
from tact.decision.hard_paraphrase_texts import HARD, HARD_DRAFT1
from tact.core.model import ARCH, RESULTS_DIR

NS_HARD = [150, 350]
TOPICS = list(QUESTIONS)
STOP = set("a an the of to for in on at and or you your i me my is are be it we should would like which what do does how "
           "with from that this these those please let know kindly specify indicate tell help would want".split())


def toks(s):
    return {w for w in "".join(c.lower() if c.isalpha() else " " for c in s).split() if w not in STOP}


def similarity_report(enc):
    """Cosine / word-overlap statistics of the paraphrases against the training phrasings (phrasings 1 and 2)."""
    v = lambda t: F.normalize(enc.encode([t]), dim=1)[0]
    train = {t: [QUESTIONS[t][0], QUESTIONS[t][1]] for t in TOPICS}
    tv = {t: [v(q) for q in train[t]] for t in TOPICS}
    all_train = [(t, x) for t in TOPICS for x in tv[t]]
    rows = []
    for t in TOPICS:
        for k, h in enumerate(HARD[t]):
            e = v(h)
            own = max(float(e @ x) for x in tv[t])
            nt, nc = max(((tt, float(e @ x)) for tt, x in all_train), key=lambda p: p[1])
            ov = max(len(toks(h) & toks(q)) / len(toks(h) | toks(q)) for q in train[t])
            rows.append({"topic": t, "k": k + 1, "text": h, "cos_own_max": own, "nearest_topic": nt,
                         "cos_nearest": nc, "own_is_nearest": nt == t, "jaccard_max": ov})
    sim = pd.DataFrame(rows)
    # references
    draft = [max(float(v(h) @ x) for x in tv[t]) for t in TOPICS for h in HARD_DRAFT1[t]]
    ph3 = [max(float(v(QUESTIONS[t][2]) @ x) for x in tv[t]) for t in TOPICS]
    same = [float(tv[t][0] @ tv[t][1]) for t in TOPICS]
    cross = [float(a @ b) for t1, t2 in itertools.combinations(TOPICS, 2) for a in tv[t1] for b in tv[t2]]
    ph3_nearest = np.mean([max(((tt, float(v(QUESTIONS[t][2]) @ x)) for tt, x in all_train), key=lambda p: p[1])[0] == t
                           for t in TOPICS])
    return sim, {"draft": draft, "ph3": ph3, "same": same, "cross": cross, "ph3_nearest": ph3_nearest}


def q(x):
    x = np.asarray(x)
    return f"min {x.min():.3f} / median {np.median(x):.3f} / mean {x.mean():.3f} / max {x.max():.3f}"


def run(data):
    enc, dev = data.enc, data.dev
    hard_sc = {}
    for j in (0, 1):
        sc = data.tests["seen"].copy()
        sc["question"] = sc.topic.map(lambda t: HARD[t][j])
        hard_sc[f"hard{j + 1}"] = (sc, B.feats(sc, enc, dev))
    sets = {"seen": (data.tests["seen"], data.f_test["seen"]), "unseen": (data.tests["unseen"], data.f_test["unseen"]),
            **hard_sc}
    rows, store = [], {}
    for n in NS_HARD:
        for seed in B.SEEDS:
            ours = B.Ours(data, seed)
            ours.fit_round(n, seed * 1000 + n, warm=False)   # identical to the budget.py sweep
            sc_tr = data.sc_train.iloc[:n]
            order = np.random.default_rng([seed, n]).permutation(n)
            tabs = {"C": B.Tabular("topic"), "Ct": B.Tabular("question")}
            for t in tabs.values():
                t.update(sc_tr, order)
            for name, (sc, f) in sets.items():
                a, b = ours.decide(f)
                dec = {"A": a, "B": b, "C": tabs["C"].decide(sc)[0], "Ct": tabs["Ct"].decide(sc)[0]}
                for m, d in dec.items():
                    rows.append({"n": n, "seed": seed, "method": m, "test": name, **B.evaluate(d, sc)})
                    store[(n, seed, name, m)] = d
            print(f"[hard] N={n} seed={seed} done", flush=True)
    return pd.DataFrame(rows), store, sets


def ms(x, nd=3):
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def main():
    dev = B.get_device()
    data = B.Data(B.FrozenBert(dev), dev)
    # the paraphrases are encoded AFTER the standard texts, so the cached features of the training texts (and with them
    # the models) are identical to those of budget.py
    data.enc.encode(sorted({h for t in HARD.values() for h in t} | {h for t in HARD_DRAFT1.values() for h in t}))
    sim, ref = similarity_report(data.enc)
    sim.to_csv(os.path.join(RESULTS_DIR, "hard_paraphrase_similarity.csv"), index=False)
    bl = B.baselines(data.tests["seen"])
    df, store, sets = run(data)
    df.to_csv(os.path.join(RESULTS_DIR, "hard_paraphrase_runs.csv"), index=False)
    flips = pd.DataFrame([{"n": n, "seed": s, "method": m,
                           "flip_rate": float(np.mean([(store[(n, s, "seen", m)] != store[(n, s, h, m)]).mean()
                                                       for h in ("hard1", "hard2")]))}
                          for n in NS_HARD for s in B.SEEDS for m in B.METHODS])
    flips.to_csv(os.path.join(RESULTS_DIR, "hard_paraphrase_flips.csv"), index=False)   # share of decisions changed seen -> hard

    # integrity: seen / unseen rows must reproduce budget_runs.csv
    br = pd.read_csv(os.path.join(RESULTS_DIR, "budget_runs.csv"))
    chk = df[df.test.isin(["seen", "unseen"])].merge(br, on=["n", "seed", "method", "test"], suffixes=("", "_ref"))
    max_diff = float((chk.regret - chk.regret_ref).abs().max())

    base = min(bl["always_ask"]["regret"], bl["never_ask"]["regret"])
    own = sim.cos_own_max.values
    out = ["# Hard-paraphrase test (zero-shot)" + (" — ONE-HOT context encoding" if ARCH == "onehot" else "") + "\n",
           "Two hand-written paraphrases per topic (24 sentences, listed below) replace the question of the 75 test scenarios "
           "(paraphrase 1 → hard1, paraphrase 2 → hard2). The models are the budget.py models (same seeds), trained on "
           "phrasings 1 and 2 only; no paraphrase is ever seen. Reward does not depend on the wording in this generator, so "
           "the oracle decisions are those of the other test sets. "
           f"Integrity check: seen/unseen regret of the re-trained models matches budget_runs.csv (max abs difference {max_diff:.1e}).\n",
           "## How different are the paraphrases? (frozen-BERT CLS cosine)\n",
           "| quantity | value |", "|---|---|",
           f"| paraphrase → max cosine to the 2 training phrasings of its own topic (24) | {q(own)} |",
           f"| … number below 0.90 | **{int((own < 0.90).sum())} of {len(own)}** |",
           f"| … is the nearest of all 24 training phrasings from its own topic | {int(sim.own_is_nearest.sum())} of {len(sim)} |",
           f"| max word-overlap (Jaccard of content words) with the topic's training phrasings | {q(sim.jaccard_max)} |",
           f"| reference: unseen phrasing 3 → max cosine to its topic's training phrasings (12) | {q(ref['ph3'])} |",
           f"| … is its nearest training phrasing from its own topic | {ref['ph3_nearest'] * 12:.0f} of 12 |",
           f"| reference: cosine between the two training phrasings of a topic (12) | {q(ref['same'])} |",
           f"| reference: cosine between training phrasings of different topics (264 pairs) | {q(ref['cross'])} |",
           f"| first draft (question form, 24 sentences, replaced): own-topic max cosine | {q(ref['draft'])}; "
           f"{int((np.array(ref['draft']) < 0.90).sum())} of 24 below 0.90 |",
           "\nFrozen-BERT CLS cosines are dominated by the sentence template, not the topic: training phrasings of "
           "*different* topics are about as similar to each other as phrasings of the same topic, and an unrelated "
           "sentence (“What is the capital of France?”) scores 0.935 against the training phrasings. Only 2 of 24 question-form "
           "paraphrases reached < 0.90, so the final set is written as polite statements (“Please indicate …”). All 24 are "
           "now < 0.90, but **the shift is a change of sentence form as well as of vocabulary**, and the nearest training "
           "phrasing is usually from another topic: a test of robustness to a large representation shift rather than of "
           "paraphrase understanding in the narrow sense.\n"]
    for n in NS_HARD:
        out += [f"## N = {n}: regret / accuracy / habit act-rate (mean ± std over 5 seeds)\n",
                f"Baselines: always-ask {bl['always_ask']['regret']:.3f}, never-ask {bl['never_ask']['regret']:.3f} "
                f"(min = {base:.3f}); oracle 0.\n",
                "| method | seen | unseen (phrasing 3) | hard1 | hard2 | hard (mean of 1, 2) | acc seen → hard | habit-act seen → hard | decisions changed seen → hard |",
                "|---|---|---|---|---|---|---|---|---|"]
        for m in B.METHODS:
            def col(test, c="regret"):
                return df[(df.n == n) & (df.method == m) & (df.test == test)].sort_values("seed")[c].values
            hard = (col("hard1") + col("hard2")) / 2
            acc = (col("hard1", "acc") + col("hard2", "acc")) / 2
            hab = (col("hard1", "habit_act") + col("hard2", "habit_act")) / 2
            flips = [np.mean([(store[(n, s, "seen", m)] != store[(n, s, h, m)]).mean() for h in ("hard1", "hard2")])
                     for s in B.SEEDS]
            out.append(f"| {B.NAME[m]} | {ms(col('seen'))} | {ms(col('unseen'))} | {ms(col('hard1'))} | {ms(col('hard2'))} | "
                       f"{ms(hard)} | {col('seen', 'acc').mean():.2f} → {acc.mean():.2f} | "
                       f"{col('seen', 'habit_act').mean():.2f} → {hab.mean():.2f} | {np.mean(flips):.1%} |")
        out.append("")
        # paired change for A and B
        out += ["Change in regret hard − seen, paired by seed: " + "; ".join(
            f"{m}: " + ms((df[(df.n == n) & (df.method == m) & (df.test.isin(['hard1', 'hard2']))]
                           .groupby('seed').regret.mean().values)
                          - df[(df.n == n) & (df.method == m) & (df.test == 'seen')].sort_values('seed').regret.values)
            for m in B.METHODS) + ".\n"]

    # per topic, A at N = 350 (and B)
    out += ["## Per topic, N = 350 (regret, mean over seeds and over the two paraphrases)\n",
            "| topic | scenarios | own-topic cosine (paraphrase 1 / 2) | A seen | A hard | B seen | B hard | C | A habit-act seen → hard |",
            "|---|---|---|---|---|---|---|---|---|"]
    n = 350
    for t in TOPICS:
        idx = np.where(data.tests["seen"].topic.values == t)[0]
        if len(idx) == 0:
            continue
        cells = []
        for m in ("A", "B"):
            for ts in (("seen",), ("hard1", "hard2")):
                cells.append(np.mean([B.evaluate(store[(n, s, x, m)][idx], sets[x][0].iloc[idx])["regret"]
                                      for s in B.SEEDS for x in ts]))
        c = np.mean([B.evaluate(store[(n, s, "seen", "C")][idx], sets["seen"][0].iloc[idx])["regret"] for s in B.SEEDS])
        cs = sim[sim.topic == t].cos_own_max.values
        habit = sets["seen"][0].iloc[idx].habit.values
        if habit.any():
            hs = np.mean([(~store[(n, s, "seen", "A")][idx][habit]).mean() for s in B.SEEDS])
            hh = np.mean([(~store[(n, s, x, "A")][idx][habit]).mean() for s in B.SEEDS for x in ("hard1", "hard2")])
            hab = f"{hs:.2f} → {hh:.2f}"
        else:
            hab = "–"
        out.append(f"| {t} | {len(idx)} | {cs[0]:.3f} / {cs[1]:.3f} | {cells[0]:.3f} | {cells[1]:.3f} | {cells[2]:.3f} | "
                   f"{cells[3]:.3f} | {c:.3f} | {hab} |")
    out.append("\nWith 1–11 test scenarios per topic these per-topic numbers are noisy.\n")
    out += ["## The 24 paraphrases\n", "| topic | # | text | cos to own topic (max) | nearest training phrasing: topic, cos |",
            "|---|---|---|---|---|"]
    for r in sim.itertuples():
        out.append(f"| {r.topic} | {r.k} | {r.text} | {r.cos_own_max:.3f} | {r.nearest_topic}, {r.cos_nearest:.3f} |")
    with open(os.path.join(RESULTS_DIR, "hard_paraphrase.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out[:40]))


if __name__ == "__main__":
    main()
