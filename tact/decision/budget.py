"""Minimum viable data budget: ask / act decisions, generalisation to an unseen question phrasing.

    python -m tact.core.generate_dataset --n 500 --seed 42 --with-act --outdir data_dpo   # once
    python -m tact.decision.budget                                                            # everything below

Data (data_dpo/, long format): per scenario (topic, emotion, day, time, question phrasing) an 'ask' row and an 'act'
row. ask: score = necessity * receptiveness (+noise); act: score = 1 - necessity (+noise); the act text is the constant
"(no question, act directly)". Training/validation scenarios use phrasings 1 and 2 only. test_seen = held-out scenarios
(phrasings 1/2); test_unseen = the SAME scenarios with the question rewritten to phrasing 3 (never seen in training).

Methods (N = number of training scenarios, taken in file order; each scenario carries both candidates' scores):
  A  Ours        : frozen-BERT reward model r_hat (BertScorer) -> SFT policy -> DPO policy (beta 0.1)
  B  Ours-direct : same r_hat, action = argmax r_hat over {act, ask}
  C  Tabular     : Q[(topic, emotion, day, time, action)], constant lr 0.5; unseen key -> mean of (topic, action) ->
                   global action mean. Sees only the topic, never the phrasing.
  Ct Tabular-text (reference, my addition): same, but keyed by the question string instead of the topic.
Evaluation uses the noise-free score_clean: regret = E[max(ask, act) - value(chosen)], accuracy = agreement with the
oracle (ask iff ask > act). Writes results/budget_*.csv, budget_curves.png, dpo_semionline.png, budget_summary.md.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")   # deterministic cuBLAS (set before torch is used)

import argparse
import copy
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tact.core import plots  # shared plot style / colors
from tact.core.generate_dataset import ACT_TEXT, QUESTIONS
from tact.core.model import ARCH, EXTRA_ORDER, INPUT_COLS, MODELS, RESULTS_DIR, ROOT, V2, FrozenBert, get_device, make_tensors

torch.use_deterministic_algorithms(True, warn_only=True)

DATA = os.path.join(ROOT, "data_dpo")
# Decision rule. "dpo" (default): the SFT + DPO policy A of the original paper. "threshold" (default with TACT_V2=1): ask iff r_hat(ask) - r_hat(act) > 0 (Eq. 2);
# no SFT / DPO is trained, A and B of every script are then the same decision. DECISION=dpo with TACT_V2=1 trains the policy on top of the same reward model.
DECISION = os.environ.get("DECISION", "threshold" if V2 else "dpo")
assert DECISION in ("dpo", "threshold")
MODEL_KW = {}   # extra model arguments, e.g. {"location_dim": 8} / {"n_locations": 3} (extensibility.py)
NS = [25, 50, 75, 100, 150, 250, 350]
SEEDS = [0, 1, 2, 3, 4]
BETA = 0.1
BATCH = 16
# Fixed a priori, not tuned. Reward/SFT: same settings as replay.py. DPO lr is lower (preference optimisation).
HP = {"reward": {"lr": 1e-3, "max_epochs": 100, "patience": 15},
      "sft": {"lr": 1e-3, "max_epochs": 100, "patience": 15},
      "dpo": {"lr": 1e-4, "max_epochs": 100, "patience": 15}}
NOVAL_EPOCHS = {"reward": 100, "sft": 100, "dpo": 50}   # sensitivity run without any validation data
Q_LR = 0.5
METHODS = ["A", "B", "C", "Ct"]
LABEL = {"A": "A  Ours (SFT + DPO)", "B": "B  Ours-direct (argmax r_hat)", "C": "C  Tabular (topic key)",
         "Ct": "Ct Tabular (text key, reference)", "H": "H  Hybrid (A, topic-mean fallback)"}
NAME = {"A": "A  Ours (reward + SFT + DPO)", "B": "B  Ours-direct (argmax r_hat)", "C": "C  Tabular (topic key)",
        "Ct": "Ct Tabular (question-text key, reference)"}


# ------------------------------------------------------------------ data
def load_long(name, data_dir=None):
    """Only model-input columns plus what evaluation needs (necessity / receptiveness are never read)."""
    path = os.path.join(data_dir or DATA, name)
    cols = INPUT_COLS + ["id", "cand", "topic", "phrasing", "score_clean", "habit_hit"]
    header = pd.read_csv(path, nrows=0).columns
    cols = cols + [c for c in EXTRA_ORDER if c in header]   # extra context variables (data_ext, data_ctx*)
    return pd.read_csv(path, usecols=cols)


def scenarios(long_df):
    ask = long_df[long_df.cand == "ask"].reset_index(drop=True)
    act = long_df[long_df.cand == "act"].reset_index(drop=True)
    assert (ask.id.values == act.id.values).all()
    sc = pd.DataFrame({"id": ask.id, "topic": ask.topic, "phrasing": ask.phrasing, "question": ask.question,
                       "emotion": ask.emotion, "day": ask.day, "time": ask.time, "score": 0.0,
                       "s_ask": ask.score, "s_act": act.score, "c_ask": ask.score_clean, "c_act": act.score_clean,
                       "habit": ask.habit_hit.notna()})
    for c in EXTRA_ORDER:
        if c in ask.columns:
            sc[c] = ask[c]
    return sc


def feats(sc, enc, dev):
    """Scenario features: CLS of the question text (ask), CLS of the constant act text, and the context ids."""
    t = make_tensors(sc[["question", "emotion", "day", "time", "score"] + [c for c in EXTRA_ORDER if c in sc]], enc, dev)
    cls_act = enc.encode([ACT_TEXT]).to(dev).expand(len(sc), -1)
    f = {"cls_ask": t["cls"], "cls_act": cls_act, "emotion": t["emotion"], "day": t["day"], "time": t["time"]}
    for k in ("location", "extra"):
        if k in t:
            f[k] = t[k]
    return f


class Data:
    """Everything the methods need, built once per process."""

    def __init__(self, enc, dev, data_dir=None):
        self.enc, self.dev = enc, dev
        self.train_long, self.val_long = load_long("train.csv", data_dir), load_long("val.csv", data_dir)
        self.sc_train, self.sc_val = scenarios(self.train_long), scenarios(self.val_long)
        self.tests = {"seen": scenarios(load_long("test_seen.csv", data_dir)),
                      "unseen": scenarios(load_long("test_unseen.csv", data_dir))}
        assert (self.tests["seen"].id.values == self.tests["unseen"].id.values).all()
        texts = sorted(set(self.train_long.question) | set(self.val_long.question)
                       | {q for t in QUESTIONS.values() for q in t} | {ACT_TEXT})
        enc.encode(texts)   # prime the CLS cache in a fixed order -> identical features regardless of call order
        self.f_val = feats(self.sc_val, enc, dev)
        self.f_test = {k: feats(v, enc, dev) for k, v in self.tests.items()}
        self.loc_cols = [c for c in EXTRA_ORDER if c in self.train_long]
        self.rm_val = make_tensors(self.val_long[INPUT_COLS + self.loc_cols], enc, dev)

    def prefix(self, n):
        """First n training scenarios: (scenario frame, reward-model tensors over their 2n rows, policy features)."""
        sc = self.sc_train.iloc[:n]
        rows = self.train_long[self.train_long.id.isin(set(sc.id))]
        assert len(rows) == 2 * n
        # keep file order (ask, act per scenario)
        return sc, make_tensors(rows[INPUT_COLS + self.loc_cols], self.enc, self.dev), feats(sc, self.enc, self.dev)


# ------------------------------------------------------------------ generic training loop
def fit(model, loss_fn, tr, va, *, lr, max_epochs, patience, seed, train_mode=True):
    """Adam, minibatch 16, early stopping on the validation loss (best weights restored).
    va=None -> no validation data: exactly max_epochs epochs, final weights."""
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    g = torch.Generator().manual_seed(seed)
    n = len(next(iter(tr.values())))
    dev = next(iter(tr.values())).device
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for epoch in range(max_epochs):
        model.train(train_mode)
        perm = torch.randperm(n, generator=g).to(dev)
        for i in range(0, n, BATCH):
            idx = perm[i:i + BATCH]
            loss = loss_fn(model, {k: v[idx] for k, v in tr.items()})
            opt.zero_grad()
            loss.backward()
            opt.step()
        epochs = epoch + 1
        if va is None:
            continue
        model.eval()
        with torch.no_grad():
            v = loss_fn(model, va).item()
        if v < best - 1e-7:
            best, best_state, bad = v, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    if va is not None:
        model.load_state_dict(best_state)
    model.eval()
    return epochs


def reward_loss(m, b):
    return F.mse_loss(m(b["cls"], b["emotion"], b["day"], b["time"], b.get("location"), b.get("extra")), b["y"])


def margin(pol, b):
    """logit(ask) - logit(act): the policy over the candidate set {act, ask} is softmax([z_act, z_ask])."""
    loc, ex = b.get("location"), b.get("extra")
    return (pol(b["cls_ask"], b["emotion"], b["day"], b["time"], loc, ex) - pol(b["cls_act"], b["emotion"], b["day"], b["time"], loc, ex))


def sft_loss(pol, b):
    return F.binary_cross_entropy_with_logits(margin(pol, b), b["y_ask"])


def dpo_loss(pol, b):
    """-log sigmoid(beta * [(log pi(w) - log pi(l)) - (log pi_ref(w) - log pi_ref(l))]); for two candidates
    log pi(w) - log pi(l) = +/- margin, the normaliser cancels."""
    return -F.logsigmoid(BETA * b["sign"] * (margin(pol, b) - b["ref"])).mean()


@torch.no_grad()
def reward_scores(rm, f):
    rm.eval()
    loc, ex = f.get("location"), f.get("extra")
    return (rm(f["cls_ask"], f["emotion"], f["day"], f["time"], loc, ex), rm(f["cls_act"], f["emotion"], f["day"], f["time"], loc, ex))


def new_scorer(enc, dev, seed, sigmoid):
    torch.manual_seed(seed)
    return MODELS[ARCH](enc, sigmoid=sigmoid, **MODEL_KW).to(dev)   # BertScorer (default) or OneHotScorer (ARCH=onehot)


def with_extra(f, **kw):
    return {**f, **kw}


# ------------------------------------------------------------------ method A / B
class Ours:
    """Reward model + policy. fit_round() trains from scratch (warm=False) or continues (warm=True);
    fit_reward() / fit_policy() are its two stages, exposed for the diagnostics in semionline_gap.py."""

    def __init__(self, data, seed, val=True, tag=0):
        self.d, self.seed, self.val, self.tag = data, seed, val, tag
        self.rm = self.pol = None
        self.t = {}

    def _hp(self, k):   # hyper-parameters of stage k; without validation data the epoch count is fixed instead
        return dict(HP[k]) if self.val else dict(HP[k], max_epochs=NOVAL_EPOCHS[k])

    def fit_reward(self, n, rnd_seed, warm):
        d, dev = self.d, self.d.dev
        _, rm_tr, _ = d.prefix(n)
        t0 = time.perf_counter()
        if self.rm is None or not warm:
            self.rm = new_scorer(d.enc, dev, rnd_seed, sigmoid=True)
        h = self._hp("reward")
        fit(self.rm, reward_loss, rm_tr, d.rm_val if self.val else None, lr=h["lr"], max_epochs=h["max_epochs"],
            patience=h["patience"], seed=rnd_seed)
        _sync(dev)
        self.t["reward"] = time.perf_counter() - t0

    def fit_policy(self, n, rnd_seed, mode):
        """mode: 'scratch' = new policy, SFT, then DPO; 'continue' = DPO only, reference = current policy;
        'refresh' = continue SFT from the current policy on all N scenarios, then DPO with that as the reference."""
        d, dev = self.d, self.d.dev
        sc, _, f_tr = d.prefix(n)
        t0 = time.perf_counter()
        ra, rc = reward_scores(self.rm, f_tr)
        va_ra, va_rc = reward_scores(self.rm, d.f_val)
        sign_tr = torch.where(ra > rc, 1.0, -1.0)
        sign_va = torch.where(va_ra > va_rc, 1.0, -1.0)
        if mode == "scratch":
            self.pol = new_scorer(d.enc, dev, rnd_seed + 100000, sigmoid=False)
        if mode in ("scratch", "refresh"):   # SFT: observed best action
            y_tr = (torch.tensor(sc.s_ask.values, device=dev) > torch.tensor(sc.s_act.values, device=dev)).float()
            y_va = (torch.tensor(d.sc_val.s_ask.values, device=dev) > torch.tensor(d.sc_val.s_act.values, device=dev)).float()
            h = self._hp("sft")
            fit(self.pol, sft_loss, with_extra(f_tr, y_ask=y_tr), with_extra(d.f_val, y_ask=y_va) if self.val else None,
                lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed + 1)
        _sync(dev)
        t1 = time.perf_counter()
        # DPO: reference = frozen copy of the current policy (SFT policy, or last round's policy)
        ref = copy.deepcopy(self.pol).eval()
        with torch.no_grad():
            ref_tr, ref_va = margin(ref, f_tr), margin(ref, d.f_val)
        h = self._hp("dpo")
        fit(self.pol, dpo_loss, with_extra(f_tr, sign=sign_tr, ref=ref_tr),
            with_extra(d.f_val, sign=sign_va, ref=ref_va) if self.val else None,
            lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed + 2, train_mode=False)
        _sync(dev)
        self.t["sft"], self.t["dpo"] = t1 - t0, time.perf_counter() - t1

    def fit_round(self, n, rnd_seed, warm):
        self.fit_reward(n, rnd_seed, warm)
        if DECISION == "dpo":
            self.fit_policy(n, rnd_seed, "scratch" if (self.pol is None or not warm) else "continue")
        else:
            self.t["sft"] = self.t["dpo"] = 0.0

    @torch.no_grad()
    def decide_policy(self, f):
        """Decisions of the SFT + DPO policy (needs fit_policy); True = ask."""
        self.pol.eval()
        return (margin(self.pol, f) > 0).cpu().numpy()

    @torch.no_grad()
    def decide(self, f):
        """(A decisions, B decisions): True = ask. B = argmax of the reward model (the threshold rule); A = the DPO policy, or the same as B with DECISION=threshold."""
        ra, rc = reward_scores(self.rm, f)
        b = (ra > rc).cpu().numpy()
        if DECISION == "dpo":
            return self.decide_policy(f), b
        return b, b.copy()


def _sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


# ------------------------------------------------------------------ method C
class Tabular:
    """Q[(key..., action)] with constant learning rate; first visit initialises Q to the observed reward."""

    def __init__(self, key_col, ctx_cols=("emotion", "day", "time")):
        self.key_col, self.ctx_cols, self.Q = key_col, tuple(ctx_cols), {}
        self.sum = {}
        self.cnt = {}
        self.gsum, self.gcnt = {"ask": 0.0, "act": 0.0}, {"ask": 0, "act": 0}

    def update(self, sc, order):
        for i in order:
            r = sc.iloc[i]
            for a, val in (("ask", r.s_ask), ("act", r.s_act)):
                k = (r[self.key_col], *[r[c] for c in self.ctx_cols], a)
                self.Q[k] = val if k not in self.Q else self.Q[k] + Q_LR * (val - self.Q[k])
                self.sum[(r[self.key_col], a)] = self.sum.get((r[self.key_col], a), 0.0) + val
                self.cnt[(r[self.key_col], a)] = self.cnt.get((r[self.key_col], a), 0) + 1
                self.gsum[a] += val
                self.gcnt[a] += 1

    def decide(self, sc):
        """Returns (ask decisions, fraction resolved at level 1 / 2 / 3 of the fallback chain)."""
        out, lv = [], np.zeros(3)
        for r in sc.itertuples():
            vals = {}
            lvl = 0
            for a in ("ask", "act"):
                k = (getattr(r, self.key_col), *[getattr(r, c) for c in self.ctx_cols], a)
                if k in self.Q:
                    vals[a], l = self.Q[k], 0
                elif (getattr(r, self.key_col), a) in self.cnt:
                    kk = (getattr(r, self.key_col), a)
                    vals[a], l = self.sum[kk] / self.cnt[kk], 1
                else:
                    vals[a], l = self.gsum[a] / max(self.gcnt[a], 1), 2
                lvl = max(lvl, l)
            lv[lvl] += 1
            out.append(vals["ask"] > vals["act"])
        return np.array(out), lv / len(sc)


# ------------------------------------------------------------------ evaluation
def evaluate(ask, sc):
    """ask: bool array of decisions (True = ask). All quantities use the noise-free clean values."""
    c_ask, c_act = sc.c_ask.values, sc.c_act.values
    oracle_ask = c_ask >= c_act
    chosen = np.where(ask, c_ask, c_act)
    habit = sc.habit.values
    return {"acc": float((ask == oracle_ask).mean()), "regret": float((np.maximum(c_ask, c_act) - chosen).mean()),
            "habit_act": float((~ask[habit]).mean()) if habit.any() else float("nan"), "act_rate": float((~ask).mean())}


def baselines(sc):
    n = len(sc)
    return {"always_ask": evaluate(np.ones(n, bool), sc), "never_ask": evaluate(np.zeros(n, bool), sc),
            "oracle": evaluate(sc.c_ask.values >= sc.c_act.values, sc)}


# ------------------------------------------------------------------ experiments
def sweep(data, val=True, keep=(100, 350)):
    rows, store = [], {}
    for n in NS:
        for seed in SEEDS:
            s = seed * 1000 + n
            ours = Ours(data, seed, val=val)
            ours.fit_round(n, s, warm=False)
            sc = data.sc_train.iloc[:n]
            order = np.random.default_rng([seed, n]).permutation(n)
            tabs = {"C": Tabular("topic"), "Ct": Tabular("question")}
            for t in tabs.values():
                t.update(sc, order)
            for ts in ("seen", "unseen"):
                tsc, f = data.tests[ts], data.f_test[ts]
                a, b = ours.decide(f)
                dec = {"A": (a, {}), "B": (b, {})}
                for name, t in tabs.items():
                    d, lv = t.decide(tsc)
                    dec[name] = (d, {"lvl_exact": lv[0], "lvl_topic": lv[1], "lvl_global": lv[2]})
                for m, (d, extra) in dec.items():
                    rows.append({"n": n, "seed": seed, "method": m, "test": ts, **evaluate(d, tsc), **extra})
                rows.append({"n": n, "seed": seed, "method": "A_vs_B", "test": ts, "acc": float((a == b).mean()),
                             "regret": np.nan, "habit_act": np.nan, "act_rate": np.nan})
                if n in keep:
                    store[(n, seed, ts)] = {m: d for m, (d, _) in dec.items()}
            print(f"[sweep{'' if val else ' no-val'}] N={n:3d} seed={seed} regret seen/unseen: "
                  + " ".join(f"{m}={[round(r['regret'], 3) for r in rows[-10:] if r['method'] == m]}" for m in ("A", "B", "C")),
                  flush=True)
    return pd.DataFrame(rows), store


def semi_online(data, seed, timing=False, return_ours=False):
    """7 rounds x 50 scenarios: A warm-starts the reward model and continues DPO; C updates its Q table."""
    ours, tab = Ours(data, seed), Tabular("topic")
    rows, times = [], []
    for rnd in range(1, 8):
        n = 50 * rnd
        s = seed * 1000 + 700 + rnd
        ours.fit_round(n, s, warm=rnd > 1)
        batch = data.sc_train.iloc[n - 50:n]
        t0 = time.perf_counter()
        tab.update(batch.reset_index(drop=True), np.random.default_rng([seed, rnd, 7]).permutation(50))
        tq = time.perf_counter() - t0
        times.append({"round": rnd, **{f"t_{k}": v for k, v in ours.t.items()}, "t_q": tq})
        for ts in ("seen", "unseen"):
            a, b = ours.decide(data.f_test[ts])
            c, _ = tab.decide(data.tests[ts])
            for m, d in (("A", a), ("B", b), ("C", c)):
                rows.append({"round": rnd, "n": n, "seed": seed, "method": m, "test": ts, **evaluate(d, data.tests[ts])})
    if return_ours:
        return pd.DataFrame(rows), pd.DataFrame(times), ours
    return pd.DataFrame(rows), pd.DataFrame(times)


# ------------------------------------------------------------------ cost evidence (C3)
def count(m, trainable=True):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


def cost_report():
    """Trainable parameters and wall-clock per update on CPU (the device asked for in this experiment)."""
    from transformers import AutoModel
    cpu = torch.device("cpu")
    data = Data(FrozenBert(cpu), cpu)
    _, times = semi_online(data, seed=0, timing=True)
    upd = times[times["round"] >= 2]
    rm = MODELS[ARCH](None)
    bert = AutoModel.from_pretrained("bert-base-uncased")
    n_bert = count(bert)
    # BERT forward for 50 new question texts
    tok = data.enc.tokenizer(data.sc_train.question.iloc[:50].tolist(), padding=True, return_tensors="pt")
    t0 = time.perf_counter()
    with torch.no_grad():
        data.enc.bert(**tok)
    t_bert = time.perf_counter() - t0
    # comparator: one epoch of full BERT fine-tuning (fwd + bwd + AdamW) on the 350 training questions
    bert.train()
    head = torch.nn.Linear(768, 1)
    opt = torch.optim.AdamW(list(bert.parameters()) + list(head.parameters()), lr=2e-5)
    qs = data.sc_train.question.tolist()
    t0 = time.perf_counter()
    for i in range(0, len(qs), BATCH):
        tk = data.enc.tokenizer(qs[i:i + BATCH], padding=True, return_tensors="pt")
        loss = head(bert(**tk).last_hidden_state[:, 0]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    t_ft = time.perf_counter() - t0
    return {"params_reward": count(rm), "params_policy": count(rm), "params_bert": n_bert,
            "t_reward": (upd.t_reward.mean(), upd.t_reward.std()), "t_dpo": (upd.t_dpo.mean(), upd.t_dpo.std()),
            "t_q": (upd.t_q.mean(), upd.t_q.std()), "t_bert50": t_bert, "t_ft_epoch": t_ft,
            "cpu_threads": torch.get_num_threads()}


# ------------------------------------------------------------------ reporting
def ms(x, nd=3):
    return f"{np.mean(x):.{nd}f} ± {np.std(x, ddof=1):.{nd}f}"


def first_n(df, method, ts, base, rule="mean"):
    """Smallest N meeting a regret criterion. rules: mean <= base; mean+std <= base; every seed <= base;
    half = mean <= base / 2 (a stricter bar than the one asked for, added because a constant policy already
    meets the plain threshold)."""
    for n in NS:
        r = df[(df.method == method) & (df.test == ts) & (df.n == n)].regret
        v, b = {"mean": (r.mean(), base), "sd": (r.mean() + r.std(ddof=1), base), "max": (r.max(), base),
                "half": (r.mean(), base / 2)}[rule]
        if v <= b:
            return n
    return None


def curves_figure(df, bl, fname, xcol, xs, title, methods, xlabel, logx):
    COL = {"A": "#2a78d6", "B": "#eb6834", "C": "#1baf7a", "Ct": "#6250d6", "H": "#d55181"}
    LS = {"A": "-", "B": "-", "C": "-", "Ct": ":", "H": "--"}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3), sharey=True)
    for ax, ts in zip(axes, ("seen", "unseen")):
        for m in methods:
            g = df[(df.method == m) & (df.test == ts)].groupby(xcol).regret.agg(["mean", "std"])
            ax.plot(g.index, g["mean"], color=COL[m], ls=LS[m], lw=2 if m != "Ct" else 1.6, marker="o", ms=5,
                    label=LABEL[m])
            if m != "Ct":
                ax.fill_between(g.index, g["mean"] - g["std"], g["mean"] + g["std"], color=COL[m], alpha=0.14, lw=0)
        b = bl[ts]
        ax.axhline(b["always_ask"]["regret"], color=plots.INK2, ls="--", lw=1.2, label="always-ask")
        ax.axhline(b["never_ask"]["regret"], color=plots.INK2, ls="-.", lw=1.2, label="never-ask")
        ax.axhline(0, color=plots.INK, lw=1.0, label="oracle (0)")
        if logx:
            ax.set_xscale("log")
            ax.minorticks_off()
        ax.set_xticks(xs, [str(x) for x in xs])
        ax.set_xlabel(xlabel)
        ax.set_title("test_seen: held-out scenarios, phrasings 1/2" if ts == "seen"
                     else "test_unseen: same scenarios, unseen phrasing 3", fontsize=10, loc="left", color=plots.INK)
    axes[0].set_ylabel("mean regret (clean reward; lower is better)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=8.5)
    fig.suptitle(title + (" [one-hot context encoding]" if ARCH == "onehot" else ""), x=0.01, ha="left", fontsize=11,
                 color=plots.INK2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(RESULTS_DIR, fname), dpi=170)
    plt.close(fig)


def per_topic_table(store, data, ns=(100, 350)):
    """Regret by topic (the unseen phrasing is the same for every row, so topic is the natural grouping) for A and
    B, plus how close the unseen phrasing's frozen-BERT CLS is to the training phrasings of its topic."""
    enc = data.enc
    out = []
    for n in ns:
        out += [f"\n### N = {n}: regret by topic, mean over 5 seeds (A / B / C)\n",
                "| topic | habit rule | seen A / B / C | unseen A / B / C | cos(phrasing 3, nearest train phrasing) | cos(phrasing 1, 2) |",
                "|---|---|---|---|---|---|"]
        for topic in QUESTIONS:
            idx = np.where(data.tests["seen"].topic.values == topic)[0]
            if len(idx) == 0:
                continue
            cells = {}
            for ts in ("seen", "unseen"):
                sc = data.tests[ts].iloc[idx]
                vals = []
                for m in ("A", "B", "C"):
                    vals.append(np.mean([evaluate(store[(n, s, ts)][m][idx], sc)["regret"] for s in SEEDS]))
                cells[ts] = " / ".join(f"{v:.3f}" for v in vals)
            v = [F.normalize(enc.encode([q]), dim=1)[0] for q in QUESTIONS[topic]]
            cos13, cos23, cos12 = float(v[0] @ v[2]), float(v[1] @ v[2]), float(v[0] @ v[1])
            habit = "yes" if data.tests["seen"].iloc[idx].habit.any() else ""
            out.append(f"| {topic} (n={len(idx)}) | {habit} | {cells['seen']} | {cells['unseen']} | "
                       f"{max(cos13, cos23):.3f} | {cos12:.3f} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cost", action="store_true", help="skip the CPU wall-clock measurements")
    args = ap.parse_args()
    dev = get_device()
    print("device:", dev, flush=True)
    data = Data(FrozenBert(dev), dev)
    bl = {ts: baselines(sc) for ts, sc in data.tests.items()}

    df, store = sweep(data)
    df.to_csv(os.path.join(RESULTS_DIR, "budget_runs.csv"), index=False)
    dfn, _ = sweep(data, val=False, keep=())
    dfn.to_csv(os.path.join(RESULTS_DIR, "budget_runs_noval.csv"), index=False)

    so, sot = [], []
    for seed in SEEDS:
        a, t = semi_online(data, seed)
        so.append(a)
        sot.append(t.assign(seed=seed))
        print(f"[semi-online] seed={seed} done", flush=True)
    so = pd.concat(so)
    so.to_csv(os.path.join(RESULTS_DIR, "dpo_semionline.csv"), index=False)

    curves_figure(df, bl, "budget_curves.png", "n", NS,
                  "Regret vs. number of training scenarios (mean ± std over 5 seeds; N counts scenarios)",
                  ["A", "B", "C", "Ct"], "training scenarios N", True)
    curves_figure(so, bl, "dpo_semionline.png", "n", [50 * r for r in range(1, 8)],
                  "Semi-online loop: 7 rounds x 50 scenarios (A: warm reward update + continued DPO; C: Q-table update)",
                  ["A", "C"], "scenarios seen so far", False)

    cost = None if args.skip_cost else cost_report()
    write_summary(df, dfn, so, bl, store, data, cost)
    print("done")


def write_summary(df, dfn, so, bl, store, data, cost):
    base = {ts: min(b["always_ask"]["regret"], b["never_ask"]["regret"]) for ts, b in bl.items()}
    out = ["# Minimum viable data budget (`budget.py`)" + (" — ONE-HOT context encoding (OneHotScorer as reward model and policy)" if ARCH == "onehot" else "") + "\n",
           "Every number is mean ± std over 5 seeds unless stated otherwise. N counts training *scenarios*; each scenario "
           "carries the scores of both candidates (ask and act), so a method sees 2N labelled (candidate, context) rows. "
           "Regret and accuracy are measured against the noise-free `score_clean` of the 75 test scenarios "
           "(test_unseen = the same 75 scenarios with the question rewritten to the never-trained phrasing 3). "
           "With only 75 test scenarios, the baselines and all curves are noisy.\n",
           "## Baselines on the test scenarios\n", "| | always-ask | never-ask | oracle |", "|---|---|---|---|"]
    b = bl["seen"]
    out.append("| regret | " + " | ".join(f"{b[k]['regret']:.3f}" for k in ("always_ask", "never_ask", "oracle")) + " |")
    out.append("| accuracy | " + " | ".join(f"{b[k]['acc']:.3f}" for k in ("always_ask", "never_ask", "oracle")) + " |")
    out.append("| act rate on habit scenarios | " + " | ".join(f"{b[k]['habit_act']:.2f}" for k in ("always_ask", "never_ask", "oracle")) + " |")
    out.append(f"\nSeen and unseen test sets hold identical scenarios, so the baselines are identical. The oracle chooses act in "
               f"{b['oracle']['act_rate']:.0%} of the test scenarios. Threshold for the budget question: "
               f"min(always-ask, never-ask) = **{base['seen']:.3f}** regret.\n")

    out.append("## Smallest N reaching regret ≤ min(always-ask, never-ask)\n")
    out.append("First N in {25, 50, 75, 100, 150, 250, 350} meeting each criterion; “–” = not reached up to N = 350. "
               "b = min(always-ask, never-ask) regret. The criterion asked for is “mean ≤ b”. It is a low bar: a policy "
               "that is almost constant (always-ask or never-ask) already meets it, so three stricter readings are added: "
               "mean + 1 std ≤ b, every one of the 5 seeds ≤ b, and mean ≤ b/2 (halving the baseline regret; my addition).\n")
    rules = [("mean", "mean ≤ b"), ("sd", "mean+std ≤ b"), ("max", "all seeds ≤ b"), ("half", "mean ≤ b/2")]
    out += ["| method | " + " | ".join(f"seen: {t}" for _, t in rules) + " | " + " | ".join(f"unseen: {t}" for _, t in rules) + " |",
            "|---|" + "---|" * (2 * len(rules))]

    def fmt(n):
        return "–" if n is None else str(n)

    for m in METHODS:
        cells = [fmt(first_n(df, m, ts, base[ts], r)) for ts in ("seen", "unseen") for r, _ in rules]
        out.append(f"| {NAME[m]} | " + " | ".join(cells) + " |")
    out.append("\nSensitivity without any validation data (reward model, SFT and DPO trained for a fixed 100 / 100 / 50 epochs, "
               "final weights, no early stopping; the main runs early-stop on the 75 validation scenarios, which is extra "
               "labelled data beyond N):\n")
    out += ["| method | " + " | ".join(f"seen: {t}" for _, t in rules) + " | " + " | ".join(f"unseen: {t}" for _, t in rules) + " |",
            "|---|" + "---|" * (2 * len(rules))]
    for m in ("A", "B"):
        cells = [fmt(first_n(dfn, m, ts, base[ts], r)) for ts in ("seen", "unseen") for r, _ in rules]
        out.append(f"| {NAME[m]}, no validation data | " + " | ".join(cells) + " |")

    for ts in ("seen", "unseen"):
        out.append(f"\n## test_{ts}: regret / accuracy / act-rate on habit scenarios by N\n")
        out.append("| N | " + " | ".join(f"{m} regret" for m in METHODS) + " | " + " | ".join(f"{m} acc" for m in METHODS)
                   + " | " + " | ".join(f"{m} habit-act" for m in METHODS) + " |")
        out.append("|" + "---|" * (1 + 3 * len(METHODS)))
        for n in NS:
            cells = []
            for col, nd in (("regret", 3), ("acc", 2), ("habit_act", 2)):
                for m in METHODS:
                    cells.append(ms(df[(df.method == m) & (df.test == ts) & (df.n == n)][col], nd))
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out.append(f"\nBaselines: always-ask regret {bl[ts]['always_ask']['regret']:.3f}, never-ask {bl[ts]['never_ask']['regret']:.3f}; "
                   f"habit-scenario act rate: always-ask 0, never-ask 1, oracle 1.")

    out += ["\n## Seen → unseen phrasing: change in regret (unseen − seen), paired by seed\n",
            "| N | A | B | C | Ct |", "|---|---|---|---|---|"]
    for n in NS:
        cells = []
        for m in METHODS:
            s = df[(df.method == m) & (df.test == "seen") & (df.n == n)].sort_values("seed").regret.values
            u = df[(df.method == m) & (df.test == "unseen") & (df.n == n)].sort_values("seed").regret.values
            cells.append(ms(u - s))
        out.append(f"| {n} | " + " | ".join(cells) + " |")
    out.append("\nC cannot change between the two test sets by construction: its key contains the topic, not the phrasing, "
               "and in this generator the reward does not depend on the phrasing. Ct is keyed by the question string, so an "
               "unseen phrasing falls through to the global action mean.")

    out += ["\n## Does the DPO policy keep the reward model's decisions? (A vs B)\n",
            "| N | agreement A–B, seen | agreement A–B, unseen | regret A − B, seen | regret A − B, unseen |", "|---|---|---|---|---|"]
    for n in NS:
        ag = [ms(df[(df.method == "A_vs_B") & (df.test == ts) & (df.n == n)].acc, 3) for ts in ("seen", "unseen")]
        dr = []
        for ts in ("seen", "unseen"):
            a = df[(df.method == "A") & (df.test == ts) & (df.n == n)].sort_values("seed").regret.values
            bb = df[(df.method == "B") & (df.test == ts) & (df.n == n)].sort_values("seed").regret.values
            dr.append(ms(a - bb))
        out.append(f"| {n} | {ag[0]} | {ag[1]} | {dr[0]} | {dr[1]} |")

    out += ["\n## Fallback levels of the tabular agent C (share of test scenarios resolved at each level)\n",
            "Level 1 = exact (topic, emotion, day, time) key seen for both actions; level 2 = (topic, action) mean; "
            "level 3 = global action mean.\n", "| N | exact key | topic mean | global mean |", "|---|---|---|---|"]
    for n in NS:
        g = df[(df.method == "C") & (df.test == "seen") & (df.n == n)]
        out.append(f"| {n} | {g.lvl_exact.mean():.2f} | {g.lvl_topic.mean():.2f} | {g.lvl_global.mean():.2f} |")

    out += ["\n## Error analysis by topic (A, B, C)",
            "\nThe unseen phrasing is phrasing 3 for every scenario, so the natural grouping is by topic. The test set has only "
            "1–11 scenarios per topic, so per-topic regrets are very noisy. Cosines are between frozen-BERT CLS vectors: the "
            "unseen phrasing is about as close to a training phrasing of its topic as the two training phrasings are to each "
            "other, i.e. the shift is small (all phrasings of a topic share its keyword)."] + per_topic_table(store, data)

    so_ = so.groupby(["method", "test", "round"]).regret.agg(["mean", "std"]).reset_index()
    out += ["\n## Semi-online loop (7 rounds x 50 scenarios): regret by round\n",
            "| round | n | A seen | C seen | A unseen | C unseen |", "|---|---|---|---|---|---|"]
    for r in range(1, 8):
        cells = []
        for ts in ("seen", "unseen"):
            for m in ("A", "C"):
                x = so[(so.method == m) & (so.test == ts) & (so["round"] == r)].regret
                cells.append(ms(x))
        out.append(f"| {r} | {50 * r} | {cells[0]} | {cells[1]} | {cells[2]} | {cells[3]} |")

    if cost:
        out += ["\n## Training-volume evidence (C3)\n",
                "| | value |", "|---|---|",
                f"| trainable parameters, reward model ({MODELS[ARCH].__name__}) | {cost['params_reward']:,} |",
                f"| trainable parameters, policy (same architecture, no sigmoid) | {cost['params_policy']:,} |",
                f"| Ours total trainable (reward + policy) | {cost['params_reward'] + cost['params_policy']:,} "
                f"({(cost['params_reward'] + cost['params_policy']) / cost['params_bert']:.2%} of BERT) |",
                f"| BERT-base, full fine-tuning | {cost['params_bert']:,} |",
                f"| trainable ratio, reward model only | {cost['params_reward'] / cost['params_bert']:.3%} |",
                f"| per-round update, reward model (warm start, early stopping, ≤350 scenarios), CPU | "
                f"{cost['t_reward'][0]:.2f} ± {cost['t_reward'][1]:.2f} s |",
                f"| per-round update, DPO policy (continued), CPU | {cost['t_dpo'][0]:.2f} ± {cost['t_dpo'][1]:.2f} s |",
                f"| per-round update, Q-table (50 scenarios), CPU | {cost['t_q'][0] * 1000:.1f} ± {cost['t_q'][1] * 1000:.1f} ms |",
                f"| frozen-BERT forward for 50 new question texts, CPU | {cost['t_bert50']:.2f} s |",
                f"| reference: ONE epoch of full BERT fine-tuning on 350 questions (fwd+bwd+AdamW, batch 16), CPU | "
                f"{cost['t_ft_epoch']:.1f} s |",
                f"\nCPU timings from the semi-online loop (seed 0, rounds 2–7, mean ± std over rounds), torch threads = "
                f"{cost['cpu_threads']}, the model features are cached CLS vectors (BERT forward is counted separately). "
                "The reward-model and DPO times include early stopping, so they vary with the number of epochs run. The full "
                "fine-tuning row is a single epoch on questions only (no context, no policy), a lower bound on what BERT "
                "fine-tuning would cost. The parameter saving is large (about 0.2% of BERT per model), but the wall-clock "
                "saving on this tiny text task is not: one full-BERT epoch takes about as long as one of our rounds, so the "
                "advantage in time depends on how many epochs full fine-tuning would need, which I did not measure."]
    out += ["\n## Design notes and assumptions\n",
            "- The policy design (“candidate set {act, question}, DPO β = 0.1”) was not available to me as code; I implemented it as "
            f"follows. The policy is the {MODELS[ARCH].__name__} architecture without the final sigmoid, scoring each candidate "
            "(question text, emotion, day, time) with a logit; π is the softmax over the two logits. SFT: cross-entropy towards the "
            "action with the higher *observed* score. DPO: preference pairs labelled by the reward model r_hat "
            "(preferred = argmax r_hat), reference policy = the SFT policy (semi-online: the previous round's policy), β = 0.1, "
            "dropout off. Learning rates, epochs and patience are fixed a priori (see `HP` in budget.py) and were not tuned.",
            "- The act candidate's text is the constant string from the spec, so the reward model's act score cannot depend on the topic; "
            "the topic enters decisions only through the ask text. In habit scenarios the ask score is low, which is enough to make "
            "act win; only near-ties (|ask − act| small) can be affected, which costs little regret.",
            "- Tabular C is updated with both candidates' rewards of each scenario (same information as A), constant lr 0.5, first "
            "visit initialises Q to the observed reward. Its updates are applied in a seed-dependent shuffled order within the first "
            "N scenarios (otherwise it would be fully deterministic).",
            "- Early stopping uses the 75 validation scenarios (150 rows) for A and B in the main runs; this is data beyond N. "
            "The no-validation sensitivity above removes it.",
            "- Devices: sweeps ran on " + str(data.dev) + " with deterministic algorithms; timings in the cost section are CPU."]
    if DECISION == "threshold":
        out.insert(1, "> **v2 (final architecture, threshold rule).** The decision of A and B is the same threshold rule on the reward model (ask iff r_hat(ask) - r_hat(act) > 0); "
                      "no SFT/DPO policy is trained, so the A and B columns are identical and the DPO sentences in the design notes below are legacy text of the original script.\n")
    with open(os.path.join(RESULTS_DIR, "budget_summary.md"), "w") as f:
        f.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
