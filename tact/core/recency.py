"""Recency-weighted replay for the semi-online loop (v3). Every training scenario is weighted 0.5 ** (age in rounds / half_life); age 0 = the round that was added last; warm start, unweighted validation loss.

    with recency.enabled(half_life):      # patches budget.Data.prefix and budget.Ours.fit_reward; the loops of budget.semi_online / v2_cold_start run unchanged
        ...
Outside the context (or half_life None) nothing is changed. Rounds are blocks of PER = 50 scenarios in file order (the first n scenarios of the stream; the current round is (n - 1) // PER).
"""
import contextlib

import numpy as np
import torch

from tact.decision import budget as B

PER = 50
_orig_prefix, _orig_fit_reward = B.Data.prefix, B.Ours.fit_reward
STATE = {"h": None}


def scenario_weights(n, h, per=PER):
    pos = np.arange(n)
    return 0.5 ** (((n - 1) // per - pos // per) / h)


def _prefix(self, n):
    sc, rm, f = _orig_prefix(self, n)
    h = STATE["h"]
    if h is not None:
        rows = self.train_long[self.train_long.id.isin(set(sc.id))]
        w = dict(zip(sc.id.values, scenario_weights(n, h)))
        rm = {**rm, "w": torch.tensor(rows.id.map(w).values, dtype=torch.float32, device=self.dev)}
    return sc, rm, f


def wloss(m, b):
    p = m(b["cls"], b["emotion"], b["day"], b["time"], b.get("location"), b.get("extra"))
    w = b["w"] if "w" in b else torch.ones_like(b["y"])          # validation: unweighted
    return (w * (p - b["y"]) ** 2).sum() / w.sum()


def _fit_reward(self, n, rnd_seed, warm):
    if STATE["h"] is None:
        return _orig_fit_reward(self, n, rnd_seed, warm)
    import time
    d, dev = self.d, self.d.dev
    _, rm_tr, _ = d.prefix(n)
    t0 = time.perf_counter()
    if self.rm is None or not warm:
        self.rm = B.new_scorer(d.enc, dev, rnd_seed, sigmoid=True)
    h = self._hp("reward")
    B.fit(self.rm, wloss, rm_tr, d.rm_val if self.val else None, lr=h["lr"], max_epochs=h["max_epochs"], patience=h["patience"], seed=rnd_seed)
    B._sync(dev)
    self.t["reward"] = time.perf_counter() - t0


@contextlib.contextmanager
def enabled(h):
    old = STATE["h"]
    STATE["h"] = h
    B.Data.prefix, B.Ours.fit_reward = _prefix, _fit_reward
    try:
        yield
    finally:
        STATE["h"] = old
        B.Data.prefix, B.Ours.fit_reward = _orig_prefix, _orig_fit_reward
