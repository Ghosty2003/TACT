"""v3, step 2a: Sec. 5.4 semi-online stability with the recency-weighted default (results/v3/semionline_*.csv).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.reward.v3_semionline

Two protocols of the paper, both re-run with weights 0.5 ** (age in rounds / h) on the CHOSEN half-life h (results/v3/chosen_half_life.json), warm start vs from scratch (BOTH weighted), reporting seeds 0-4:
 (A) replay.py protocol (ask-only train.csv, 7 rounds x 50, test MSE / Spearman / pairwise accuracy on test.csv, mean absolute prediction change between consecutive rounds on test.csv): `replay_*` rows.
     The unweighted rerun of the same code is stored too (`unweighted`) and must reproduce results/v2/replay_metrics.csv (integrity check in the report).
 (B) budget.semi_online protocol (data_dpo, ask + act rows, 7 rounds x 50, threshold rule): regret / accuracy / ask rate on the 75 test scenarios after every round for the weighted warm start (budget.semi_online under
     recency.enabled), the weighted from-scratch retraining, and (reference) unweighted from-scratch retraining with the same seeds.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import copy
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tact.decision import budget as B
from tact.core import plots
from tact.core import recency
from tact.core.evaluate import evaluate
from tact.core.model import MODELS, ROOT, FrozenBert, get_device, load_csv, make_tensors
from tact.reward.replay import HPARAMS as HP
from tact.reward.replay import run_seed

OUT = os.path.join(ROOT, "results", "v3")
PER = 50


def train_w(model, tr, va, gen_seed, w=None):
    """replay.train with optional sample weights in the training loss (validation loss unweighted)."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=HP["lr"])
    g = torch.Generator().manual_seed(gen_seed)
    n = len(tr["y"])
    best, best_state, bad = float("inf"), None, 0
    for epoch in range(HP["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g).to(tr["y"].device)
        for i in range(0, n, HP["batch_size"]):
            idx = perm[i:i + HP["batch_size"]]
            pred = model(tr["cls"][idx], tr["emotion"][idx], tr["day"][idx], tr["time"][idx])
            if w is None:
                loss = F.mse_loss(pred, tr["y"][idx])
            else:
                loss = (w[idx] * (pred - tr["y"][idx]) ** 2).sum() / w[idx].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(va["cls"], va["emotion"], va["day"], va["time"]), va["y"]).item()
        if val < best - 1e-7:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= HP["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()


def replay_protocol(h, enc, dev):
    train_df, val_df, test_df = load_csv("train.csv"), load_csv("val.csv"), load_csv("test.csv")
    va = make_tensors(val_df, enc, dev)
    rows = []
    os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
    for weighting in ("unweighted", "recency"):
        for strategy in ("scratch", "warm"):
            for seed in HP["seeds"]:
                prev_pred, prev_model = None, None
                for rnd in range(1, HP["n_batches"] + 1):
                    n_train = rnd * PER
                    tr = make_tensors(train_df.iloc[:n_train], enc, dev)
                    w = None if weighting == "unweighted" else torch.tensor(0.5 ** ((rnd - 1 - np.arange(n_train) // PER) / h), dtype=torch.float32, device=dev)
                    s = run_seed(seed, rnd)
                    torch.manual_seed(s)
                    np.random.seed(s)
                    model = MODELS["bert"](enc).to(dev)
                    if strategy == "warm" and prev_model is not None:
                        model.load_state_dict(prev_model.state_dict())
                    train_w(model, tr, va, s, w)
                    m = evaluate(model, test_df)
                    pred = model.predict(test_df)
                    drift = float(np.mean(np.abs(pred - prev_pred))) if prev_pred is not None else float("nan")
                    row = {"weighting": weighting, "strategy": strategy, "seed": seed, "round": rnd, "n_train": n_train, **m, "drift": drift}
                    if rnd == HP["n_batches"]:
                        st = plots.heatmap_stats(plots.pred_grid(model))
                        row["sat_over_other"] = st["sat_over_other"]
                        if weighting == "recency":
                            torch.save(model.state_dict(), os.path.join(OUT, "models", f"{strategy}_seed{seed}_final.pt"))
                    rows.append(row)
                    prev_pred, prev_model = pred, model
                print(f"[replay {weighting} {strategy} seed={seed}] round-7 mse {rows[-1]['mse']:.4f} drift {rows[-1]['drift']:.4f}", flush=True)
    return pd.DataFrame(rows)


def util(acc, act_rate):
    return acc / (1.0 + (1.0 - act_rate))


def budget_protocol(h, enc, dev):
    data = B.Data(enc, dev)
    rows = []
    for seed in B.SEEDS:
        with recency.enabled(h):
            df, _ = B.semi_online(data, seed)
        for r in df.itertuples():
            rows.append({"variant": "warm_recency" if r.method == "A" else ("tabular" if r.method == "C" else "B"), "seed": seed, "round": r.round, "test": r.test, "acc": r.acc, "regret": r.regret, "act_rate": r.act_rate})
        df0, _ = B.semi_online(data, seed)          # unweighted reference: must reproduce results/v2/dpo_semionline.csv
        for r in df0.itertuples():
            if r.method == "A":
                rows.append({"variant": "warm_unweighted", "seed": seed, "round": r.round, "test": r.test, "acc": r.acc, "regret": r.regret, "act_rate": r.act_rate})
        for variant, hh in (("scratch_recency", h), ("scratch_unweighted", None)):
            for rnd in range(1, 8):
                ours = B.Ours(data, seed)
                with recency.enabled(hh):
                    ours.fit_reward(50 * rnd, seed * 1000 + 700 + rnd, warm=False)
                for ts in ("seen", "unseen"):
                    a, _ = ours.decide(data.f_test[ts])
                    ev = B.evaluate(a, data.tests[ts])
                    rows.append({"variant": variant, "seed": seed, "round": rnd, "test": ts, "acc": ev["acc"], "regret": ev["regret"], "act_rate": ev["act_rate"]})
        print(f"[budget protocol] seed {seed} done", flush=True)
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    h = json.load(open(os.path.join(OUT, "chosen_half_life.json")))["half_life"]
    assert h is not None
    dev = get_device()
    enc = FrozenBert(dev)
    B.Data(enc, dev, data_dir=os.path.join(ROOT, "data_dpo"))       # cache priming in the fixed order of the other experiments
    replay_protocol(h, enc, dev).to_csv(os.path.join(OUT, "semionline_replay.csv"), index=False)
    budget_protocol(h, enc, dev).to_csv(os.path.join(OUT, "semionline_budget.csv"), index=False)
    print("done")


if __name__ == "__main__":
    main()
