"""Incremental replay experiment: from-scratch vs warm-start retraining of the reward model.

    python -m tact.reward.replay                 # full experiment -> results/replay_metrics.csv
    python -m tact.reward.replay --model onehot  # same experiment with the one-hot baseline

train.csv is cut (in file order) into 7 batches of 50; round r trains on batches 1..r.
Every round is evaluated on the fixed test.csv. Stability metrics compare round r with round r-1.
Only question/emotion/day/time/score are ever read (see model.INPUT_COLS).
"""
import argparse
import copy
import json
import os
import time as _time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import kendalltau

from tact.core.evaluate import diagnostics, evaluate
from tact.core.model import INPUT_COLS, MODELS, RESULTS_DIR, FrozenBert, get_device, load_csv, load_eval_csv, make_tensors

# ---- hyper-parameters (fixed; dumped to results/hparams.json) ----------------------------------
HPARAMS = {
    "lr": 1e-3, "batch_size": 16, "max_epochs": 100, "optimizer": "Adam",
    "early_stop_patience": 15,   # epochs without val-MSE improvement; best-val weights are restored
    "n_batches": 7, "batch_size_data": 50, "seeds": [0, 1, 2, 3, 4],
    "strategies": ["scratch", "warm"],
}
COLUMNS = ["strategy", "seed", "round", "n_train", "mse", "spearman", "pairwise_acc",
           "drift", "first_batch_mse", "kendall_tau"]


def run_seed(seed, rnd):
    """Deterministic per-(seed, round) seed, shared by both strategies (so round 1 is identical)."""
    return seed * 1000 + rnd


def train(model, tr, va, hp, gen_seed):
    """Adam + MSE, early stopping on val MSE; restores the best-val weights. Returns epochs run."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=hp["lr"])
    g = torch.Generator().manual_seed(gen_seed)
    n = len(tr["y"])
    best, best_state, bad, epochs = float("inf"), None, 0, 0
    for epoch in range(hp["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=g).to(tr["y"].device)
        for i in range(0, n, hp["batch_size"]):
            idx = perm[i:i + hp["batch_size"]]
            pred = model(tr["cls"][idx], tr["emotion"][idx], tr["day"][idx], tr["time"][idx])
            loss = F.mse_loss(pred, tr["y"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(va["cls"], va["emotion"], va["day"], va["time"]), va["y"]).item()
        epochs = epoch + 1
        if val < best - 1e-7:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["early_stop_patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return epochs, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), default="bert")
    ap.add_argument("--label", choices=["score", "score_clean"], default="score",
                    help="training/val/test label column. score_clean is ONLY for the drift-cause experiment "
                         "(results/drift_cleanlabel.md); outputs get a _cleanlabel suffix")
    ap.add_argument("--out", default=None, help="metrics csv (default results/replay_metrics.csv; "
                                               "results/replay_metrics_<model>.csv for non-bert)")
    args = ap.parse_args()
    hp = HPARAMS
    suffix = ("" if args.model == "bert" else f"_{args.model}") + ("" if args.label == "score" else "_cleanlabel")
    diag_csv = os.path.join(RESULTS_DIR, f"replay_diagnostics{suffix}.csv")
    out_csv = args.out or os.path.join(RESULTS_DIR, f"replay_metrics{suffix}.csv")
    model_dir = os.path.join(RESULTS_DIR, f"models{suffix}")
    os.makedirs(model_dir, exist_ok=True)

    device = get_device()
    print(f"device={device} model={args.model}")
    encoder = FrozenBert(device)
    train_df, val_df, test_df = load_csv("train.csv"), load_csv("val.csv"), load_csv("test.csv")
    if args.label == "score_clean":  # drift-cause experiment only: every split uses the noise-free label
        def clean(name):
            full = load_eval_csv(name)
            return full[INPUT_COLS].assign(score=full["score_clean"])
        train_df, val_df, test_df = clean("train.csv"), clean("val.csv"), clean("test.csv")
    assert len(train_df) == hp["n_batches"] * hp["batch_size_data"]
    va = make_tensors(val_df, encoder, device)
    test_y = test_df["score"].values
    first_batch = train_df.iloc[:hp["batch_size_data"]]
    # Evaluation-only frames (score_clean, habit_hit). Never used for training; see replay_diagnostics.csv.
    test_full, train_full = load_eval_csv("test.csv"), load_eval_csv("train.csv")
    if args.label == "score":
        assert (train_full["score"].values == train_df["score"].values).all()
    diag_rows = []

    with open(os.path.join(RESULTS_DIR, f"hparams{suffix}.json"), "w") as f:
        json.dump({**hp, "model": args.model, "device": str(device),
                   "label": args.label, "init": "round-1 and scratch rounds: seed*1000+round; warm: continue from previous round, fresh Adam"},
                  f, indent=2)

    rows = []
    t0 = _time.time()
    for strategy in hp["strategies"]:
        for seed in hp["seeds"]:
            prev_pred, prev_model = None, None
            for rnd in range(1, hp["n_batches"] + 1):
                n_train = rnd * hp["batch_size_data"]
                tr = make_tensors(train_df.iloc[:n_train], encoder, device)
                s = run_seed(seed, rnd)
                torch.manual_seed(s)
                np.random.seed(s)
                model = MODELS[args.model](encoder).to(device)
                if strategy == "warm" and prev_model is not None:
                    model.load_state_dict(prev_model.state_dict())
                epochs, best_val = train(model, tr, va, hp, gen_seed=s)

                m = evaluate(model, test_df)
                pred = model.predict(test_df)
                drift = tau = float("nan")
                if prev_pred is not None:
                    drift = float(np.mean(np.abs(pred - prev_pred)))
                    tau = float(kendalltau(pred, prev_pred).statistic)
                diag_rows.append({"strategy": strategy, "seed": seed, "round": rnd, "n_train": n_train,
                                  **diagnostics(model, test_full, train_full.iloc[:n_train], pred)})
                fb_mse = float(np.mean((model.predict(first_batch) - first_batch["score"].values) ** 2))
                rows.append({"strategy": strategy, "seed": seed, "round": rnd, "n_train": n_train,
                             **m, "drift": drift, "first_batch_mse": fb_mse, "kendall_tau": tau})
                print(f"[{_time.time() - t0:6.0f}s] {strategy:7s} seed={seed} round={rnd} n={n_train:3d} "
                      f"ep={epochs:3d} val={best_val:.4f} test_mse={m['mse']:.4f} rho={m['spearman']:.3f} "
                      f"pw={m['pairwise_acc']:.3f} drift={drift:.4f} fb={fb_mse:.4f} tau={tau:.3f}", flush=True)

                if rnd == hp["n_batches"]:  # final-round weights (used by grid_eval.py / drift_cleanlabel.py)
                    torch.save(model.state_dict(), os.path.join(model_dir, f"{strategy}_seed{seed}_final.pt"))
                prev_pred, prev_model = pred, model
            pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)  # incremental save
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out_csv, index=False)
    pd.DataFrame(diag_rows).to_csv(diag_csv, index=False)
    print(f"wrote {out_csv} ({len(rows)} rows) and {diag_csv}")


if __name__ == "__main__":
    main()
