"""Data-scale sweep: does the Saturday-fruit collapse sharpen with more training data?

    python -m tact.reward.scaling

Datasets: the original 500-sample set (train = 350; the saved final from-scratch checkpoints of replay.py)
and data_1000/, data_2000/ generated with `generate_dataset.py --n N --seed 42` (train = 700 / 1400).
For the two new sets a from-scratch model is trained on the full train.csv (seeds 0-4, same hyper-parameters
and per-seed init as replay.py's final round) and early-stopped on that set's val.csv.
Per model we record the collapse ratio (Saturday mean / other-days mean of the 6x7 emotion x day grid for the
fixed salad-fruit question, afternoon) and the MSE on the habit-hit subset of that set's test.csv.
Writes results/scaling_metrics.csv and results/scaling_grids.json.
"""
import json
import os

import numpy as np
import pandas as pd
import torch

from tact.core.evaluate import diagnostics, evaluate
from tact.core.model import (DATA_DIR, RESULTS_DIR, ROOT, BertScorer, FrozenBert, get_device, load_csv, load_eval_csv,
                   make_tensors)
from tact.core.plots import heatmap_stats, pred_grid, truth_grid
from tact.reward.replay import HPARAMS, run_seed, train

SETS = [  # (data dir, final from-scratch round shares the replay seeds)
    ("data", DATA_DIR),
    ("data_1000", os.path.join(ROOT, "data_1000")),
    ("data_2000", os.path.join(ROOT, "data_2000")),
]


def main():
    device = get_device()
    encoder = FrozenBert(device)
    rows, grids = [], {}
    for name, ddir in SETS:
        train_df, val_df = load_csv("train.csv", ddir), load_csv("val.csv", ddir)
        test_full, train_full = load_eval_csv("test.csv", ddir), load_eval_csv("train.csv", ddir)
        n_train = len(train_df)
        va, tr = make_tensors(val_df, encoder, device), make_tensors(train_df, encoder, device)
        grids[str(n_train)] = []
        for seed in HPARAMS["seeds"]:
            model = BertScorer(encoder).to(device)
            ckpt = os.path.join(RESULTS_DIR, "models", f"scratch_seed{seed}_final.pt")
            if name == "data":  # reuse replay.py's final-round from-scratch models
                model.load_state_dict(torch.load(ckpt, map_location=device))
                model.eval()
            else:
                s = run_seed(seed, 7)
                torch.manual_seed(s)
                np.random.seed(s)
                model = BertScorer(encoder).to(device)
                epochs, _ = train(model, tr, va, HPARAMS, gen_seed=s)
            m = evaluate(model, test_full)
            d = diagnostics(model, test_full, train_full)
            g = pred_grid(model)
            grids[str(n_train)].append(g.round(4).tolist())
            rows.append({"dataset": name, "n_train": n_train, "seed": seed,
                         "sat_over_other": heatmap_stats(g)["sat_over_other"],
                         "habit_n": d["habit_n"], "habit_mse": d["habit_mse"], "nonhabit_mse": d["nonhabit_mse"],
                         "mse": m["mse"], "spearman": m["spearman"], "pairwise_acc": m["pairwise_acc"]})
            print(f"{name:9s} n_train={n_train:4d} seed={seed} sat/other={rows[-1]['sat_over_other']:.3f} "
                  f"habit_mse={d['habit_mse']:.4f} test_mse={m['mse']:.4f}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "scaling_metrics.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "scaling_grids.json"), "w") as f:
        json.dump({"truth_sat_over_other": heatmap_stats(truth_grid())["sat_over_other"], "grids": grids}, f)
    print("wrote scaling_metrics.csv, scaling_grids.json")


if __name__ == "__main__":
    main()
