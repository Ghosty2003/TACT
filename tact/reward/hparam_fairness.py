"""Hyper-parameter fairness check: does tuning change the BertScorer vs OneHotScorer verdict?

    python -m tact.reward.hparam_fairness

Grid (from-scratch, 350 training rows, seeds 0-4, same early stopping as replay.py):
  both architectures : lr in {1e-4, 3e-4, 1e-3}  x  regression head (h1, h2) in {(256, 64), (128, 32)}
  BertScorer only    : x  context embedding dims (emotion, day, time) in {(16, 8, 8), (8, 4, 4)}
  -> 12 configs for BertScorer, 6 for OneHotScorer (the one-hot model has no embeddings).
Each architecture's config with the lowest mean validation MSE (val.csv) is selected, then evaluated on the
full ground-truth grid (grid_eval.py); the grid played no part in selection.
Writes results/hparam_fairness.md/.csv and results/hparam_selected.json (used by noise_robustness.py).
"""
import itertools
import json
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_ind

from tact.core.evaluate import evaluate
from tact.reward.grid_eval import ARCH_NAME, build_grid, grid_metrics, pm
from tact.core.model import MODELS, RESULTS_DIR, FrozenBert, get_device, load_csv, make_tensors
from tact.reward.replay import HPARAMS, run_seed, train

LRS = [1e-4, 3e-4, 1e-3]
HEADS = [(256, 64), (128, 32)]
EMBS = [(16, 8, 8), (8, 4, 4)]
DEFAULT = {"lr": 1e-3, "head": (256, 64), "emb": (16, 8, 8)}


def configs(arch):
    for lr, head in itertools.product(LRS, HEADS):
        if arch == "bert":
            for emb in EMBS:
                yield {"lr": lr, "head": head, "emb": emb}
        else:
            yield {"lr": lr, "head": head}


def kwargs_of(c):
    return {k: v for k, v in c.items() if k != "lr"}


def label(c):
    return f"lr={c['lr']:g}, head={c['head']}" + (f", emb={c['emb']}" if "emb" in c else "")


def main():
    device = get_device()
    encoder = FrozenBert(device)
    train_df, val_df, test_df = load_csv("train.csv"), load_csv("val.csv"), load_csv("test.csv")
    tr, va = make_tensors(train_df, encoder, device), make_tensors(val_df, encoder, device)
    grid, truth, habit = build_grid()

    rows = []
    for arch in ("bert", "onehot"):
        for ci, c in enumerate(configs(arch)):
            hp = {**HPARAMS, "lr": c["lr"]}
            for seed in HPARAMS["seeds"]:
                s = run_seed(seed, 7)   # same seed as the final round of replay.py
                torch.manual_seed(s)
                np.random.seed(s)
                model = MODELS[arch](encoder, **kwargs_of(c)).to(device)
                epochs, best_val = train(model, tr, va, hp, gen_seed=s)
                g = grid_metrics(model.predict(grid), truth, habit)
                rows.append({"arch": arch, "cfg": ci, "label": label(c), "lr": c["lr"], "head": str(c["head"]),
                             "emb": str(c.get("emb", "")), "seed": seed, "epochs": epochs, "val_mse": best_val,
                             "test_mse": evaluate(model, test_df)["mse"], **{f"grid_{k}": v for k, v in g.items()}})
                print(f"{arch:6s} {label(c)} seed={seed} ep={epochs:3d} val={best_val:.4f} "
                      f"grid_mse={g['mse']:.4f} grid_habit_mse={g['habit_mse']:.4f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "hparam_fairness.csv"), index=False)

    cfg = df.groupby(["arch", "cfg", "label"], sort=False).agg(
        val=("val_mse", "mean"), test=("test_mse", "mean"), gmse=("grid_mse", "mean"), gmse_sd=("grid_mse", "std"),
        ghab=("grid_habit_mse", "mean"), gsp=("grid_spearman", "mean"), ep=("epochs", "mean")).reset_index()
    best_cfg, selected = {}, {}
    for arch in ("bert", "onehot"):
        c = cfg[cfg.arch == arch].sort_values("val").iloc[0]
        best_cfg[arch] = int(c.cfg)
        chosen = list(configs(arch))[int(c.cfg)]
        selected[arch] = {"lr": chosen["lr"], "kwargs": {k: list(v) for k, v in kwargs_of(chosen).items()},
                          "label": label(chosen)}
    with open(os.path.join(RESULTS_DIR, "hparam_selected.json"), "w") as f:
        json.dump(selected, f, indent=1)

    def default_cfg(arch):
        return [i for i, c in enumerate(configs(arch))
                if c["lr"] == DEFAULT["lr"] and c["head"] == DEFAULT["head"]
                and c.get("emb", DEFAULT["emb"]) == DEFAULT["emb"]][0]

    out = ["# Hyper-parameter fairness check (from-scratch, 350 training rows, 5 seeds)\n",
           "Grid: lr ∈ {1e-4, 3e-4, 1e-3} × regression head (h1, h2) ∈ {(256, 64), (128, 32)}; BertScorer additionally "
           "context-embedding dims (emotion, day, time) ∈ {(16, 8, 8), (8, 4, 4)}. That is **12 configs for BertScorer and 6 "
           "for OneHotScorer**, so BertScorer gets twice as many chances to win the validation selection. Other settings "
           "(batch 16, Adam, ≤100 epochs, early-stopping patience 15, dropout 0.2) are as in `replay.py`. Selection = lowest "
           "mean validation MSE (val.csv, 5 seeds); the grid columns are evaluated afterwards on the 1512-cell noise-free "
           "grid and played no part in selection. Val MSE is also what early stopping used, so it is slightly optimistic "
           "for every configuration alike.\n",
           "| architecture | config | val MSE | test MSE | grid MSE | grid habit MSE | grid Spearman | mean epochs | selected |",
           "|---|---|---|---|---|---|---|---|---|"]
    for _, c in cfg.iterrows():
        sel = "✔" if best_cfg[c.arch] == c.cfg else ""
        dflt = " (spec default)" if default_cfg(c.arch) == c.cfg else ""
        out.append(f"| {ARCH_NAME[c.arch]} | {c.label}{dflt} | {c.val:.4f} | {c.test:.4f} | {c.gmse:.4f} ± {c.gmse_sd:.4f} | "
                   f"{c.ghab:.4f} | {c.gsp:.3f} | {c.ep:.0f} | {sel} |")

    def pick(arch, ci):
        return df[(df.arch == arch) & (df.cfg == ci)]

    sets = {"spec defaults": {a: pick(a, default_cfg(a)) for a in ("bert", "onehot")},
            "val-selected": {a: pick(a, best_cfg[a]) for a in ("bert", "onehot")}}
    out += ["\n## Architecture comparison\n",
            "| comparison | architecture | config | grid MSE | grid habit MSE | grid non-habit MSE | grid Spearman | grid pairwise acc | grid habit Spearman |",
            "|---|---|---|---|---|---|---|---|---|"]
    for lab, d in sets.items():
        for a in ("bert", "onehot"):
            g = d[a]
            out.append(f"| {lab} | {ARCH_NAME[a]} | {g.label.iloc[0]} | {pm(g.grid_mse)} | {pm(g.grid_habit_mse)} | "
                       f"{pm(g.grid_nonhabit_mse)} | {pm(g.grid_spearman, 3)} | {pm(g.grid_pairwise_acc, 3)} | "
                       f"{pm(g.grid_habit_spearman, 3)} |")
    out += ["\n## BertScorer − OneHotScorer (MSE rows: negative = BertScorer better; Spearman rows: positive = BertScorer better)\n",
            "| comparison | metric | difference | Welch p |", "|---|---|---|---|"]
    for lab, d in sets.items():
        for col in ("grid_mse", "grid_habit_mse", "grid_nonhabit_mse", "grid_spearman", "grid_habit_spearman"):
            diff = d["bert"][col].mean() - d["onehot"][col].mean()
            p = ttest_ind(d["bert"][col], d["onehot"][col], equal_var=False).pvalue
            out.append(f"| {lab} | {col} | {diff:+.4f} | {p:.3f} |")
    best_b, best_o = cfg[cfg.arch == "bert"].gmse.min(), cfg[cfg.arch == "onehot"].gmse.min()
    out.append(f"\nSelected: BertScorer {selected['bert']['label']}; OneHotScorer {selected['onehot']['label']}. "
               f"Best grid MSE anywhere in the sweep (oracle view, not used for selection): BertScorer {best_b:.4f}, "
               f"OneHotScorer {best_o:.4f}.")
    with open(os.path.join(RESULTS_DIR, "hparam_fairness.md"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
