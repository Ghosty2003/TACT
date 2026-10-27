"""v3, step 2b: Appendix C habit drift with the recency-weighted default (results/v3/drift_*.csv).

    TACT_V2=1 HF_HUB_OFFLINE=1 python -m tact.drift.v3_drift

The drift stream and the NO-DRIFT stream of Appendix C (drift_targeted.Sim = the simulation of drift_trigger.py plus recency weighting), reporting seeds 0-4, new default R<h*> (chosen half-life of results/v3/chosen_half_life.json).
The old default (W), the windowed method (S) and the tabular agents are NOT re-run: W and S of drift_targeted.py reproduce results/v2/drift_trigger_series.csv exactly (checked), tabular C comes from that file.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import json
import time

import pandas as pd

from tact.decision import budget as B
from tact.drift import drift_targeted as T
from tact.drift import drift_trigger as D
from tact.core.model import ROOT, FrozenBert, get_device

OUT = os.path.join(ROOT, "results", "v3")


def main():
    h = json.load(open(os.path.join(OUT, "chosen_half_life.json")))["half_life"]
    m = f"R{h}"
    dev = get_device()
    enc = FrozenBert(dev)
    B.Data(enc, dev, data_dir=os.path.join(ROOT, "data_dpo"))
    ctx = D.Ctx(enc, dev)
    path = os.path.join(OUT, "drift_final.csv")
    if os.path.exists(path):
        os.remove(path)
        for n in ("series", "rounds"):
            os.remove(os.path.join(OUT, f"drift_{n}.csv"))
    for run in ("drift", "ctrl"):
        st = D.Stream(ctx, "cont", run)
        for seed in D.SEEDS:
            t0 = time.perf_counter()
            series, rounds, _, fin = T.Sim(ctx, st, seed, m, None).run()
            meta = {"run": run, "method": m, "seed": seed}
            T.append(os.path.join(OUT, "drift_series.csv"), pd.DataFrame([{**meta, "i": i, **v} for i, v in series.items()]))
            T.append(os.path.join(OUT, "drift_rounds.csv"), pd.DataFrame([{**meta, **x} for x in rounds]))
            T.append(path, pd.DataFrame([{**meta, **fin}]))
            print(f"[{run} {m} seed={seed}] fs_new(end)={series[T.T_END]['fs_new']:.3f} test={fin['test_regret']:.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
