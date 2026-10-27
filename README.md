# TACT

Code for **TACT**: a small learned model that decides, from the **context** of a household robot (the user's emotion, the day of the week, the time of
day, optionally more variables), whether to **ask a clarifying question** about an ambiguous instruction or to **act directly**.

The experiments run on a synthetic generator with a known ground truth, so the learned preference can be compared with the true one cell by cell.
This repository reproduces all experiments of the paper; datasets, results and figures are generated and are not stored here.

> **Versions.** `main` is the code of the paper (**v2**). The earlier course-project code (DPO policy on a frozen `bert-base-uncased` CLS vector, original
> layout) is kept under the tag **`v1-course`**. `TACT_V2=0` switches the models of `main` back to the v1 architecture for comparison.

## Method (v2, as in the paper)

* **Representation.** The question text is encoded by the frozen `sentence-transformers/bert-base-nli-mean-tokens` (mean pooling, 768-d, cached per distinct text).
* **Reward model.** The context (emotion, day, time, optional extra variables) goes through learned embeddings and a *Context encoder* (64-d); a rank-16 bilinear
  term couples the text vector and the context vector, and a small regression head predicts the preference for each candidate (ask / act). Only the context encoder,
  the bilinear term and the head are trained: 253,617 parameters, 0.232 % of the 109.5 M-parameter backbone.
* **Decision rule (Eq. 2).** Ask iff `r_hat(ask) - r_hat(act) > 0` (threshold rule); no SFT / DPO stage in the main pipeline (a DPO policy is kept as an appendix ablation).
* **Semi-online retraining (default).** The reward model is retrained after every round of 50 interactions by a **recency-weighted warm start**: every stored scenario
  gets the weight `0.5 ** (age in rounds / h)` with half-life `h = 2` rounds (selected on validation seeds only, `tact.drift.v3_select`), validation loss unweighted,
  early stopping as before. Cold start: the first 50 interactions are asked.
* **Baseline.** A tabular agent (constant learning rate 0.5, back-off chain cell -> topic mean -> global mean).
* **Synthetic users.** `r*(ask) = necessity(topic, day, time) x valence(emotion) x availability(day, time)`, `r*(act) = 1 - necessity`, noise N(0, 0.05); habit rules
  ("on Saturdays the fruit is always apples") lower the necessity in some cells. The generator can add context variables with multiplicative rules, other functional
  forms of the truth, 0/1 feedback, and whole simulated users (own valence table, own habits).

## Layout

```
tact/
  core/        model.py (frozen encoder, scorers, data helpers)   heads.py (Context encoder + bilinear head)   generate_dataset.py (the synthetic generator)
               evaluate.py  grid_eval.py (1512-cell noise-free grid)  plots.py (ground-truth helpers)  recency.py (recency-weighted replay)  hard_paraphrase_texts.py
  decision/    budget.py (data-budget sweep, tabular agent, semi-online loop)   v2_main_table.py  hard_paraphrase.py  v2_ablation_dpo.py (threshold vs DPO)
               binary_feedback.py (0/1 feedback)   v2_partial_feedback.py  v2_cold_start.py  v3_partial_cold.py (both with the recency-weighted default)
               v2_rebuttal_cost.py / _graded.py / _tabular.py / _report.py  (constant-cost baselines, emotion contribution, graded feedback, tabular tuning, paired tests)
  reward/      replay.py (7 rounds x 50, warm vs scratch; Sec. 5.4)   v3_semionline.py (the same with recency weights)   heatmap_arch.py
               v2_context_ablation.py (context ablation, emotion-label noise)   noise_robustness.py
  robustness/  context_scaling.py (3 to 7 context variables)   interaction_diag.py   extensibility.py / v2_extensibility.py   misspecification.py   compositional.py
               multiuser.py (10 simulated users)   ood_eval.py  v2_ood_ask.py  v2_ood_priority.py  v3_paper_ood.py (novel topics)
  drift/       habit_drift.py (helpers)  drift_trigger.py (the 15-round habit-drift stream)  v2_drift.py  v2_drift_retention.py
               v3_select.py (half-life selection, seeds 5-9)  v3_drift.py (new default)  drift_targeted.py (targeted data invalidation: an archived negative result)
  reporting/   make_final_figures.py (single-column 84 mm vector figures)   v3_report.py (old vs new default, fig_semionline, fig_drift)   v3_paper_numbers.py
```

## Setup

```bash
pip install -r requirements.txt     # tested with Python 3.13, torch 2.14, transformers 5.20, numpy 2.5, pandas 3.0, scipy 1.18, matplotlib 3.11
```

The two encoders are downloaded from the Hugging Face hub on first use (`HF_HUB_OFFLINE=1` afterwards avoids network calls). A GPU is used automatically when available
(CUDA, then MPS, then CPU); `DEVICE=cpu|cuda|mps` overrides it. Scripts run from the repository root as modules (`python -m tact.<package>.<module>`) and write to
`results/v2/` (v3 outputs to `results/v3/`), created on demand. Every module states its design, the criteria fixed beforehand and its limits in its docstring.
Training is deterministic for a given machine (seeded, deterministic CUDA kernels); numbers can differ in the last digits across hardware.

## Data

```bash
python -m tact.core.generate_dataset --n 500  --seed 42 --outdir data                      # ask-only data; also writes data/config.json (read by tact.core.model)
python -m tact.core.generate_dataset --n 500  --seed 42 --with-act --outdir data_dpo       # ask + act candidates, phrasing split: the main data
```

Further datasets (1000 / 2000 scenarios, 0/1 labels, extra context variables, truth forms, simulated users, the drift streams) are generated on demand by the scripts that
need them. Generator options: `--with-act`, `--extra-vars location,companion,activity,noise_level`, `--truth-form {product,additive,saturating}`, `--user-seed k`,
`--label {continuous,binary}`.

## Reproducing the experiments

| paper item | command | main outputs (`results/v2/` unless noted) |
|---|---|---|
| main table, budget curve, crossover | `python -m tact.decision.v2_main_table`, `python -m tact.decision.budget` | `main_table.md`, `budget_runs.csv`, `budget_summary.md`, `dpo_semionline.csv` |
| robustness to unseen / hard paraphrases | `python -m tact.decision.hard_paraphrase` | `hard_paraphrase_runs.csv` |
| threshold rule vs DPO (appendix) | `python -m tact.decision.v2_ablation_dpo` | `ablation_threshold_vs_dpo.csv` |
| heatmap (from-scratch models) | `python -m tact.reward.replay`, `python -m tact.reward.heatmap_arch` | `heatmap_stats.json`, `models/` |
| context ablation, emotion-label noise | `python -m tact.reward.v2_context_ablation` | `context_ablation.md` |
| context scaling, 7-variable diagnosis | `python -m tact.robustness.context_scaling`, `python -m tact.robustness.interaction_diag` | `context_scaling_runs.csv`, `interaction_diag.md` |
| ten users | `python -m tact.robustness.multiuser` | `multiuser_matrix_A.csv` |
| other robustness (extra variable, truth forms, compositional, novel topics) | `tact.robustness.{v2_extensibility, misspecification, compositional, ood_eval, v2_ood_ask, v2_ood_priority}` | `extensibility.md`, `misspecification.md`, ... |
| 0/1 feedback | `python -m tact.decision.binary_feedback` | `binary_feedback.md` |
| **half-life selection** (validation seeds 5-9) | `python -m tact.drift.v3_select` | `results/v3/select.md`, `chosen_half_life.json` |
| **Sec. 5.4 semi-online stability** (recency-weighted default) | `python -m tact.reward.v3_semionline` | `results/v3/semionline_*.csv` |
| **habit drift** (appendix) | `python -m tact.drift.v2_drift`, `python -m tact.drift.v3_drift`, `python -m tact.drift.v2_drift_retention` | `drift_baselines.md`, `results/v3/drift_*.csv` |
| **partial feedback, cold start** | `python -m tact.decision.v3_partial_cold` (unweighted: `v2_partial_feedback`, `v2_cold_start`) | `results/v3/partial_runs.csv`, `cold_runs.csv` |
| novel-topic regret under the new default | `python -m tact.robustness.v3_paper_ood base` and `... prio` | `results/v3/ood_*_v3.csv` |
| rebuttal experiments | `python -m tact.decision.v2_rebuttal_{cost,graded,tabular}`, then `..._report` | `results/v2/rebuttal/rebuttal.md` |
| figures and numbers | `python -m tact.reporting.make_final_figures` (`FIG_ONLY=a,b` for a subset), `python -m tact.reporting.v3_report`, `python -m tact.reporting.v3_paper_numbers` | `figures_final/`, `figures_v3/`, `results/v3_recency.md`, `results/v3_paper_numbers.md` |

`tact.drift.drift_targeted` (drift-triggered targeted data invalidation) is an archived negative result and is not used by the paper.
The reporting scripts read the csv files of the experiments above and fail if they have not been produced yet.

## Headline results (synthetic data, 5 seeds, 75 test scenarios)

* TACT catches up with the tabular agent after about 50 training scenarios (a tie) and is ahead from 75; at 350 scenarios the regret is 0.036 against 0.079
  and the decision agreement with the oracle 0.797 against 0.613.
* Warm-started semi-online retraining matches from-scratch accuracy (round-7 MSE about 0.013 for both) with lower prediction drift between rounds (about 0.058 vs 0.078);
  the recency weights make the model adapt to a changed habit after about 160 interactions instead of 530, while the other habits stay learned.
* Own-user truth vs other users' truths (ten simulated users): column-normalised regret 0.62 on the diagonal against 1.04 off the diagonal.

Exact numbers, with their definitions, are in the reports written to `results/` by the scripts above.
