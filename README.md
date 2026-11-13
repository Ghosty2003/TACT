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
               v2_rebuttal_cost.py / v2_rebuttal_graded.py / v2_rebuttal_tabular.py / v2_rebuttal_report.py  (constant-cost baselines, emotion contribution, graded feedback, tabular tuning, paired tests)
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

Most scripts create the datasets they need; four sets must be generated by hand once, from the repository root:

```bash
python -m tact.core.generate_dataset --n 500  --seed 42 --outdir data                           # ask-only data (+ data/config.json, read by tact.core.model); used by tact.reward.*
python -m tact.core.generate_dataset --n 500  --seed 42 --with-act --outdir data_dpo            # ask + act candidates, phrasing split: the main data of almost every experiment
python -m tact.core.generate_dataset --n 1000 --seed 42 --with-act --outdir data_cont_n1000     # continuous-label reference, N = 700 (read directly by v2_ood_priority, v3_paper_ood)
python -m tact.core.generate_dataset --n 500  --seed 42 --with-act --with-location --outdir data_ext   # extra "location" variable (extensibility)
```

All other datasets (`data_ctx*`, `data_diag*`, `data_users/`, `data_comp`, `data_ext_pool`, `data_truth_*`, `data_bin*`, `data_cont_n2000`, the drift streams `data_drift*`) are
generated on first use by the script that needs them. Generator options: `--with-act`, `--with-location`, `--extra-vars location,companion,activity,noise_level`,
`--truth-form {product,additive,saturating}`, `--user-seed k`, `--label {continuous,binary}`.

## Reproducing the experiments

### Run order

Steps within a stage are independent unless noted. Every script is resumable (finished runs are skipped) and writes to `results/v2/` (v3 scripts to `results/v3/`).

| stage | commands | needs | produces |
|---|---|---|---|
| 0. data | the four commands above | - | `data/`, `data_dpo/`, `data_cont_n1000/`, `data_ext/` |
| 1. core experiments | `python -m tact.decision.budget` (budget sweep + the semi-online loop of the unweighted default)<br>`python -m tact.decision.v2_main_table`<br>`python -m tact.decision.hard_paraphrase` (reads `budget_runs.csv`)<br>`python -m tact.decision.v2_ablation_dpo`<br>`python -m tact.reward.replay` (7 rounds x 50, warm vs scratch; ask-only `data/`) | 0 | `budget_runs.csv`, `dpo_semionline.csv`, `main_table.md`, `hard_paraphrase_runs.csv`, `ablation_threshold_vs_dpo.csv`, `replay_metrics.csv`, `models/` |
| 2. heatmap | `python -m tact.reward.heatmap_arch` | `models/` from `reward.replay` | `heatmap_stats.json` |
| 3. robustness | `python -m tact.robustness.context_scaling`, `.interaction_diag` then `.interaction_diag_report`, `.multiuser`, `.v2_extensibility` (reads `budget_runs.csv`), `.misspecification`, `.compositional`, `.ood_eval` then `.v2_ood_ask` and `.v2_ood_priority`, `python -m tact.reward.v2_context_ablation`, `python -m tact.reward.noise_robustness --defaults` (without `--defaults` it reads `hparam_selected.json` of the course-project script `hparam_fairness.py`, which is not in this repository) | 0, 1 | `context_scaling_runs.csv`, `interaction_diag.md`, `multiuser_matrix_A.csv`, `extensibility.md`, `misspecification.md`, `ood_*.csv`, `context_ablation.md`, ... |
| 4. feedback variants | `python -m tact.decision.binary_feedback` (generates `data_bin*`, `data_cont_n2000`)<br>`python -m tact.decision.binary_feedback --draws` (three independent label draws; needed by stage 7) | 0, 1 | `binary_feedback.md`, `binary_feedback_runs.csv`, `binary_feedback_draws_runs.csv` |
| 5. habit drift (old default) | `python -m tact.drift.habit_drift`<br>`python -m tact.drift.drift_trigger --stage v2` (15-round stream, warm start / last-3-rounds / tabular; generates `data_drift15_src`, `data_drift15/cont_drift`; without `--stage v2` it also runs the v1 DPO-based local-relearning method, which fails with the v2 architecture)<br>`python -m tact.drift.drift_trigger_report`<br>`python -m tact.drift.v2_drift`, `python -m tact.drift.v2_drift_retention` | 0 | `drift_baselines.md`, `drift_retention.md`, `drift_trigger_*.csv` |
| 6. targeted-invalidation controls, partial feedback, cold start | `python -m tact.drift.drift_targeted` then `python -m tact.drift.drift_targeted_report`<br>`python -m tact.decision.v2_partial_feedback`<br>`python -m tact.decision.v2_cold_start` (both read `data_drift15/cont_ctrl`, hence after `drift_targeted`) | 0, 1, 5 | `drift_targeted_{series,rounds,final,events}.csv`, and the no-drift stream `data_drift15/cont_ctrl` (read by `v3_report` and `v3_paper_numbers` for the old-default `W` and windowed `S` rows of the drift figure) |
| 7. rebuttal experiments | `python -m tact.decision.v2_rebuttal_cost`, `.v2_rebuttal_graded` (needs `data_bin*`, `data_cont_n1000`), `.v2_rebuttal_tabular`, then `python -m tact.decision.v2_rebuttal_report` | 1, 4 | `rebuttal/rebuttal.md`, `rebuttal/*_runs.csv` (the *Verdicts* section of `rebuttal.md` is the hand-written `rebuttal/verdicts.md` and is embedded only if that file exists; it is not produced by any script) |
| 8. new default: half-life selection | `python -m tact.drift.v3_select` (validation seeds 5-9) | 5 | `results/v3/select.md`, **`results/v3/chosen_half_life.json`** (read by every script below) |
| 9. new default: experiments | `python -m tact.reward.v3_semionline`<br>`python -m tact.drift.v3_drift` (the old-default and tabular rows are taken from `drift_trigger_series.csv`, stage 5)<br>`python -m tact.decision.v3_partial_cold` (`--check` = integrity check against stage 4)<br>`python -m tact.robustness.v3_paper_ood base` **then** `python -m tact.robustness.v3_paper_ood prio` (separate processes: the encoder cache order must not mix) | 1, 4, 5, 6, 8, `data_cont_n1000` | `results/v3/semionline_*.csv`, `drift_*.csv`, `partial_runs.csv`, `cold_runs.csv`, `ood_*_v3.csv` |
| 10. figures and numbers | `python -m tact.reporting.make_final_figures` (`FIG_ONLY=a,b` for a subset; needs 1-3)<br>`python -m tact.reporting.v3_report` (needs 1, 5, 6, 8, 9: it reads `drift_targeted_*.csv` for the old-default and windowed rows; also writes `fig_semionline`, `fig_drift`)<br>`python -m tact.reporting.v3_paper_numbers` (needs 1-9) | all above | `figures_final/`, `figures_v3/`, `results/v3_recency.md`, `results/v3_paper_numbers.md` |

### Notes

* **Two defaults.** The *old* default is the unweighted warm start (stages 1-7, results in `results/v2`); the *new* default of the paper is the recency-weighted warm start
  (stages 8-9, results in `results/v3`). The v3 reports compare the two, so both must have been run.
* `tact.drift.drift_targeted` is the archived targeted-invalidation experiment (a negative result), but its simulator class (`Sim`, with recency weighting) is imported by
  `v3_select` and `v3_drift`, and its csv files feed `v3_report` and `v3_paper_numbers`; do not skip stage 6.
* The report modules (`*_report`) can be run alone to rebuild a markdown report from existing csv files; the reporting scripts stop with a file-not-found error if an input is missing.
* Sanity check: `v3_partial_cold --check` and the integrity paragraph of `results/v3_recency.md` compare the unweighted rerun with the stage 1 / 4 files
  (round 1 reproduces to ~1e-9; later rounds agree only to numerical noise).
* **Hand-written files** that are not regenerated: `results/v2/rebuttal/verdicts.md`, `results/v2/interaction_diag_findings.md` (appended to the rebuttal / interaction-diagnosis reports when present).
* **Verified reproduction.** The complete stage list above was rerun from scratch in a fresh clone (RTX 4090, CUDA, seeds as in the scripts): all 47 commands ran to completion and the
  result csv files (`budget_runs`, `replay_metrics`, `drift_targeted_*`, `select_*`, `semionline_*`, `drift_*`, `multiuser_matrix_A`, ... ) are byte-identical to the original run;
  `results/v3_paper_numbers.md` and `results/v3_recency.md` are identical apart from timings. Total compute about 2.6 h on that machine.
* Environment variables: `TACT_V2=0` (v1 architecture), `DEVICE`, `HF_HUB_OFFLINE=1`, `FIG_ONLY`. Each stage can be run separately.

## Headline results (synthetic data, 5 seeds, 75 test scenarios)

* TACT catches up with the tabular agent after about 50 training scenarios (a tie) and is ahead from 75; at 350 scenarios the regret is 0.036 against 0.079
  and the decision agreement with the oracle 0.797 against 0.613.
* Warm-started semi-online retraining matches from-scratch accuracy (round-7 MSE about 0.013 for both) with lower prediction drift between rounds (about 0.058 vs 0.078);
  the recency weights make the model adapt to a changed habit after about 160 interactions instead of 530, while the other habits stay learned.
* Own-user truth vs other users' truths (ten simulated users): column-normalised regret 0.62 on the diagonal against 1.04 off the diagonal.

Exact numbers, with their definitions, are in the reports written to `results/` by the scripts above.
