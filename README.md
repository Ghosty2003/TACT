# TACT

Code for **TACT**, a small learned model that decides, from the **context** of a household robot (the user's emotion, the day of the week,
the time of day, optionally more), whether to **ask a clarifying question** about an ambiguous instruction or to **act directly**.

Everything is evaluated on a synthetic generator with a known ground truth, so that the learned preference can be compared with the true one
cell by cell. The code in this repository reproduces all experiments; datasets, results and figures are generated and are not stored here.

## Method in one paragraph

* **Reward model.** A frozen `bert-base-uncased` encodes the instruction/question text (the `[CLS]` vector, cached per distinct text). The
  context (emotion, day, time, ...) goes through learned embeddings and a two-layer *Context MLP*; both are concatenated and a small regression
  head predicts how much the user prefers each candidate (ask / act). Only the embeddings, the MLP and the head are trained
  (about 0.24 M parameters, 0.2 % of BERT).
  A baseline `OneHotScorer` feeds a one-hot context straight into the same head.
* **Policy (TACT).** An SFT stage on the observed better action, followed by DPO (beta = 0.1) on preference pairs labelled by the reward model;
  the policy chooses between the two candidates `{act, ask}`. Directly taking the argmax of the reward model is the reference ("direct reward
  maximisation"). A tabular agent with constant learning rate 0.5 and a fall-back chain is the non-parametric comparison.
* **Synthetic users.** `r*(ask) = necessity(topic, day, time) x valence(emotion) x availability(day, time)`, `r*(act) = 1 - necessity`, noise
  N(0, 0.05); habit rules ("on Saturdays the fruit is always apples") drive the necessity down in some cells. The generator can add context
  variables with multiplicative rules, other functional forms of the truth, and whole simulated users (own valence table, own habits).

## Layout

```
tact/
  core/        model.py (BertScorer, OneHotScorer, FrozenBert, data helpers)   evaluate.py (MSE, Spearman, pairwise accuracy)
               generate_dataset.py (the synthetic generator)                    plots.py (shared ground-truth helpers, heatmap)
  reward/      incremental retraining of the reward model and its analysis
               replay.py (7 rounds x 50 samples, from-scratch vs warm-start, stability metrics)   grid_eval.py (full-grid truth evaluation)
               hparam_fairness.py  noise_robustness.py  drift_cleanlabel.py  scaling.py  ablation_context.py  heatmap_arch.py  plots_extra.py
  decision/    ask-vs-act decisions
               budget.py (reward model + SFT + DPO, argmax, tabular agents; data-budget sweep, semi-online loop, cost)
               hard_paraphrase.py (+ hard_paraphrase_texts.py)  semionline_gap.py  hybrid.py  utility.py  beta_sweep.py
  robustness/  stress tests and extensions
               ood_eval.py  extensibility.py (4th context variable)  context_scaling.py (3 to 7 variables)  compositional.py (unseen combinations)
               misspecification.py (non-product truths)  habit_drift.py  multiuser.py (10 simulated users)
  reporting/   transfer.py (one-hot vs Context MLP side by side)   make_paper_figures.py (single-column vector figures)
requirements.txt
```

## Setup

```bash
pip install -r requirements.txt     # tested with Python 3.13, torch 2.14, transformers 5.20, numpy 2.5, pandas 3.0, scipy 1.18, matplotlib 3.11
```

`bert-base-uncased` is downloaded from the Hugging Face hub on first use. A GPU is used automatically when available (CUDA, then MPS, then CPU);
`DEVICE=cpu|cuda|mps` overrides it. All scripts are run from the repository root as modules (`python -m tact.<package>.<module>`) and write to
`results/` (created on demand).

## Reproducing the experiments

1. **Data** (all generated, seed 42):

```bash
python -m tact.core.generate_dataset --n 500  --seed 42 --outdir data                         # ask-only data (reward-model experiments)
python -m tact.core.generate_dataset --n 500  --seed 42 --with-act --outdir data_dpo          # ask + act candidates, phrasing split (decisions)
python -m tact.core.generate_dataset --n 1000 --seed 42 --outdir data_1000                    # only for reward/scaling.py
python -m tact.core.generate_dataset --n 2000 --seed 42 --outdir data_2000
python -m tact.core.generate_dataset --n 500  --seed 42 --with-act --with-location --outdir data_ext
```

The experiments below create any further datasets they need (extra context variables, truth forms, simulated users) on their own.
Generator options: `--with-act`, `--with-location`, `--extra-vars location,companion,activity,noise_level`, `--truth-form {product,additive,saturating}`,
`--user-seed k`.

2. **Reward-model experiments** (the numbers of this family were produced on the CPU; `DEVICE=cpu` reproduces them exactly):

```bash
DEVICE=cpu python -m tact.reward.replay                              # also: --model onehot, --label score_clean, both
DEVICE=cpu python -m tact.core.plots
DEVICE=cpu python -m tact.reward.scaling
DEVICE=cpu python -m tact.reward.plots_extra
DEVICE=cpu python -m tact.reward.grid_eval
DEVICE=cpu python -m tact.reward.hparam_fairness
DEVICE=cpu python -m tact.reward.noise_robustness
DEVICE=cpu python -m tact.reward.drift_cleanlabel                    # needs the --label score_clean replays
```

3. **Decision experiments** (run on a GPU):

```bash
python -m tact.decision.budget            # data-budget sweep, DPO policy, semi-online loop, parameter / time cost
python -m tact.decision.hard_paraphrase   # zero-shot hard paraphrases
python -m tact.decision.semionline_gap
python -m tact.decision.hybrid
python -m tact.decision.utility           # clarification utility, appended to results/budget_summary.md
python -m tact.decision.beta_sweep
```

4. **Robustness and extensions**:

```bash
python -m tact.robustness.ood_eval
python -m tact.reward.ablation_context
python -m tact.robustness.extensibility
python -m tact.robustness.context_scaling
python -m tact.robustness.compositional
python -m tact.robustness.misspecification
python -m tact.robustness.habit_drift
python -m tact.robustness.multiuser
```

5. **One-hot encoding as the main model, and figures**:

```bash
ARCH=onehot python -m tact.decision.budget            # likewise hard_paraphrase and utility; outputs go to results/onehot_as_main/
ARCH=onehot python -m tact.reward.heatmap_arch
python -m tact.reporting.transfer
python -m tact.reporting.make_paper_figures           # figures/*.pdf (8.6 cm wide)
```

Each script states its design, the criteria fixed beforehand and its limits in its module docstring and writes a markdown summary plus csv files
to `results/`.

## Headline results (synthetic data, 5 seeds, 75 test scenarios)

| | |
|---|---|
| Reward model, 350 samples, full grid vs the true preference | MSE 0.011, Spearman 0.855 |
| Warm-start vs from-scratch retraining | prediction drift 0.047 vs 0.060, forgetting (first batch) 0.0066 vs 0.0117 |
| Ask/act decision at N = 350 scenarios: regret of TACT / tabular agent / always-ask / never-ask | 0.036 ± 0.008 / 0.079 / 0.158 / 0.157 |
| Agreement with the oracle decision | TACT 0.789 ± 0.024, tabular 0.613, always-ask 0.64, never-ask 0.36 |
| Data budget | TACT matches the tabular agent at N = 75 scenarios, is clearly better from N = 100, and halves the best constant policy's regret by N = 150; below N = 75 the tabular agent is as good or better |
| Hard paraphrases of the question (zero-shot) | regret 0.050 ± 0.015 (policy) vs 0.092 ± 0.010 (direct reward maximisation) |
| Ten simulated users, two with a non-monotone emotion mapping | a user's own model is the better match for its own truth than the average other model for 10 of 10 truths (diagonal-dominant cross matrix, off-diagonal / diagonal regret 1.53); the non-monotone mapping is learned |

Known limits, all measured in the scripts: the habit collapses are learned only partly at 350 samples (about 60 % of the true collapse); a rule
that combines two variables (activity x topic) is not learned; with an additive ground truth a topic-level table beats the policy; the
data are synthetic, test sets are small (75 scenarios) and most comparisons use 5 seeds, so differences below about 0.01 in regret are not
resolved.

## Reproducibility

Seeds are fixed. Runs are deterministic on a given device (CUDA runs use deterministic algorithms), but training reacts to perturbations of
about 1e-6, so a given seed differs between CPU and GPU; compare statistics across seeds, not single-seed values, when changing device.
