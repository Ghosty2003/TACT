"""Reward models + shared data helpers for the incremental-retraining experiment.

BertScorer  : frozen bert-base-uncased CLS (768) + learned context stream (64) -> regression head.
OneHotScorer: baseline, one-hot context (6+7+3 = 16) appended to the CLS vector, no context MLP.

BERT is frozen and kept in eval mode, so the CLS vector of a given question is a constant.
`FrozenBert` therefore encodes each distinct question once and caches it; this is numerically
identical to running BERT on every batch, and makes the 70 training runs take seconds each.
Only the embeddings, the context MLP and the regression head are trained.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repository root
# The task says data lives in data/; in this repo the csv files sit in the repo root.
DATA_DIR = os.path.join(ROOT, "data") if os.path.isdir(os.path.join(ROOT, "data")) else ROOT
# ARCH=onehot (environment variable) switches the main reward/policy model of budget.py, hard_paraphrase.py and utility.py
# to OneHotScorer and redirects their output to results/onehot_as_main/. Default: BertScorer, results/.
# TACT_V2 (default 1; TACT_V2=0 = the original course-project model of the tag v1-course) selects the final architecture of Experiment D for EVERY script (results/arch_selection.md): the frozen sentence encoder
# sentence-transformers/bert-base-nli-mean-tokens with mean pooling, the Context encoder + rank-16 bilinear head (heads.BilinearScorer) under the
# name "bert", and results written to results/v2/. TACT_V2=0: the original model (CLS pooling, BertScorer, results/).
V2 = os.environ.get("TACT_V2", "1") != "0"
ARCH = os.environ.get("ARCH", "bert")
assert ARCH in ("bert", "onehot"), ARCH
RESULTS_DIR = (os.path.join(ROOT, "results", "v2") if V2 else
               (os.path.join(ROOT, "results") if ARCH == "bert" else os.path.join(ROOT, "results", "onehot_as_main")))
os.makedirs(RESULTS_DIR, exist_ok=True)
BERT_NAME = "sentence-transformers/bert-base-nli-mean-tokens" if V2 else "bert-base-uncased"
POOL = "mean" if V2 else "cls"   # "cls": last-layer [CLS]; "mean": attention-mask mean over the last layer's tokens

with open(os.path.join(DATA_DIR, "config.json")) as _f:
    CONFIG = json.load(_f)
EMOTIONS = CONFIG["emotions"]  # neutral/calm/happy/sad/angry/fearful -> index 0..5
EMOTION_IDX = {e: i for i, e in enumerate(EMOTIONS)}
N_EMOTIONS, N_DAYS, N_TIMES = len(EMOTIONS), 7, 3
LOCATIONS = ["kitchen", "living_room", "office"]   # optional 4th context variable (data_ext, extensibility.py)
LOCATION_IDX = {l: i for i, l in enumerate(LOCATIONS)}
N_LOCATIONS = len(LOCATIONS)
# Generic extra context variables (context_scaling.py): present in a csv as columns of these names, fed to the model as one
# LongTensor `extra` of shape [N, k] in this order.
EXTRA_SPEC = {"location": LOCATIONS, "companion": ["alone", "with_guests"], "activity": ["relaxing", "working", "cooking"],
              "noise_level": ["quiet", "noisy"]}
EXTRA_ORDER = list(EXTRA_SPEC)

# Columns the model may see. necessity/receptiveness/score_clean/habit_hit are never loaded for training.
INPUT_COLS = ["question", "emotion", "day", "time", "score"]


def get_device():
    """CUDA if available, else MPS, else CPU. Override with DEVICE=cpu|mps|cuda."""
    if os.environ.get("DEVICE"):
        return torch.device(os.environ["DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve(path, data_dir):
    """A bare file name always resolves inside `data_dir` (default DATA_DIR), never against the cwd."""
    if os.path.dirname(path):
        return path
    return os.path.join(data_dir or DATA_DIR, path)


def load_csv(path, data_dir=None):
    """Read only the columns the model is allowed to use (training / prediction path)."""
    return pd.read_csv(_resolve(path, data_dir), usecols=INPUT_COLS)


# Extra columns that exist only for post-hoc diagnostics. NEVER feed these to a model or use them in training.
EVAL_ONLY_COLS = ["score_clean", "habit_hit"]


def load_eval_csv(path, data_dir=None):
    """EVALUATION ONLY: model inputs + score_clean + habit_hit (for noise / habit-subset diagnostics)."""
    return pd.read_csv(_resolve(path, data_dir), usecols=INPUT_COLS + EVAL_ONLY_COLS)


class FrozenBert:
    """bert-base-uncased, frozen, eval mode; returns the [CLS] vector (768-d), cached per question."""

    def __init__(self, device=None, name=BERT_NAME):
        from transformers import AutoModel, AutoTokenizer
        self.device = device or get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.bert = AutoModel.from_pretrained(name).to(self.device).eval()
        for p in self.bert.parameters():
            p.requires_grad_(False)
        self.dim = self.bert.config.hidden_size
        self._cache = {}

    @torch.no_grad()
    def encode(self, questions):
        missing = sorted({q for q in questions if q not in self._cache})
        if missing:
            tok = self.tokenizer(missing, padding=True, truncation=True, max_length=64,
                                 return_tensors="pt").to(self.device)
            h = self.bert(**tok).last_hidden_state
            if POOL == "cls":
                cls = h[:, 0].float().cpu()
            else:
                m = tok["attention_mask"].unsqueeze(-1).to(h.dtype)
                cls = ((h * m).sum(1) / m.sum(1)).float().cpu()
            for q, v in zip(missing, cls):
                self._cache[q] = v
        return torch.stack([self._cache[q] for q in questions])


def make_tensors(df, encoder, device):
    """DataFrame -> dict of tensors on `device` (cls, emotion, day, time, y)."""
    out = {
        "cls": encoder.encode(df["question"].tolist()).to(device),
        "emotion": torch.tensor(df["emotion"].map(EMOTION_IDX).values, dtype=torch.long, device=device),
        "day": torch.tensor(df["day"].values, dtype=torch.long, device=device),
        "time": torch.tensor(df["time"].values, dtype=torch.long, device=device),
        "y": torch.tensor(df["score"].values, dtype=torch.float32, device=device),
    }
    if "location" in df.columns:   # only data_ext has this column
        out["location"] = torch.tensor(df["location"].map(LOCATION_IDX).values, dtype=torch.long, device=device)
    cols = [c for c in EXTRA_ORDER if c in df.columns]
    if cols:
        out["extra"] = torch.stack([torch.tensor(df[c].map({v: i for i, v in enumerate(EXTRA_SPEC[c])}).values,
                                                 dtype=torch.long, device=device) for c in cols], dim=1)
    return out


def _head(in_dim, head=(256, 64), sigmoid=True):
    """Regression head: Linear(in,h1)-ReLU-Dropout(0.2)-Linear(h1,h2)-ReLU-Linear(h2,1)[-Sigmoid].
    sigmoid=False gives an unbounded logit (used by the DPO policy in budget.py)."""
    layers = [nn.Linear(in_dim, head[0]), nn.ReLU(), nn.Dropout(0.2),
              nn.Linear(head[0], head[1]), nn.ReLU(), nn.Linear(head[1], 1)]
    return nn.Sequential(*layers, nn.Sigmoid()) if sigmoid else nn.Sequential(*layers)


class _Scorer(nn.Module):
    """Shared plumbing. The (frozen) encoder is NOT a submodule: it has no trainable state and
    must stay out of state_dict / optimizer, so warm-starting copies only trainable weights."""

    def __init__(self, encoder=None):
        super().__init__()
        self._encoder = encoder

    @property
    def encoder(self):
        if self._encoder is None:
            self._encoder = FrozenBert()
        return self._encoder

    def context_features(self, emotion, day, time, location=None, extra=None):
        raise NotImplementedError

    def forward(self, cls, emotion, day, time, location=None, extra=None):
        x = torch.cat([cls, self.context_features(emotion, day, time, location, extra)], dim=1)
        return self.head(x).squeeze(1)

    @torch.no_grad()
    def predict(self, df):
        """Predictions (numpy) for a DataFrame with question/emotion/day/time columns."""
        was_training = self.training
        self.eval()
        dev = next(self.parameters()).device
        t = make_tensors(df.assign(score=0.0) if "score" not in df else df, self.encoder, dev)
        out = self(t["cls"], t["emotion"], t["day"], t["time"], t.get("location"), t.get("extra")).cpu().numpy()
        self.train(was_training)
        return out


class BertScorer(_Scorer):
    def __init__(self, encoder=None, emb=(16, 8, 8), head=(256, 64), sigmoid=True, drop_ctx=(), location_dim=0,
                 extra_sizes=(), extra_dim=8):
        """Defaults are the specified architecture; the arguments exist only for the hyper-parameter sweep."""
        super().__init__(encoder)
        self.emotion_emb = nn.Embedding(N_EMOTIONS, emb[0])
        self.day_emb = nn.Embedding(N_DAYS, emb[1])
        self.time_emb = nn.Embedding(N_TIMES, emb[2])
        self.location_dim = location_dim      # 0 = the specified 3-variable model; 8 = extra location embedding (data_ext)
        if location_dim:
            self.location_emb = nn.Embedding(N_LOCATIONS, location_dim)
        self.extra_sizes = tuple(extra_sizes)   # generic extra variables: one embedding of extra_dim dims each
        if self.extra_sizes:
            self.extra_emb = nn.ModuleList([nn.Embedding(n, extra_dim) for n in self.extra_sizes])
        self.context_mlp = nn.Sequential(nn.Linear(sum(emb) + location_dim + extra_dim * len(self.extra_sizes), 64), nn.ReLU(),
                                         nn.Linear(64, 64))
        self.head = _head(768 + 64, head, sigmoid)
        self.drop_ctx = tuple(drop_ctx)   # ablation: names in {"emotion", "day", "time"} whose embedding is replaced by zeros

    def context_features(self, emotion, day, time, location=None, extra=None):
        e, d, t = self.emotion_emb(emotion), self.day_emb(day), self.time_emb(time)
        if "emotion" in self.drop_ctx:
            e = torch.zeros_like(e)
        if "day" in self.drop_ctx:
            d = torch.zeros_like(d)
        if "time" in self.drop_ctx:
            t = torch.zeros_like(t)
        parts = [e, d, t] + ([self.location_emb(location)] if self.location_dim else [])
        parts += [emb(extra[:, i]) for i, emb in enumerate(self.extra_emb)] if self.extra_sizes else []
        return self.context_mlp(torch.cat(parts, dim=1))


class OneHotScorer(_Scorer):
    """Baseline: one-hot context (6+7+3) concatenated after CLS, no context MLP, same head."""

    def __init__(self, encoder=None, head=(256, 64), sigmoid=True, n_locations=0, extra_sizes=()):
        super().__init__(encoder)
        self.n_locations = n_locations        # 0 = the specified 3-variable model; 3 = extra one-hot location (data_ext)
        self.extra_sizes = tuple(extra_sizes)
        self.head = _head(768 + N_EMOTIONS + N_DAYS + N_TIMES + n_locations + sum(self.extra_sizes), head, sigmoid)

    def context_features(self, emotion, day, time, location=None, extra=None):
        oh = nn.functional.one_hot
        parts = [oh(emotion, N_EMOTIONS), oh(day, N_DAYS), oh(time, N_TIMES)]
        if self.n_locations:
            parts.append(oh(location, N_LOCATIONS))
        parts += [oh(extra[:, i], n) for i, n in enumerate(self.extra_sizes)]
        return torch.cat(parts, dim=1).float()


MODELS = {"bert": BertScorer, "onehot": OneHotScorer}
if V2:
    def _final_scorer(*a, **kw):
        from tact.core.heads import BilinearScorer          # lazy: heads.py subclasses BertScorer of this module
        return BilinearScorer(*a, **kw)
    _final_scorer.__name__ = "BilinearScorer"
    MODELS["bert"] = _final_scorer
