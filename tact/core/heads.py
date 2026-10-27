"""Interaction-capable small heads on the frozen BERT CLS (Experiment C, interaction_heads.py).

Same plumbing and call signature as model.BertScorer: forward(cls, emotion, day, time, location=None, extra=None), `sigmoid=False`
gives an unbounded logit (the DPO policy). BERT stays frozen (the cached CLS vector is the input f(q)); only these heads are trained.

FiLMScorer      g(c) = the Context MLP encoding (64); film = Linear(64,64)-ReLU-Linear(64,2*768) -> gamma(c), beta(c);
                x = (1 + gamma) * f(q) + beta; then the ORIGINAL regression layer (_head with input 768).
BilinearScorer  the original concatenation [f(q), g(c)] plus the 16 rank-one bilinear products (U^T f(q)) * (V^T g(c)) (U 768x16,
                V 64x16; their sum is the scalar f^T U V^T g, here kept as a 16-d vector); original regression layer on 832 + 16.
FactorScorer    control: N(f(q), x) and T(s, x), two sub-networks (s = emotion, x = all other context variables: day, time and the
                extra variables); ask = N * T, act = 1 - N. A row is an act row iff its CLS is the constant act-text CLS. With
                sigmoid=False (policy) the output is logit(ask value) / logit(act value).
"""
import torch
import torch.nn as nn

from tact.core.generate_dataset import ACT_TEXT
from tact.core.model import N_DAYS, N_EMOTIONS, N_TIMES, BertScorer, _head, _Scorer


class FiLMScorer(BertScorer):
    def __init__(self, encoder=None, head=(256, 64), sigmoid=True, extra_sizes=(), extra_dim=8, **kw):
        super().__init__(encoder, head=head, sigmoid=sigmoid, extra_sizes=extra_sizes, extra_dim=extra_dim, **kw)
        self.film = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2 * 768))
        self.head = _head(768, head, sigmoid)           # replaces the 832-input head of BertScorer

    def forward(self, cls, emotion, day, time, location=None, extra=None):
        g = self.context_features(emotion, day, time, location, extra)
        gamma, beta = self.film(g).chunk(2, dim=1)
        return self.head((1.0 + gamma) * cls + beta).squeeze(1)


class BilinearScorer(BertScorer):
    def __init__(self, encoder=None, head=(256, 64), sigmoid=True, extra_sizes=(), extra_dim=8, rank=16, **kw):
        super().__init__(encoder, head=head, sigmoid=sigmoid, extra_sizes=extra_sizes, extra_dim=extra_dim, **kw)
        d = encoder.dim if encoder is not None else 768    # representation size (768 for every encoder of Experiments A-C)
        self.U = nn.Linear(d, rank, bias=False)
        self.V = nn.Linear(64, rank, bias=False)
        self.head = _head(d + 64 + rank, head, sigmoid)

    def forward(self, cls, emotion, day, time, location=None, extra=None):
        g = self.context_features(emotion, day, time, location, extra)
        z = self.U(cls) * self.V(g)
        return self.head(torch.cat([cls, g, z], dim=1)).squeeze(1)


class FactorScorer(_Scorer):
    def __init__(self, encoder=None, head=(256, 64), sigmoid=True, extra_sizes=(), extra_dim=8, **kw):
        super().__init__(encoder)
        self.sigmoid = sigmoid
        self.extra_sizes = tuple(extra_sizes)
        k = len(self.extra_sizes)
        dx = 8 + 8 + extra_dim * k                                    # day, time, extras
        # N(f(q), x): the original head's shape on [CLS, x-embeddings]; T(s, x): small MLP on [emotion embedding, x-embeddings]
        self.n_day, self.n_time = nn.Embedding(N_DAYS, 8), nn.Embedding(N_TIMES, 8)
        self.t_day, self.t_time = nn.Embedding(N_DAYS, 8), nn.Embedding(N_TIMES, 8)
        self.t_emotion = nn.Embedding(N_EMOTIONS, 16)
        self.n_extra = nn.ModuleList([nn.Embedding(n, extra_dim) for n in self.extra_sizes])
        self.t_extra = nn.ModuleList([nn.Embedding(n, extra_dim) for n in self.extra_sizes])
        self.N_net = _head(768 + dx, head, sigmoid=False)
        self.T_net = nn.Sequential(nn.Linear(16 + dx, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
        act = encoder.encode([ACT_TEXT])[0] if encoder is not None else torch.zeros(768)
        self.register_buffer("act_cls", act.clone().float(), persistent=False)

    def forward(self, cls, emotion, day, time, location=None, extra=None):
        xn = [self.n_day(day), self.n_time(time)] + [e(extra[:, i]) for i, e in enumerate(self.n_extra)]
        xt = [self.t_emotion(emotion), self.t_day(day), self.t_time(time)] + [e(extra[:, i]) for i, e in enumerate(self.t_extra)]
        n = torch.sigmoid(self.N_net(torch.cat([cls] + xn, dim=1)).squeeze(1))
        t = torch.sigmoid(self.T_net(torch.cat(xt, dim=1)).squeeze(1))
        is_act = (cls == self.act_cls).all(dim=1)
        p = torch.where(is_act, 1.0 - n, n * t)
        if self.sigmoid:
            return p
        p = p.clamp(1e-6, 1 - 1e-6)
        return torch.log(p) - torch.log1p(-p)


ARCHS = {"orig": None, "film": FiLMScorer, "bilinear": BilinearScorer, "factor": FactorScorer}


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
