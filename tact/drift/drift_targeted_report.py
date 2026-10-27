"""Report of drift_targeted.py: results/drift_targeted.md and results/fig_drift_methods.pdf (+ .png)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text

from tact.drift.drift_targeted import DEFAULT, DRIFT_I, EVENTS, FINAL, HAB3, ROUNDS_CSV, SERIES, SWEEP, T_END, TARGET, cfg_name
from tact.core.model import RESULTS_DIR, ROOT

OUT_MD = os.path.join(ROOT, "results", "drift_targeted.md")
OUT_FIG = os.path.join(ROOT, "results", "fig_drift_methods")
CAP = T_END - DRIFT_I + 1
NAME = {"W": "(a) default: warm start on all data", "S": "(b) windowed: from scratch, last 3 rounds", "R2": "(c) recency-weighted replay, half-life 2 rounds",
        "R4": "(c) recency-weighted replay, half-life 4 rounds", "T": "targeted reset"}
COL = {"W": "#0072B2", "S": "#E69F00", "R2": "#009E73", "R4": "#56B4E9", "T": "#D55E00"}
DFLT = cfg_name("T", DEFAULT)


def ms(x, nd=3):
    x = np.asarray(x, float)
    return f"{x.mean():.{nd}f} ± {x.std(ddof=1):.{nd}f}" if len(x) > 1 else f"{x.mean():.{nd}f}"


def sub(df, run, m, cfg="-"):
    return df[(df.run == run) & (df.method == m) & (df.cfg == cfg)]


def reach(S, col="fs_new"):
    """per seed: (first, stays) interactions after the drift with regret < TARGET; None if never."""
    first, stays = {}, {}
    for seed, g in S.groupby("seed"):
        g = g[g.i > DRIFT_I].sort_values("i")
        below = (g[col] < TARGET).values
        first[seed] = int(g.i.values[below][0] - DRIFT_I) if below.any() else None
        st = None
        for i, b in zip(g.i.values[::-1], below[::-1]):
            if b:
                st = int(i - DRIFT_I)
            else:
                break
        stays[seed] = st
    return first, stays


def cens(d):
    return np.array([CAP if v is None else v for v in d.values()], float)


def reached(d):
    return f"{sum(v is not None for v in d.values())}/{len(d)}"


def write_report():
    S, R, F = pd.read_csv(SERIES), pd.read_csv(ROUNDS_CSV), pd.read_csv(FINAL)
    E = pd.read_csv(EVENTS) if os.path.exists(EVENTS) else pd.DataFrame()
    entries = [("W", "-"), ("S", "-"), ("R2", "-"), ("R4", "-")] + [("T", cfg_name("T", c)) for c in SWEEP]

    def label(m, cfg):
        if m != "T":
            return NAME[m]
        return "targeted reset " + ("(default τ = 0.3, k = 2, s = 0.8)" if cfg == DFLT else "(" + ", ".join(f"{p}" for p in cfg.replace("tau", "τ = ").replace("_k", ", k = ").replace("_s", ", s = ").split(", ")) + ")")

    # ---------------- per-entry statistics
    rows = {}
    for m, cfg in entries:
        sd, sc_ = sub(S, "drift", m, cfg), sub(S, "ctrl", m, cfg)
        fd, fc = sub(F, "drift", m, cfg).sort_values("seed"), sub(F, "ctrl", m, cfg).sort_values("seed")
        first, stays = reach(sd)
        Ld = sub(R, "drift", m, cfg)
        Ld = Ld[Ld["round"] == 15].sort_values("seed")
        L3 = Ld[[f"L_{h}" for h in HAB3]].mean(axis=1)
        ev = E[(E.run == "drift") & (E.method == m) & (E.cfg == cfg)] if len(E) else E
        evc = E[(E.run == "ctrl") & (E.method == m) & (E.cfg == cfg)] if len(E) else E
        rows[(m, cfg)] = dict(first=first, stays=stays, L3=L3, Lh={h: Ld[f"L_{h}"] for h in HAB3}, fd=fd, fc=fc, ev=ev, evc=evc,
                              grid_end=sd[sd.i == T_END].sort_values("seed").grid_new, fs_end=sd[sd.i == T_END].sort_values("seed").fs_new,
                              fs_curve=sd.pivot(index="seed", columns="i", values="fs_new"))
    w = rows[("W", "-")]
    w_mean, w_std = w["fc"].test_regret.mean(), w["fc"].test_regret.std(ddof=1)

    def crit(r):
        a = cens(r["first"]).mean()
        l = r["L3"].mean()
        nd = r["fc"].test_regret.mean()
        ok = (a <= 150, l >= 0.9, abs(nd - w_mean) <= w_std)
        return a, l, nd, ok

    out = ["# Drift-triggered targeted data invalidation (`drift_targeted.py`)\n",
           "Protocol of Appendix C (15 rounds × 50 interactions, dense Saturday-fruit evidence, the Saturday-fruit habit stops being true after interaction 150, validation scenarios follow the drift, final architecture "
           "with the threshold rule, continuous labels), 5 seeds (0–4), mean ± std over seeds. `ctrl` = the same stream with NO drift. **Targeted reset** (new): surprise on ACT decisions only "
           "(|observed act label − r̂(act, c)| > τ), trigger when ≥ k surprises of the current round fall into the same cell (cell = question cluster × day × time; cluster = cosine ≥ s under the frozen NLI-BERT encoder; "
           "the oracle topic is not used), on trigger the cell's older stored interactions (older than the first surprise of the group) are deleted and the reward model is retrained from scratch on the remaining data and deployed immediately; "
           "at every round end the model is updated by warm start as in (a). Full definitions and the interpretations I had to make are in the header of `drift_targeted.py`. Defaults τ = 0.3, k = 2, s = 0.8.\n",
           "**Criterion fixed in advance** (applied to the seed means): adaptation ≤ 150 interactions (first interaction after the drift with regret on the 18 drifted cells < 0.05; seeds that never get there count as 601) "
           "AND retention L ≥ 0.9 (mean of the three unchanged habits at round 15) AND the final test regret of the NO-DRIFT run within 1 std of the default (a): "
           f"|mean − {w_mean:.4f}| ≤ {w_std:.4f}, the mean and std of (a) in the same no-drift run (the 0.029 quoted in the request is the semi-online N = 350 test regret; the default's value after 750 interactions of the no-drift stream is {w_mean:.3f} ± {w_std:.3f}).\n"]

    # ---------------- main table
    out += ["## Main results\n",
            "Adaptation: interactions after the drift until the regret on the 18 drifted cells (new truth) is below 0.05, mean over seeds (never = 601), seeds that reach it, and the first interaction from which it stays below until the end. "
            "Test regret = 75 test scenarios, truth of the moment (Saturday-fruit scenarios follow the new rule in the drift run); 'grid regret' = 1512-cell grid, new truth. Deleted rows = replay-buffer scenarios deleted (each scenario = one ask and one act row); "
            "precision = share of them that belongs to the drifted habit (topic fruit, Saturday — evaluation only).\n",
            "| method | adaptation (first) | seeds reaching | stays below from (mean) | retention L (3 other habits) | NO-DRIFT: triggers (mean, total) | NO-DRIFT: final test regret | drift: final test regret | drift: final grid regret | drift: triggers (true / all) | deleted scenarios (mean) | precision | criterion |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m, cfg in entries:
        r = rows[(m, cfg)]
        a, l, nd, ok = crit(r)
        ev, evc = r["ev"], r["evc"]
        n_tr_c = r["fc"].n_triggers.values
        if m == "T":
            tot = ev.deleted.sum() if len(ev) else 0
            prec = f"{ev.deleted_drifted_habit.sum() / tot:.2f} ({int(ev.deleted_drifted_habit.sum())}/{int(tot)})" if tot else "n/a (nothing deleted)"
            trg = f"{int(ev.true_drift.sum()) if len(ev) else 0} / {len(ev)}"
            nd_tr = f"{n_tr_c.mean():.1f}, {int(n_tr_c.sum())}"
            dele = f"{r['fd'].n_deleted.mean():.1f}"
            verdict = ("**PASS**" if all(ok) else "fails: " + ", ".join(n for n, g in zip(("adaptation", "retention", "no-drift regret"), ok) if not g))
        else:
            prec, trg, nd_tr, dele = "–", "–", "0", "0" if m != "S" else "(window)"
            verdict = ("**PASS**" if all(ok) else "fails: " + ", ".join(n for n, g in zip(("adaptation", "retention", "no-drift regret"), ok) if not g))
        out.append(f"| {label(m, cfg)} | {a:.0f} | {reached(r['first'])} | {cens(r['stays']).mean():.0f} | {ms(r['L3'], 2)} | {nd_tr} | {ms(r['fc'].test_regret)} | {ms(r['fd'].test_regret)} | {ms(r['grid_end'])} | {trg} | {dele} | {prec} | {verdict} |")
    # ---------------- retention per habit
    out += ["\n## Retention per habit at round 15 (learning degree L)\n", "| method | bread, Sunday mornings | coffee, weekday mornings | snack, Friday evenings | mean |", "|---|---|---|---|---|"]
    for m, cfg in entries[:5] + [("T", cfg_name("T", c)) for c in SWEEP[1:]]:
        r = rows[(m, cfg)]
        out.append(f"| {label(m, cfg)} | " + " | ".join(ms(r["Lh"][h], 2) for h in HAB3) + f" | {ms(r['L3'], 2)} |")
    # ---------------- regret curve table
    ks = (50, 100, 150, 200, 300, 450, 600)
    out += ["\n## Regret on the 18 drifted cells (new truth) after k interactions of the drifted rule\n", "| k | " + " | ".join(NAME[m] if m != "T" else "targeted reset (default)" for m in ("W", "S", "R2", "R4", "T")) + " |", "|---|" + "---|" * 5]
    for k in ks:
        cells = []
        for m, cfg in (("W", "-"), ("S", "-"), ("R2", "-"), ("R4", "-"), ("T", DFLT)):
            c = rows[(m, cfg)]["fs_curve"]
            i = min(DRIFT_I + k, T_END)
            cells.append(ms(c[i]))
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    # ---------------- paired comparison with default
    out += ["\n## Targeted reset (default) against the default warm start, paired by seed (Δ = targeted reset − default)\n", "| quantity | targeted reset | default (a) | Δ (mean ± SE) | seeds where targeted reset is better |", "|---|---|---|---|---|"]
    t = rows[("T", DFLT)]
    pairs = [("adaptation, first (interactions; lower is better)", cens(t["first"]), cens(w["first"]), "lower"), ("retention L (higher is better)", t["L3"].values, w["L3"].values, "higher"),
             ("NO-DRIFT final test regret (lower)", t["fc"].test_regret.values, w["fc"].test_regret.values, "lower"), ("drift: final test regret (lower)", t["fd"].test_regret.values, w["fd"].test_regret.values, "lower"),
             ("drift: final regret on the drifted cells (lower)", t["fs_end"].values, w["fs_end"].values, "lower")]
    for nm, a, b, bt in pairs:
        d = a - b
        k = int((d < 0).sum() if bt == "lower" else (d > 0).sum())
        nd = 1 if "interactions" in nm else 3
        out.append(f"| {nm} | {a.mean():.{nd}f} | {b.mean():.{nd}f} | {d.mean():+.{nd}f} ± {d.std(ddof=1) / np.sqrt(len(d)):.{nd}f} | {k}/{len(d)} |")
    # ---------------- cells deleted
    if len(E):
        out += ["\n## Which cells were deleted (default configuration, summed over the 5 seeds)\n",
                "Cell of the triggering interaction (topic shown for evaluation only); rows = deleted replay-buffer scenarios of that cell's cluster. 'drifted habit' = fruit / Saturday; 'other habit' = a cell of one of the three unchanged habits.\n"]
        for run, ttl in (("drift", "Drift run"), ("ctrl", "NO-DRIFT run (all triggers are false)")):
            ev = E[(E.run == run) & (E.method == "T") & (E.cfg == DFLT)]
            out += [f"\n### {ttl}\n"]
            if not len(ev):
                out.append("No trigger fired.")
                continue
            g = ev.groupby(["topic", "day", "time"]).agg(triggers=("i", "size"), seeds=("seed", "nunique"), deleted=("deleted", "sum"), drifted=("deleted_drifted_habit", "sum"), other_habit=("deleted_other_habit", "sum")).reset_index().sort_values("triggers", ascending=False)
            DAYS, TIMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], ["morning", "afternoon", "evening"]
            out += ["| triggering cell | triggers | seeds | scenarios deleted | of them drifted habit | of them other habit |", "|---|---|---|---|---|---|"]
            out += [f"| {r.topic}, {DAYS[r.day]} {TIMES[r.time]} | {r.triggers} | {r.seeds} | {r.deleted} | {r.drifted} | {r.other_habit} |" for r in g.itertuples()]
            first_tr = ev[ev.true_drift].groupby("seed").i.min() if ev.true_drift.any() else pd.Series(dtype=float)
            if run == "drift":
                out.append(f"\nFirst trigger on the drifted habit (interaction, per seed): {', '.join(f'seed {s_}: {int(v)}' for s_, v in first_tr.items()) or 'none'} (drift starts after interaction {DRIFT_I}). "
                           f"Triggers before the drift (false): {int((ev.i <= DRIFT_I).sum())}; after the drift on cells that are not the drifted one: {int(((ev.i > DRIFT_I) & ~ev.true_drift).sum())}.")
    # ---------------- sweeps
    out += ["\n## Sensitivity: one parameter at a time (default τ = 0.3, k = 2, s = 0.8)\n", "| τ | k | s | adaptation (first) | seeds reaching | retention L | NO-DRIFT triggers (mean) | NO-DRIFT test regret | drift: final test regret | drift: triggers | precision | criterion |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in SWEEP:
        cfg = cfg_name("T", c)
        r = rows[("T", cfg)]
        a, l, nd, ok = crit(r)
        ev = r["ev"]
        tot = ev.deleted.sum() if len(ev) else 0
        prec = f"{ev.deleted_drifted_habit.sum() / tot:.2f}" if tot else "n/a"
        out.append(f"| {c['tau']} | {c['k']} | {c['s']} | {a:.0f} | {reached(r['first'])} | {ms(r['L3'], 2)} | {r['fc'].n_triggers.mean():.1f} | {ms(r['fc'].test_regret)} | {ms(r['fd'].test_regret)} | {len(ev)} | {prec} | "
                   + ("**PASS**" if all(ok) else "fails: " + ", ".join(n for n, g in zip(("adaptation", "retention", "no-drift regret"), ok) if not g)) + " |")
    # ---------------- summary
    passed = [label(m, cfg) for m, cfg in entries if all(crit(rows[(m, cfg)])[3])]
    out += ["\n## Criterion outcome\n", ("Configurations meeting all three conditions: " + "; ".join(passed) + ".") if passed else "**No configuration meets all three conditions.**"]
    one = [label(m, cfg) for m, cfg in entries if (cens(rows[(m, cfg)]["first"]).mean() <= 150 and rows[(m, cfg)]["L3"].mean() >= 0.9 and rows[(m, cfg)]["fc"].test_regret.mean() <= w_mean + w_std)]
    out.append("\nAlternative reading, stated after seeing the results and NOT replacing the outcome above: the third condition as a one-sided 'no worse than the default' "
               f"(NO-DRIFT test regret ≤ default mean + 1 std = {w_mean + w_std:.4f}). Under it the literal two-sided rule penalises the targeted reset for being *better* than the default in the no-drift run "
               "(it retrains from scratch many times there, see the false-trigger counts). Configurations meeting adaptation ≤ 150, L ≥ 0.9 and the one-sided condition: " + ("; ".join(one) if one else "none") + ". "
               "Caveats that apply to every targeted-reset row: (i) 'adaptation' is the FIRST crossing of 0.05; the regret on the drifted cells fluctuates and stays below only from the 'stays below from' column on; "
               "(ii) the number of triggers in the NO-DRIFT run (each one a full retraining from scratch and a deletion) is large; (iii) in the NO-DRIFT run 'drifted habit' in the cell table is just the Saturday-fruit habit, "
               "which does not drift there, so those deletions are false.")
    best_a = min(entries, key=lambda e: cens(rows[e]["first"]).mean())
    out.append(f"Fastest adaptation: {label(*best_a)} ({cens(rows[best_a]['first']).mean():.0f} interactions, {reached(rows[best_a]['first'])} seeds reached).")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(out) + "\n")
    print("\n".join(out))
    figure(rows)


def figure(rows):
    W = 84 / 25.4
    plt.style.use("default")      # plots.py (imported through drift_trigger) sets a grey, gridded style
    plt.rcParams.update({"font.family": ["Liberation Sans", "DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7, "axes.linewidth": 0.6,
                         "lines.linewidth": 1.2, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "custom", "mathtext.rm": "Liberation Sans"})
    fig, ax = plt.subplots(figsize=(W, 2.5), constrained_layout=True)
    for m, cfg, lab in (("W", "-", "(a) warm start, all data"), ("S", "-", "(b) windowed, last 3 rounds"), ("R2", "-", "(c) recency-weighted, half-life 2"),
                        ("R4", "-", "(c) recency-weighted, half-life 4"), ("T", DFLT, "targeted reset")):
        c = rows[(m, cfg)]["fs_curve"]
        c = c[[i for i in c.columns if i >= 100]]
        mu, sd = c.mean(), c.std(ddof=1)
        ax.plot(c.columns, mu, color=COL[m], lw=1.8 if m == "T" else 1.1, label=lab)
        ax.fill_between(c.columns, mu - sd, mu + sd, color=COL[m], alpha=0.12, lw=0)
    ax.axvline(DRIFT_I, color="#555555", ls=":", lw=0.8)
    ax.axhline(TARGET, color="k", ls="--", lw=0.8)
    ax.text(104, TARGET + 0.006, "target 0.05", ha="left", va="bottom", fontsize=7)
    ax.text(DRIFT_I + 8, ax.get_ylim()[1] * 0.97, "drift", fontsize=7, va="top")
    ax.set_xlabel("interactions")
    ax.set_ylabel("regret on drifted cells")
    ax.set_xlim(100, T_END)
    ax.legend(frameon=False, loc="upper right", handlelength=1.5, labelspacing=0.25, borderaxespad=0.1)
    sizes = [t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip()]
    assert min(sizes) >= 7.0
    fig.savefig(OUT_FIG + ".pdf")
    fig.savefig(OUT_FIG + ".png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    write_report()
