"""
Synthetic preference dataset generator for the adaptive clarification paper.

Ground-truth preference function (see Sec. 3.1 decomposition):
    r*(q, e, d, t) = N(topic(q), d, t) * T(e, d, t) + eps
where
    N = necessity of asking: low if a habit rule resolves the ambiguity
        for (topic, day[, time]); otherwise a topic-specific base value.
    T = receptiveness: emotion valence * availability(day, time).
    eps ~ Normal(0, LABEL_NOISE), final score clipped to [0, 1].

Outputs (in --outdir):
    dataset.csv          all samples with ground-truth components
    train.csv / val.csv / test.csv   fixed 70/15/15 split (seeded)
    heatmap_truth_<topic>.png        ground-truth day x emotion heatmaps
    config.json          the exact config used (for the paper appendix)
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Configuration (edit here; everything is dumped to config.json for the paper)
# ----------------------------------------------------------------------------

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]  # index 0..6
TIMES = ["morning", "afternoon", "evening"]               # index 0..2
EMOTIONS = ["neutral", "calm", "happy", "sad", "angry", "fearful"]

# Emotion valence -> willingness to be interrupted
VALENCE = {
    "happy": 0.95,
    "calm": 0.85,
    "neutral": 0.70,
    "sad": 0.45,
    "fearful": 0.40,
    "angry": 0.25,
}

USER_AVAIL = None   # set by apply_user (--user-seed): {"morning": c, "weekend_open": bool}


# Availability by (day, time): weekday mornings are busy, weekends relaxed.
def availability(day_idx: int, time_idx: int) -> float:
    if USER_AVAIL is not None:             # a simulated user: weekday mornings = c, weekends fully open or like weekdays
        if day_idx >= 5 and USER_AVAIL["weekend_open"]:
            return 1.0
        if time_idx == 0:
            return USER_AVAIL["morning"]
        return 0.80 if time_idx == 1 else 0.90
    weekend = day_idx >= 5
    if weekend:
        return 1.0
    if time_idx == 0:                      # weekday morning
        return 0.50 if day_idx == 0 else 0.60   # Monday morning worst
    if time_idx == 1:                      # weekday afternoon
        return 0.80
    return 0.90                            # weekday evening

# Kitchen topics (AmbiK-style) with several question phrasings each.
QUESTIONS = {
    "fruit": [
        "Which fruit should I use for the salad?",
        "Which type of fruit would you like me to wash?",
        "Which fruit should I cut up for you?",
    ],
    "plate": [
        "Which type of plate should I use to arrange the donuts?",
        "Which plate should I put the sandwich on?",
        "Which plate would you like for the pasta?",
    ],
    "knife": [
        "Which type of knife should I use to slice the pineapple?",
        "Which knife should I use for the bread?",
        "Which knife would you like me to use for the vegetables?",
    ],
    "cup": [
        "Which cup should I pour the juice into?",
        "Which type of cup would you like for your water?",
        "Which cup should I use for the smoothie?",
    ],
    "bowl": [
        "Which bowl should I put the cereal in?",
        "Which type of bowl should I use for the soup?",
        "Which bowl would you like for the rice?",
    ],
    "yogurt": [
        "Which type of yogurt should be used to spread onto the toasted bread?",
        "Which yogurt should I mix with the granola?",
        "Which flavor of yogurt would you like?",
    ],
    "bread": [
        "Which type of bread should I toast for breakfast?",
        "Which bread should I use for the sandwich?",
        "Which bread would you like me to warm up?",
    ],
    "eggs": [
        "Where should I place the eggs?",
        "How should I cook the eggs for you?",
        "How many eggs should I use for the omelette?",
    ],
    "coffee": [
        "Which type of coffee should I brew?",
        "Which coffee cup should I prepare?",
        "How strong should I make the coffee?",
    ],
    "tea": [
        "Which type of tea should I prepare?",
        "Which tea would you like with your snack?",
        "How long should I steep the tea?",
    ],
    "snack": [
        "Which snack should I set out for you?",
        "Which type of snack should I pack?",
        "Which snack would you like with the movie?",
    ],
    "cleaning": [
        "Which cloth should I use to wipe the counter?",
        "Which detergent should I use for the dishes?",
        "What should be dried first?",
    ],
}

# Topic-specific base necessity when NO habit resolves the ambiguity.
BASE_NECESSITY = {
    "fruit": 0.80, "plate": 0.75, "knife": 0.75, "cup": 0.70,
    "bowl": 0.70, "yogurt": 0.80, "bread": 0.80, "eggs": 0.85,
    "coffee": 0.80, "tea": 0.75, "snack": 0.75, "cleaning": 0.85,
}

HABIT_NECESSITY = 0.10  # necessity when a habit already resolves the ambiguity

# Habit rules: (topic, day_idx or None=any, time_idx or None=any, description)
# Keep the rule set small (4) so each cell gets enough samples to be learnable.
HABITS = [
    {"topic": "fruit",  "day": 5,    "time": None, "desc": "Saturdays the user always eats apples (fruit = apple)"},
    {"topic": "bread",  "day": 6,    "time": 0,    "desc": "Sunday mornings breakfast bread is always toast"},
    {"topic": "coffee", "day": None, "time": 0,    "desc": "Weekday mornings coffee is always black",
     "weekday_only": True},
    {"topic": "snack",  "day": 4,    "time": 2,    "desc": "Friday evenings the movie snack is always popcorn"},
]

LABEL_NOISE = 0.05
N_SAMPLES = 500
MIN_PER_HABIT_POS = 20   # habit-day samples per rule
MIN_PER_HABIT_NEG = 20   # same-topic control samples on non-habit days
SPLIT = (0.70, 0.15, 0.15)
SEED = 42

# ----------------------------------------------------------------------------


def habit_hit(topic: str, day: int, time: int):
    for h in HABITS:
        if h["topic"] != topic:
            continue
        if h.get("weekday_only") and day >= 5:
            continue
        if h.get("weekend_only") and day < 5:
            continue
        if h["day"] is not None and h["day"] != day:
            continue
        if h["time"] is not None and h["time"] != time:
            continue
        return h
    return None


def necessity(topic: str, day: int, time: int) -> float:
    return HABIT_NECESSITY if habit_hit(topic, day, time) else BASE_NECESSITY[topic]


def receptiveness(emotion: str, day: int, time: int) -> float:
    return VALENCE[emotion] * availability(day, time)


def true_score(topic: str, emotion: str, day: int, time: int) -> float:
    return necessity(topic, day, time) * receptiveness(emotion, day, time)


def sample_row(rng: random.Random, topic=None, day=None, time=None):
    topic = topic if topic is not None else rng.choice(list(QUESTIONS))
    question = rng.choice(QUESTIONS[topic])
    emotion = rng.choice(EMOTIONS)
    day = day if day is not None else rng.randrange(7)
    time = time if time is not None else rng.randrange(3)
    n = necessity(topic, day, time)
    t = receptiveness(emotion, day, time)
    clean = n * t
    noisy = float(np.clip(clean + rng.gauss(0, LABEL_NOISE), 0.0, 1.0))
    hit = habit_hit(topic, day, time)
    return {
        "question": question, "topic": topic, "emotion": emotion,
        "day": day, "day_name": DAYS[day], "time": time, "time_name": TIMES[time],
        "necessity": round(n, 3), "receptiveness": round(t, 3),
        "score_clean": round(clean, 3), "score": round(noisy, 3),
        "habit_hit": hit["desc"] if hit else "",
    }


ACT_TEXT = "(no question, act directly)"
LOCATIONS = ["kitchen", "living_room", "office"]   # --with-location: 4th context variable
OFFICE_FACTOR = 0.3                                 # in the office every clarification's receptiveness is multiplied by this
# --extra-vars: further categorical context variables, each with one multiplicative rule (cumulative, in this order).
#   location    office            : receptiveness x 0.3
#   companion   with_guests       : receptiveness x 0.5
#   activity    cooking           : necessity x 0.5 for the utensil topics (changes the ask AND the act value; a 2-variable rule)
#   noise_level noisy             : receptiveness x 0.7
EXTRA_VARS = {"location": (LOCATIONS, 2), "companion": (["alone", "with_guests"], 4),
              "activity": (["relaxing", "working", "cooking"], 5), "noise_level": (["quiet", "noisy"], 6)}
KITCHEN_TOPICS = {"plate", "knife", "cup", "bowl"}
TRUTH_FORMS = ("product", "additive", "saturating")

# --user-seed: a simulated user. Everything below is derived from the user seed alone.
HABIT_POOL = HABITS + [   # the 4 default habits + 6 new ones (other topic x day x time combinations)
    {"topic": "tea", "day": 6, "time": 1, "desc": "Sunday afternoons the tea is always green tea"},
    {"topic": "eggs", "day": 0, "time": 0, "desc": "Monday mornings the eggs are always scrambled"},
    {"topic": "yogurt", "day": None, "time": 0, "desc": "Weekend mornings the yogurt is always plain", "weekend_only": True},
    {"topic": "cup", "day": 2, "time": 2, "desc": "Wednesday evenings the cup is always the blue mug"},
    {"topic": "bowl", "day": 3, "time": None, "desc": "Thursdays the soup bowl is always the deep one"},
    {"topic": "cleaning", "day": 5, "time": 0, "desc": "Saturday mornings the counter is always wiped with the green cloth"},
]
NONMONOTONE_USERS = (9, 10)    # sad is the most welcome emotion (valence 0.85), angry stays low


def user_profile(user_seed):
    rng = random.Random(97 * user_seed + 13)
    val = {e: round(float(np.clip(VALENCE[e] + rng.uniform(-0.15, 0.15), 0.1, 1.0)), 3) for e in EMOTIONS}
    if user_seed in NONMONOTONE_USERS:
        val["sad"] = 0.85
        val["angry"] = min(val["angry"], 0.4)
    idx = sorted(rng.sample(range(len(HABIT_POOL)), 4))
    return {"user_seed": user_seed, "valence": val, "habit_pool_indices": idx, "habits": [HABIT_POOL[i] for i in idx],
            "weekday_morning_availability": round(rng.uniform(0.4, 0.7), 3), "weekend_open": bool(rng.random() < 0.5),
            "nonmonotone": user_seed in NONMONOTONE_USERS}


def apply_user(profile):
    """Replace the module-level valence table, habit rules and availability by the user's."""
    global USER_AVAIL
    VALENCE.clear()
    VALENCE.update(profile["valence"])
    HABITS[:] = profile["habits"]
    USER_AVAIL = {"morning": profile["weekday_morning_availability"], "weekend_open": profile["weekend_open"]}


def ask_value(n, t, form="product"):
    """True (clean) ask value from the necessity n and the receptiveness t = valence * availability.
    product (the default) n * t; additive 0.5 n + 0.5 t; saturating min(n, t). The act value stays 1 - n."""
    if form == "product":
        return n * t
    if form == "additive":
        return 0.5 * n + 0.5 * t
    if form == "saturating":
        return min(n, t)
    raise ValueError(form)
EXTRA_RULES = {"location": "office: receptiveness x 0.3", "companion": "with_guests: receptiveness x 0.5",
               "activity": "cooking: necessity x 0.5 for topics plate, knife, cup, bowl", "noise_level": "noisy: receptiveness x 0.7"}
UNSEEN_PHRASING = 3   # 1-based index into QUESTIONS[topic]; never used for train/val/test_seen in --with-act mode


def make_cfg(n, seed):
    return {
        "days": DAYS, "times": TIMES, "emotions": EMOTIONS,
        "valence": VALENCE, "base_necessity": BASE_NECESSITY,
        "habit_necessity": HABIT_NECESSITY, "habits": HABITS,
        "availability": {d: [availability(i, t) for t in range(3)] for i, d in enumerate(DAYS)},
        "label_noise": LABEL_NOISE, "n_samples": n,
        "min_per_habit_pos": MIN_PER_HABIT_POS,
        "min_per_habit_neg": MIN_PER_HABIT_NEG,
        "split": SPLIT, "seed": seed,
        "formula": "r* = necessity(topic, day, time) * valence(emotion) * availability(day, time) + N(0, label_noise), clipped to [0, 1]",
    }


def write_act_dataset(df, tr_idx, va_idx, te_idx, args):
    """--with-act: long format, one 'ask' row and one 'act' row per scenario, plus the phrasing split.

    Scenarios, split and ask labels are exactly those of the default mode (same rng stream). Extra randomness
    (re-drawing phrasing 3 -> {1, 2}, act-label noise) comes from a separate Random(seed + 1).
      act row : question = ACT_TEXT, score_clean = 1 - necessity, score = clip(1 - necessity + N(0, 0.05), 0, 1)
    train / val / test_seen use phrasings 1 and 2 only; test_unseen = the same scenarios as test_seen but with the
    question rewritten to phrasing 3, which appears in no training data."""
    rr = random.Random(args.seed + 1)
    scen = df.to_dict("records")
    extras = list(getattr(args, "extras", []))
    with_loc = bool(extras)
    for r in scen:  # seen version: phrasing in {1, 2}
        ph = QUESTIONS[r["topic"]].index(r["question"]) + 1
        if ph == UNSEEN_PHRASING:
            ph = rr.choice([1, 2])
        r["phrasing"], r["question"] = ph, QUESTIONS[r["topic"]][ph - 1]
        a_clean = 1.0 - r["necessity"]
        r["act_clean"] = round(a_clean, 3)
        r["act_score"] = round(float(np.clip(a_clean + rr.gauss(0, LABEL_NOISE), 0.0, 1.0)), 3)

    if with_loc:
        # Each extra variable: uniform per scenario from its own Random(seed + offset), drawn in scenario order.
        # Rules (see EXTRA_VARS): only scenarios touched by a rule are relabelled; their ask label is re-derived
        # (clean = necessity' * receptiveness', noise from Random(seed + 3), one draw per relabelled scenario) and, when the
        # necessity changes, the act label too (noise from Random(seed + 7)). Everything else (scenarios, phrasings, labels of
        # untouched scenarios) is identical to the plain --with-act data. With only `location` this reproduces data_ext.
        rngs = {v: random.Random(args.seed + EXTRA_VARS[v][1]) for v in extras}
        for r in scen:
            for v in extras:
                r[v] = rngs[v].choice(EXTRA_VARS[v][0])
        noise_rng, act_rng = random.Random(args.seed + 3), random.Random(args.seed + 7)
        for r in scen:
            rec_f, nec_f = 1.0, 1.0
            if r.get("location") == "office":
                rec_f *= OFFICE_FACTOR
            if r.get("companion") == "with_guests":
                rec_f *= 0.5
            if r.get("activity") == "cooking" and r["topic"] in KITCHEN_TOPICS:
                nec_f *= 0.5
            if r.get("noise_level") == "noisy":
                rec_f *= 0.7
            if rec_f != 1.0 or nec_f != 1.0:
                rec, nec = r["receptiveness"] * rec_f, r["necessity"] * nec_f
                clean = nec * rec
                r["necessity"], r["receptiveness"] = round(nec, 3), round(rec, 3)
                r["score_clean"] = round(clean, 3)
                r["score"] = round(float(np.clip(clean + noise_rng.gauss(0, LABEL_NOISE), 0.0, 1.0)), 3)
                if nec_f != 1.0:
                    a_clean = 1.0 - nec
                    r["act_clean"] = round(a_clean, 3)
                    r["act_score"] = round(float(np.clip(a_clean + act_rng.gauss(0, LABEL_NOISE), 0.0, 1.0)), 3)

    truth_form = getattr(args, "truth_form", "product")
    if truth_form != "product":
        # Misspecification test: another functional form of the ask value; everything else (scenarios, split, phrasings, act labels)
        # is identical to the plain --with-act data. The ask noise is redrawn (same N(0, label_noise)) from Random(seed + 8).
        nrng = random.Random(args.seed + 8)
        for r in scen:
            t = VALENCE[r["emotion"]] * availability(r["day"], r["time"])
            clean = ask_value(r["necessity"], t, truth_form)
            r["score_clean"] = round(clean, 3)
            r["score"] = round(float(np.clip(clean + nrng.gauss(0, LABEL_NOISE), 0.0, 1.0)), 3)

    if getattr(args, "label", "continuous") == "binary":
        # 0/1 feedback: the user only accepts or rejects. y ~ Bernoulli(r*) with the noise-free value r* of the action
        # (ask: n * t, act: 1 - n), no thresholding; one draw per scenario and action from Random(seed + 9).
        # score_clean (the evaluation truth) and everything else are unchanged.
        brng = random.Random(args.seed + getattr(args, "label_offset", 9))
        for r in scen:
            r["score"] = float(brng.random() < r["score_clean"])
            r["act_score"] = float(brng.random() < r["act_clean"])

    def long_rows(ids, unseen=False):
        out = []
        for i in ids:
            r = scen[i]
            ph = UNSEEN_PHRASING if unseen else r["phrasing"]
            base = {k: r[k] for k in ("id", "topic", "emotion", "day", "day_name", "time", "time_name",
                                      "necessity", "receptiveness", "habit_hit") + tuple(extras)}
            out.append({**base, "cand": "ask", "question": QUESTIONS[r["topic"]][ph - 1], "phrasing": ph,
                        "score_clean": r["score_clean"], "score": r["score"]})
            out.append({**base, "cand": "act", "question": ACT_TEXT, "phrasing": 0,
                        "score_clean": r["act_clean"], "score": r["act_score"]})
        cols = ["id", "cand", "question", "topic", "phrasing", "emotion", "day", "day_name", "time", "time_name"] + \
               list(extras) + ["necessity", "receptiveness", "score_clean", "score", "habit_hit"]
        return pd.DataFrame(out)[cols]

    files = {"dataset.csv": long_rows(range(len(scen))), "train.csv": long_rows(tr_idx),
             "val.csv": long_rows(va_idx), "test_seen.csv": long_rows(te_idx),
             "test_unseen.csv": long_rows(te_idx, unseen=True)}
    for name, d in files.items():
        d.to_csv(os.path.join(args.outdir, name), index=False)
    cfg = make_cfg(args.n, args.seed)
    if getattr(args, "label", "continuous") == "binary":
        cfg["label"] = "binary: score ~ Bernoulli(score_clean); score_clean is the noise-free truth"
    if getattr(args, "profile", None):
        cfg["user"] = args.profile
    if truth_form != "product":
        cfg.update({"truth_form": truth_form, "ask_formula": {"additive": "0.5 * necessity + 0.5 * receptiveness + noise",
                                                              "saturating": "min(necessity, receptiveness) + noise"}[truth_form]})
    if "location" in extras:
        cfg.update({"locations": LOCATIONS, "office_factor": OFFICE_FACTOR,
                    "location_rule": "ask receptiveness is multiplied by office_factor when location == office"})
    if extras != ["location"] and extras:
        cfg.update({"extra_vars": {v: EXTRA_VARS[v][0] for v in extras}, "extra_rules": {v: EXTRA_RULES[v] for v in extras}})
    cfg.update({"act_text": ACT_TEXT, "act_formula": "act: score_clean = 1 - necessity; score = clip(1 - necessity + N(0, label_noise), 0, 1)",
                "train_phrasings": [1, 2], "unseen_phrasing": UNSEEN_PHRASING,
                "note": "long format, rows of one scenario are adjacent (ask then act); test_unseen has the same scenarios "
                        "as test_seen with the question rewritten to the unseen phrasing"})
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    ask = pd.DataFrame(scen)
    oracle_act = (ask.act_clean > ask.score_clean)
    for v in extras:
        print(f"oracle chooses act by {v}:", oracle_act.groupby(ask[v]).mean().round(3).to_dict(),
              "| scenarios:", ask[v].value_counts().to_dict())
    print(f"scenarios: {len(scen)} (train {len(tr_idx)}, val {len(va_idx)}, test {len(te_idx)}); "
          f"oracle chooses act in {oracle_act.mean():.1%} of scenarios "
          f"({oracle_act[ask.habit_hit != ''].mean():.1%} of habit scenarios); phrasing counts (seen version): "
          f"{ask.phrasing.value_counts().sort_index().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="out")
    ap.add_argument("--n", type=int, default=N_SAMPLES)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--with-location", action="store_true",
                    help="(with --with-act) add a location context variable {kitchen, living_room, office}; office multiplies "
                         "the clarification receptiveness by 0.3")
    ap.add_argument("--extra-vars", default="",
                    help="(with --with-act) comma list of extra context variables, cumulative in this order: "
                         "location,companion,activity,noise_level (each adds one multiplicative rule)")
    ap.add_argument("--truth-form", choices=TRUTH_FORMS, default="product",
                    help="(with --with-act, no extra variables) functional form of the ask value: product (default, the "
                         "specified n * t), additive 0.5 n + 0.5 t, saturating min(n, t)")
    ap.add_argument("--user-seed", type=int, default=0,
                    help="(with --with-act) simulate user k >= 1: valence table (default +-0.15, users 9 and 10 non-monotone), 4 habit rules "
                         "from a pool of 10, weekday-morning availability and weekend openness, independent scenario draw; 0 = the default user")
    ap.add_argument("--label", choices=("continuous", "binary"), default="continuous",
                    help="(with --with-act) continuous noisy scores (default) or 0/1 feedback drawn as Bernoulli(r*)")
    ap.add_argument("--label-offset", type=int, default=9,
                    help="(with --label binary) the Bernoulli labels are drawn with Random(seed + offset); default 9 = the draw of data_bin*. "
                         "Other offsets give independent label draws for the SAME scenarios")
    ap.add_argument("--with-act", action="store_true",
                    help="long format with an 'act' candidate per scenario + unseen-phrasing test set")
    args = ap.parse_args()
    args.extras = [v for v in args.extra_vars.split(",") if v]
    if args.with_location and "location" not in args.extras:
        args.extras.insert(0, "location")
    if args.extras and not args.with_act:
        ap.error("--with-location / --extra-vars require --with-act")
    for v in args.extras:
        if v not in EXTRA_VARS:
            ap.error(f"unknown extra variable {v}")
    if args.label == "binary" and not args.with_act:
        ap.error("--label binary requires --with-act")
    if args.user_seed and (not args.with_act or args.extras or args.truth_form != "product"):
        ap.error("--user-seed requires --with-act and no extra variables / truth form")
    if args.truth_form != "product" and (not args.with_act or args.extras):
        ap.error("--truth-form other than product requires --with-act and no extra variables")
    os.makedirs(args.outdir, exist_ok=True)

    profile = None
    if args.user_seed:
        profile = user_profile(args.user_seed)
        apply_user(profile)
        args.seed = args.seed + 1000 * args.user_seed       # independent scenario draw per user
        print("user", args.user_seed, "| valence", profile["valence"], "| habits", [h["desc"] for h in profile["habits"]],
              "| weekday-morning availability", profile["weekday_morning_availability"], "| weekend open", profile["weekend_open"])
    args.profile = profile
    rng = random.Random(args.seed)
    rows = []

    # 1) Stratified habit coverage: positives on the habit cell,
    #    controls on the same topic off the habit cell.
    for h in HABITS:
        for _ in range(MIN_PER_HABIT_POS):
            if h.get("weekday_only"):
                d = rng.randrange(5)
            elif h.get("weekend_only"):
                d = 5 + rng.randrange(2)
            else:
                d = h["day"]
            t = h["time"] if h["time"] is not None else rng.randrange(3)
            rows.append(sample_row(rng, topic=h["topic"], day=d, time=t))
        for _ in range(MIN_PER_HABIT_NEG):
            while True:
                d, t = rng.randrange(7), rng.randrange(3)
                if habit_hit(h["topic"], d, t) is None:
                    break
            rows.append(sample_row(rng, topic=h["topic"], day=d, time=t))

    # 2) Fill the rest uniformly at random.
    while len(rows) < args.n:
        rows.append(sample_row(rng))
    rng.shuffle(rows)

    df = pd.DataFrame(rows)
    df.insert(0, "id", range(len(df)))
    df.to_csv(os.path.join(args.outdir, "dataset.csv"), index=False)

    # Fixed split.
    idx = list(range(len(df)))
    rng.shuffle(idx)
    n_tr = int(SPLIT[0] * len(df))
    n_va = int(SPLIT[1] * len(df))
    if args.with_act:
        write_act_dataset(df, idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:], args)
        return
    df.iloc[idx[:n_tr]].to_csv(os.path.join(args.outdir, "train.csv"), index=False)
    df.iloc[idx[n_tr:n_tr + n_va]].to_csv(os.path.join(args.outdir, "val.csv"), index=False)
    df.iloc[idx[n_tr + n_va:]].to_csv(os.path.join(args.outdir, "test.csv"), index=False)

    # Ground-truth heatmaps (day x emotion) for a habit topic and a neutral one.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for topic, time_idx in [("fruit", 1), ("plate", 1)]:
            grid = np.zeros((len(EMOTIONS), 7))
            for i, e in enumerate(EMOTIONS):
                for d in range(7):
                    grid[i, d] = true_score(topic, e, d, time_idx)
            fig, ax = plt.subplots(figsize=(7.5, 4))
            im = ax.imshow(grid, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks(range(7), DAYS)
            ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
            ax.set_title(f'Ground-truth r* for "{QUESTIONS[topic][0]}" ({TIMES[time_idx]})')
            for i in range(len(EMOTIONS)):
                for d in range(7):
                    ax.text(d, i, f"{grid[i, d]:.2f}", ha="center", va="center",
                            color="white" if grid[i, d] < 0.55 else "black", fontsize=8)
            fig.colorbar(im, ax=ax, label="r*")
            fig.tight_layout()
            fig.savefig(os.path.join(args.outdir, f"heatmap_truth_{topic}.png"), dpi=200)
            plt.close(fig)
    except ImportError:
        print("matplotlib not available; skipped heatmaps")

    # Dump config for the paper appendix.
    cfg = make_cfg(args.n, args.seed)
    with open(os.path.join(args.outdir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # Quick sanity report.
    print(f"total samples: {len(df)}")
    print(f"habit-hit samples: {(df['habit_hit'] != '').sum()}")
    print("\nmean score by emotion:")
    print(df.groupby("emotion")["score"].mean().round(3).sort_values())
    print("\nfruit topic, mean score Sat vs other days:")
    fr = df[df.topic == "fruit"]
    print(f"  Saturday: {fr[fr.day == 5]['score'].mean():.3f}  "
          f"(n={len(fr[fr.day == 5])})")
    print(f"  other:    {fr[fr.day != 5]['score'].mean():.3f}  "
          f"(n={len(fr[fr.day != 5])})")


if __name__ == "__main__":
    main()
