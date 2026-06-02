"""
features.py — single source of truth for feature extraction.

This module mirrors STEP 5 (tweet-level spaCy features) and STEP 7 (thread-level
aggregation) from the training pipeline. The Flask prototype imports it so that a
pasted conversation is turned into exactly the same 27-column feature vector the
XGBoost model was trained on.

PARITY REQUIREMENT (read this):
The model and StandardScaler were fit on values produced by your training code.
If any function here computes a feature differently, the scaler and model receive
shifted inputs and live predictions drift from reported test performance. The
robust fix is to make your training notebook IMPORT these same functions instead
of keeping a second copy. Three definitions below are marked CONFIRM because they
were not fully specified in the handoff — verify them against your training code.
"""

from collections import Counter

import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# spaCy pipeline (loaded once at import)
# ---------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm")

# ---------------------------------------------------------------------------
# Lexicons — copied verbatim from STEP 5
# ---------------------------------------------------------------------------
neg_words = {
    "not", "no", "never", "neither", "nor",
    "n't", "cant", "wont", "didnt", "dont",
    "isnt", "wasnt", "couldnt", "wouldnt", "shouldnt",
}

intent_terms = {
    "cancel", "cancelled", "cancelling", "cancellation",
    "sue", "suing", "lawsuit", "lawyer", "legal",
    "ombudsman", "ofcom", "regulator", "watchdog",
    "chargeback", "refund", "reimbursement",
    "switching", "leaving", "elsewhere",
}

frustration_terms = {
    "unacceptable", "ridiculous", "disgusting", "outrageous",
    "appalling", "pathetic", "terrible", "horrible",
    "useless", "incompetent", "shameful", "shocking",
    "furious", "livid", "angry", "disgusted",
    "scam", "fraud", "lied", "liar", "cheated",
}

deflection_terms = {
    "unable", "cannot", "policy", "procedure",
    "terms", "conditions", "unfortunately",
    "department", "team", "contact", "reach",
}

bureaucratic_terms = {
    "kindly", "advised", "patience",
    "appreciate", "bear", "investigate",
    "flag", "raise", "pass",
}

neg_emojis = {"😡", "😤", "🤬", "😠", "😒", "🙄", "😑", "😫", "😩", "😣", "😖"}

contractions = {
    "i'm", "we'll", "can't", "won't", "didn't", "we're",
    "you're", "it's", "that's", "there's", "don't", "isn't",
    "they're", "i'll", "i've", "you've", "we've", "couldn't",
    "wouldn't", "shouldn't", "aren't", "weren't", "hasn't",
}

abbreviations = {
    "dm", "asap", "fyi", "tbh", "imo", "lol", "omg",
    "btw", "ngl", "ty", "thx", "pls", "plz",
}

# --- CONFIRM #1: has_formal_vocab -----------------------------------------
# Not specified in the handoff. Best-guess lexicon below. Replace with your
# actual training definition or out_formal_vocab_any will diverge.
formal_vocab_terms = {
    "furthermore", "however", "subsequently", "regarding",
    "henceforth", "accordance", "hereby", "therein",
    "aforementioned", "notwithstanding", "whilst",
    "therefore", "thus", "hence", "wherein"
}

# --- CONFIRM #2: uses_we_not_i pronoun sets --------------------------------
we_tokens = {"we", "our", "us", "ourselves"}
i_tokens = {"i", "me", "my", "myself"}


def get_fscore(doc):
    """Heylighen & Dewaele (1999) F-score — copied from STEP 5."""
    toks = [t for t in doc if not t.is_space and not t.is_punct]
    if len(toks) == 0:
        return 50.0
    pos = Counter(t.pos_ for t in toks)
    n = len(toks)
    nouns = pos.get("NOUN", 0) + pos.get("PROPN", 0)
    adj = pos.get("ADJ", 0)
    prep = pos.get("ADP", 0)
    det = pos.get("DET", 0)
    pron = pos.get("PRON", 0)
    verb = pos.get("VERB", 0) + pos.get("AUX", 0)
    adv = pos.get("ADV", 0)
    intj = pos.get("INTJ", 0)
    f = ((nouns + adj + prep + det - pron - verb - adv - intj) / n * 100 + 100) / 2
    return round(f, 4)


def extract_tweet_features(text, is_inbound, doc=None):
    """Tweet-level features. Mirrors STEP 5 extract_features().

    Adds has_passive, has_formal_vocab, uses_we_not_i (the three columns the
    aggregation reads but that were appended to your tweet frame separately).
    """
    if doc is None:
        doc = nlp(text)
    toks = [t for t in doc if not t.is_space]
    alpha = [t for t in toks if t.is_alpha]
    upper_count = sum(1 for t in alpha if t.text.isupper())
    caps_ratio = upper_count / len(alpha) if len(alpha) > 0 else 0.0
    has_neg = int(any(t.lower_ in neg_words for t in toks))

    pos = Counter(t.pos_ for t in toks)
    ents = Counter(e.label_ for e in doc.ents)
    has_money = int(ents.get("MONEY", 0) > 0)
    has_org = int(ents.get("ORG", 0) > 0)

    toks_lower = [t.lower_ for t in doc if not t.is_space]

    # is_passive == has_passive: STEP 5 names it is_passive; STEP 7 reads
    # has_passive. Same dependency check. CONFIRM #3 if your training code
    # used a different passive definition.
    passive = int(any(t.dep_ == "auxpass" for t in doc))

    if is_inbound:
        has_intent = int(any(t.lower_ in intent_terms for t in toks))
        has_frustration = int(any(t.lower_ in frustration_terms for t in toks))
        has_neg_emoji = int(any(c in neg_emojis for c in text))
        has_deflection = 0
        has_bureaucratic = 0
        fscore = 0.0
        contraction_count = 0
        abbrev_count = 0
        has_formal_vocab = 0
        uses_we_not_i = 0
    else:
        has_intent = 0
        has_frustration = 0
        has_neg_emoji = 0
        has_deflection = int(any(t.lower_ in deflection_terms for t in toks))
        has_bureaucratic = int(any(t.lower_ in bureaucratic_terms for t in toks))
        fscore = get_fscore(doc)
        contraction_count = sum(1 for t in toks_lower if t in contractions)
        abbrev_count = sum(1 for t in toks_lower if t in abbreviations)
        has_formal_vocab = int(any(t in formal_vocab_terms for t in toks_lower))
        we_n = sum(1 for t in toks_lower if t in we_tokens)
        i_n = sum(1 for t in toks_lower if t in i_tokens)
        uses_we_not_i = int(we_n > i_n)

    return {
        "caps_ratio": round(caps_ratio, 4),
        "has_negation": has_neg,
        "has_money_entity": has_money,
        "has_org_entity": has_org,
        "has_intent_term": has_intent,
        "has_frustration_term": has_frustration,
        "has_negative_emoji": has_neg_emoji,
        "has_deflection_term": has_deflection,
        "has_bureaucratic_term": has_bureaucratic,
        "formality_score": fscore,
        "contraction_count": contraction_count,
        "abbreviation_count": abbrev_count,
        "has_passive": passive,
        "has_formal_vocab": has_formal_vocab,
        "uses_we_not_i": uses_we_not_i,
    }


def get_agent_sim(texts):
    """Mean pairwise cosine TF-IDF similarity of agent turns. STEP 7."""
    if len(texts) < 2:
        return 0.0
    try:
        tf = TfidfVectorizer().fit_transform(texts)
        sim = cosine_similarity(tf)
        n = sim.shape[0]
        vals = [sim[i][j] for i in range(n) for j in range(i + 1, n)]
        return round(sum(vals) / len(vals), 4)
    except Exception:
        return 0.0


def aggregate_thread(turns, score_fn, start_hour=12):
    """Build the thread-level feature dict from a list of turns.

    turns: list of dicts {inbound: bool, text: str, gap_min: float}
           gap_min is minutes since the previous turn (first turn -> 0).
    score_fn: callable(text) -> (neg, neu, pos) sentiment probabilities.
    start_hour: hour-of-day the conversation began (0-23), for root_hour.

    Returns a dict of the 27 model features plus diagnostic extras. The caller
    reindexes to the saved training column order before scoring.
    """
    inb_rows, out_rows = [], []
    inb_neg, out_neg, out_neu = [], [], []
    inb_texts, out_texts = [], []
    gaps = []

    for turn in turns:
        text = turn.get("text", "") or ""
        is_inbound = bool(turn.get("inbound"))
        gaps.append(float(turn.get("gap_min", 0) or 0))

        feats = extract_tweet_features(text, is_inbound)
        neg, neu, pos = score_fn(text)
        feats["sentiment_neg"] = round(float(neg), 4)
        feats["sentiment_neu"] = round(float(neu), 4)
        feats["sentiment_pos"] = round(float(pos), 4)

        if is_inbound:
            inb_rows.append(feats)
            inb_neg.append(feats["sentiment_neg"])
            inb_texts.append(text)
        else:
            out_rows.append(feats)
            out_neg.append(feats["sentiment_neg"])
            out_neu.append(feats["sentiment_neu"])
            out_texts.append(text)

    def mean(xs):
        return round(float(np.mean(xs)), 4) if xs else 0.0

    def median(xs):
        return round(float(np.median(xs)), 4) if xs else 0.0

    def any_flag(rows, key):
        return int(any(r[key] for r in rows)) if rows else 0

    d = {
        "inb_turns": len(inb_rows),
        "out_turns": len(out_rows),
        "neg_ratio": round(float(np.mean([n > 0.5 for n in inb_neg])), 4) if inb_neg else 0.0,
        "inb_neg_mean": mean(inb_neg),
        "inb_neg_max": round(float(max(inb_neg)), 4) if inb_neg else 0.0,
        "out_neg_mean": mean(out_neg),
        "out_neg_median": median(out_neg),
        "inb_intent_any": any_flag(inb_rows, "has_intent_term"),
        "inb_frustration_any": any_flag(inb_rows, "has_frustration_term"),
        "inb_neg_emoji_any": any_flag(inb_rows, "has_negative_emoji"),
        "inb_negation_ratio": mean([r["has_negation"] for r in inb_rows]),
        "inb_caps_mean": mean([r["caps_ratio"] for r in inb_rows]),
        "out_deflection_any": any_flag(out_rows, "has_deflection_term"),
        "out_bureaucratic_any": any_flag(out_rows, "has_bureaucratic_term"),
        "out_formality_mean": mean([r["formality_score"] for r in out_rows]),
        "out_formal_vocab_any": any_flag(out_rows, "has_formal_vocab"),
        "out_passive_ratio": mean([r["has_passive"] for r in out_rows]),
        "out_contraction_mean": mean([r["contraction_count"] for r in out_rows]),
        "out_we_ratio": mean([r["uses_we_not_i"] for r in out_rows]),
        "out_abbrev_mean": mean([r["abbreviation_count"] for r in out_rows]),
        "out_neu_mean": mean(out_neu),
        "inb_money_any": any_flag(inb_rows, "has_money_entity"),
        "agent_sim": get_agent_sim([t for t in out_texts if t.strip()]),
        # response_lag and root_hour are DEMO approximations: a pasted thread has
        # no real timestamps. gap_min and start_hour are user-supplied; if left at
        # defaults these features are 0 / midday, which biases toward non-escalation
        # (response_lag_max was a significant predictor). Surface them in the UI.
        "response_lag_mean": round(float(np.mean(gaps)), 4) if gaps else 0.0,
        "response_lag_max": round(float(max(gaps)), 4) if gaps else 0.0,
        "root_hour": int(start_hour),
        "started_out_of_hours": int(start_hour < 8 or start_hour >= 18),
    }
    return d