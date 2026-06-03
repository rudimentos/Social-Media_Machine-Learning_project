"""
app.py — Flask backend for Escalation Lab (home, dashboard, predictor).

Graceful degradation (page works before notebook exports exist):
  - spaCy + Cardiff sentiment: loaded lazily on first analysis call.
  - XGBoost model + scaler (thread-level probability + SHAP): optional. Without
    them, /api/score returns no probability and the panel says so.
  - predictor_threads.json / dashboard_data.json: optional; bundled fallbacks used.

Routes:
  GET  /                  home
  GET  /dashboard         dashboard
  GET  /predictor         -> redirect /predictor/start
  GET  /predictor/start   predictor, mode = from-the-start
  GET  /predictor/middle  predictor, mode = jump-in-the-middle
  GET  /api/root          random opener
  GET  /api/middle        random >=4-turn thread + continuation + outcome
  POST /api/tweet         {text, inbound} -> tweet-level features (+ profanity warning)
  POST /api/score         {turns, start_hour} -> model probability + SHAP (+ gating)
  GET  /api/dashboard     dashboard data
"""

import json
import os
import random
import re

from flask import Flask, jsonify, redirect, render_template, request

import features

ART = "artifacts"
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Optional model artifacts (thread-level probability + SHAP)
# ---------------------------------------------------------------------------
MODEL = SCALER = FEATURE_COLUMNS = None
_EXPLAINER = None
try:
    import joblib
    from xgboost import XGBClassifier
    if os.path.exists(os.path.join(ART, "model.json")):
        MODEL = XGBClassifier()
        MODEL.load_model(os.path.join(ART, "model.json"))
        SCALER = joblib.load(os.path.join(ART, "scaler.pkl"))
        with open(os.path.join(ART, "feature_columns.json")) as f:
            FEATURE_COLUMNS = json.load(f)
        print(f"[startup] model loaded ({len(FEATURE_COLUMNS)} features) — thread scoring enabled")
    else:
        print("[startup] no model artifacts — thread scoring disabled (run export_artifacts.py)")
except Exception as e:
    print(f"[startup] model load skipped: {e}")


def _get_explainer():
    global _EXPLAINER
    if _EXPLAINER is None and MODEL is not None:
        import shap
        _EXPLAINER = shap.TreeExplainer(MODEL)
    return _EXPLAINER


# ---------------------------------------------------------------------------
# Feature explanations (thread-level)
# ---------------------------------------------------------------------------
EXPLAIN = {
    "inb_turns": "Number of customer messages.",
    "out_turns": "Number of agent messages.",
    "neg_ratio": "Share of customer messages read as negative.",
    "inb_neg_mean": "Average negativity across customer messages.",
    "inb_neg_max": "Peak customer negativity.",
    "out_neg_mean": "Average negativity of agent messages.",
    "out_neg_median": "Median negativity of agent messages.",
    "inb_intent_any": "Customer used escalation-intent vocabulary (cancel, refund, legal…).",
    "inb_frustration_any": "Customer used strong negative vocabulary.",
    "inb_neg_emoji_any": "Customer used a negative emoji.",
    "inb_negation_ratio": "How often customer messages contain negation.",
    "inb_caps_mean": "Average share of all-caps words in customer messages.",
    "out_deflection_any": "Agent used deflection vocabulary (unable, policy, department…).",
    "out_bureaucratic_any": "Agent used bureaucratic/formulaic vocabulary (kindly, appreciate your patience…).",
    "out_formality_mean": "Average agent formality (Heylighen–Dewaele F-score).",
    "out_formal_vocab_any": "Agent used overtly formal vocabulary.",
    "out_passive_ratio": "Share of agent messages using passive voice.",
    "out_contraction_mean": "Average contractions per agent message (informality).",
    "out_we_ratio": "How often the agent uses 'we' rather than 'I'.",
    "out_abbrev_mean": "Average abbreviations per agent message.",
    "out_neu_mean": "Average neutrality of agent messages.",
    "inb_money_any": "Customer referenced a monetary amount.",
    "agent_sim": "Similarity between the agent's messages (templated-reply proxy).",
    "response_lag_mean": "Average gap between turns (minutes).",
    "response_lag_max": "Longest gap between turns (minutes).",
    "root_hour": "Hour of day the thread started.",
    "started_out_of_hours": "Whether the thread started outside 08:00–18:00.",
}

# ---------------------------------------------------------------------------
# Profanity (non-blocking warning only — see disclaimer)
# ---------------------------------------------------------------------------
PROFANITY = {"fuck", "fucking", "shit", "bastard", "asshole", "arsehole",
             "bollocks", "bitch", "dickhead", "wanker", "twat", "cunt"}
_PROF_RE = re.compile(r"\b(" + "|".join(PROFANITY) + r")\b", re.I)
_PROF_MSG = ("Profanity detected. The framework was built from dataset agent language, which "
             "contains no profanity, so abusive wording is outside what the model can assess — "
             "flagged here as a system limitation, not scored.")


def _profanity(text):
    return bool(_PROF_RE.search(text or ""))


# ---------------------------------------------------------------------------
# Thread data (real export or bundled sample)
# ---------------------------------------------------------------------------
_BUNDLED = {
    "roots": [
        {"opening_text": "I've been charged twice for my subscription this month. Can you sort this out?"},
        {"opening_text": "My delivery still hasn't arrived and it's been two weeks now."},
        {"opening_text": "The app keeps logging me out every few minutes."},
    ],
    "middles": [{
        "prompt_turns": [
            {"inbound": True, "text": "I've been charged twice for my subscription and I want a refund."},
            {"inbound": False, "text": "We're sorry to hear that. Could you confirm the email on your account?"},
            {"inbound": True, "text": "It's the same one I'm messaging from. This has happened twice now, it's not good enough."},
        ],
        "real_continuation": [
            {"inbound": False, "text": "Unfortunately refunds are handled by our billing department. Kindly contact them via the form."},
            {"inbound": True, "text": "Unbelievable. I'm cancelling."},
        ],
        "escalated": 1,
    }],
}
THREADS = _BUNDLED
_p = os.path.join(ART, "predictor_threads.json")
if os.path.exists(_p):
    try:
        with open(_p, encoding="utf-8") as f:
            THREADS = json.load(f)
        print(f"[startup] dataset threads: {len(THREADS.get('roots', []))} roots, "
              f"{len(THREADS.get('middles', []))} middles")
    except Exception as e:
        print(f"[startup] threads read failed, using bundled: {e}")
else:
    print("[startup] no predictor_threads.json — using bundled sample")


# ---------------------------------------------------------------------------
# Lazy sentiment model
# ---------------------------------------------------------------------------
_SENT = {}

def _ensure_sentiment():
    if _SENT:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("[lazy] loading Cardiff sentiment model...")
    name = "cardiffnlp/twitter-roberta-base-sentiment"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()
    _SENT.update(tok=tok, mdl=mdl, dev=dev, torch=torch)


def score_sentiment(text):
    if not (text or "").strip():
        return 0.0, 1.0, 0.0
    _ensure_sentiment()
    torch = _SENT["torch"]
    t = re.sub(r"http\S+", "http", text)
    t = re.sub(r"@\w+", "@user", t).strip()
    inp = _SENT["tok"](t, return_tensors="pt", truncation=True, max_length=128, padding=True).to(_SENT["dev"])
    with torch.no_grad():
        logits = _SENT["mdl"](**inp).logits
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[0]), float(probs[1]), float(probs[2])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/predictor")
def predictor_root():
    return render_template("predictor_home.html")


@app.route("/predictor/<submode>")
def predictor(submode):
    if submode not in ("start", "middle"):
        return redirect("/predictor/start")
    return render_template("predictor.html", submode=submode)


# ---------------------------------------------------------------------------
# Data API
# ---------------------------------------------------------------------------
@app.route("/api/root")
def api_root():
    return jsonify(random.choice(THREADS.get("roots") or _BUNDLED["roots"]))


@app.route("/api/middle")
def api_middle():
    return jsonify(random.choice(THREADS.get("middles") or _BUNDLED["middles"]))


# ---------------------------------------------------------------------------
# Tweet-level analysis
# ---------------------------------------------------------------------------
def _tweet_report(text, inbound):
    f = features.extract_tweet_features(text, inbound)
    neg, neu, pos = score_sentiment(text)
    if inbound:
        rows = [
            {"label": "Caps ratio", "value": f["caps_ratio"], "flag": f["caps_ratio"] > 0.3,
             "explain": "Share of all-caps words; a shouting proxy."},
            {"label": "Negation", "value": bool(f["has_negation"]), "flag": bool(f["has_negation"]),
             "explain": "Negation words present (not, never, can't…)."},
            {"label": "Intent terms", "value": bool(f["has_intent_term"]), "flag": bool(f["has_intent_term"]),
             "explain": "Escalation-intent vocabulary (cancel, refund, legal…)."},
            {"label": "Frustration terms", "value": bool(f["has_frustration_term"]), "flag": bool(f["has_frustration_term"]),
             "explain": "Strong negative vocabulary (unacceptable, useless…)."},
            {"label": "Negative emoji", "value": bool(f["has_negative_emoji"]), "flag": bool(f["has_negative_emoji"]),
             "explain": "An angry/negative emoji is present."},
            {"label": "Money reference", "value": bool(f["has_money_entity"]), "flag": bool(f["has_money_entity"]),
             "explain": "A monetary amount was detected."},
        ]
    else:
        rows = [
            {"label": "Deflection terms", "value": bool(f["has_deflection_term"]), "flag": bool(f["has_deflection_term"]),
             "explain": "Deflection vocabulary (unable, cannot, policy, department…)."},
            {"label": "Bureaucratic terms", "value": bool(f["has_bureaucratic_term"]), "flag": bool(f["has_bureaucratic_term"]),
             "explain": "Formulaic reassurance (kindly, appreciate your patience…)."},
            {"label": "Passive voice", "value": bool(f["has_passive"]), "flag": bool(f["has_passive"]),
             "explain": "A passive construction is present."},
            {"label": "Formality (F-score)", "value": f["formality_score"],
             "explain": "Heylighen–Dewaele formality score (higher = more formal)."},
            {"label": "Formal vocabulary", "value": bool(f["has_formal_vocab"]), "flag": bool(f["has_formal_vocab"]),
             "explain": "Overtly formal wording present."},
            {"label": "Contractions", "value": f["contraction_count"],
             "explain": "Count of contractions (informality marker)."},
            {"label": "Abbreviations", "value": f["abbreviation_count"],
             "explain": "Count of abbreviations (dm, asap…)."},
        ]
    out = {"sentiment": {"neg": round(neg, 4), "neu": round(neu, 4), "pos": round(pos, 4)},
           "features": rows}
    if _profanity(text):
        out["warning"] = _PROF_MSG
    return out


@app.route("/api/tweet", methods=["POST"])
def api_tweet():
    p = request.get_json(force=True)
    text = (p.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Empty message — nothing to analyse."}), 400
    return jsonify(_tweet_report(text, bool(p.get("inbound"))))


# ---------------------------------------------------------------------------
# Thread-level scoring: model probability + SHAP
# ---------------------------------------------------------------------------
@app.route("/api/score", methods=["POST"])
def api_score():
    p = request.get_json(force=True)
    turns = [t for t in p.get("turns", []) if (t.get("text") or "").strip()]
    start_hour = int(p.get("start_hour", 12))

    if not turns:
        return jsonify({"error": "No text to score yet."}), 400

    resp = {}
    if any(_profanity(t.get("text", "")) for t in turns):
        resp["warning"] = _PROF_MSG

    n_turns = len(turns)
    if n_turns < 2:
        resp.update(probability=None, note="Add at least two turns to evaluate the thread.")
        return jsonify(resp)

    if MODEL is None:
        resp.update(probability=None,
                    note="Trained model not loaded — run export_artifacts.py and place the "
                         "artifacts in the app's artifacts/ folder to enable thread scoring.")
        return jsonify(resp)

    import numpy as np
    import pandas as pd

    feat = features.aggregate_thread(turns, score_sentiment, start_hour=start_hour)
    row = pd.DataFrame([feat]).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    row_scaled = SCALER.transform(row)
    proba = float(MODEL.predict_proba(row_scaled)[:, 1][0])

    sv = np.asarray(_get_explainer().shap_values(row_scaled))
    contribs = sv[0] if sv.ndim == 2 else sv.reshape(-1)
    total = float(np.sum(np.abs(contribs))) or 1.0

    rows = sorted(
        [{"feature": col,
          "label": _pretty(col),
          "value": round(float(row.iloc[0][col]), 4),
          "contribution": round(float(c), 4),
          "share": round(100 * abs(float(c)) / total, 1),
          "explain": EXPLAIN.get(col, col)}
         for col, c in zip(FEATURE_COLUMNS, contribs)],
        key=lambda d: abs(d["contribution"]), reverse=True,
    )[:12]

    resp.update(probability=round(proba, 4), contributions=rows)
    if n_turns < 4:
        resp["gate_warning"] = ("Fewer than 4 turns: the model was trained on completed threads, so this "
                                "probability is extrapolation (thread features like agent similarity are not "
                                "yet meaningfully defined) — treat as indicative only.")
    return jsonify(resp)


_PRETTY = {
    "out_deflection_any": "Deflection language (agent)",
    "out_bureaucratic_any": "Bureaucratic language (agent)",
    "inb_frustration_any": "Frustration terms (customer)",
    "inb_intent_any": "Intent terms (customer)",
    "out_formal_vocab_any": "Formal vocabulary (agent)",
    "inb_neg_emoji_any": "Negative emoji (customer)",
    "agent_sim": "Agent template similarity",
    "response_lag_max": "Max response lag",
    "response_lag_mean": "Mean response lag",
    "out_passive_ratio": "Passive voice (agent)",
    "inb_negation_ratio": "Negation ratio (customer)",
    "inb_caps_mean": "Caps ratio (customer)",
    "out_formality_mean": "Formality F-score (agent)",
    "out_we_ratio": "We-not-I ratio (agent)",
    "out_contraction_mean": "Contractions (agent)",
    "out_abbrev_mean": "Abbreviations (agent)",
    "neg_ratio": "Customer negativity ratio",
    "inb_neg_mean": "Customer negativity (mean)",
    "inb_neg_max": "Customer negativity (peak)",
    "out_neg_mean": "Agent negativity (mean)",
    "out_neg_median": "Agent negativity (median)",
    "out_neu_mean": "Agent neutrality (mean)",
    "inb_money_any": "Money reference (customer)",
    "inb_turns": "Customer turn count",
    "out_turns": "Agent turn count",
    "root_hour": "Start hour",
    "started_out_of_hours": "Started out of hours",
}

def _pretty(col):
    return _PRETTY.get(col, col)


# ---------------------------------------------------------------------------
# Dashboard data
# ---------------------------------------------------------------------------
def _build_bundled_dashboard():
    import numpy as np
    rng = np.random.default_rng(42)
    p_non = rng.beta(2.0, 6.0, 3387); p_esc = rng.beta(6.0, 2.5, 579)
    y_true = [0] * 3387 + [1] * 579
    y_proba = [round(float(v), 4) for v in np.concatenate([p_non, p_esc])]
    return {
        "source": "bundled",
        "kpis": {"threads": 19828, "escalated": 2893, "esc_rate": 0.146, "auc": 0.8918},
        "binary_drivers": [
            {"feature": "out_deflection_any", "label": "Deflection language (agent)", "esc_rate": 0.315, "non_rate": 0.172, "cramers_v": 0.127, "p_value": 9.468e-72, "significant": True},
            {"feature": "out_bureaucratic_any", "label": "Bureaucratic language (agent)", "esc_rate": 0.207, "non_rate": 0.103, "cramers_v": 0.114, "p_value": 9.503e-58, "significant": True},
            {"feature": "inb_frustration_any", "label": "Frustration terms (customer)", "esc_rate": 0.112, "non_rate": 0.044, "cramers_v": 0.106, "p_value": 8.273e-51, "significant": True},
            {"feature": "inb_intent_any", "label": "Intent terms (customer)", "esc_rate": 0.118, "non_rate": 0.050, "cramers_v": 0.102, "p_value": 1.788e-46, "significant": True},
            {"feature": "out_formal_vocab_any", "label": "Formal vocabulary (agent)", "esc_rate": 0.047, "non_rate": 0.019, "cramers_v": 0.064, "p_value": 2.807e-19, "significant": True},
            {"feature": "inb_neg_emoji_any", "label": "Negative emoji (customer)", "esc_rate": 0.052, "non_rate": 0.041, "cramers_v": 0.018, "p_value": 9.375e-03, "significant": False},
        ],
        "continuous_drivers": [
            {"feature": "agent_sim", "label": "Agent template similarity", "esc_median": 0.141, "non_median": 0.0, "rank_biserial": -0.675, "p_value": 0.0, "significant": True},
            {"feature": "response_lag_max", "label": "Max response lag (min)", "esc_median": 118.033, "non_median": 41.817, "rank_biserial": -0.250, "p_value": 6.471e-103, "significant": True},
            {"feature": "inb_negation_ratio", "label": "Negation ratio (customer)", "esc_median": 0.333, "non_median": 0.0, "rank_biserial": -0.206, "p_value": 2.83e-91, "significant": True},
            {"feature": "inb_caps_mean", "label": "Caps ratio (customer)", "esc_median": 0.042, "non_median": 0.035, "rank_biserial": -0.107, "p_value": 1.614e-21, "significant": True},
            {"feature": "out_passive_ratio", "label": "Passive voice ratio (agent)", "esc_median": 0.0, "non_median": 0.0, "rank_biserial": -0.103, "p_value": 3.917e-45, "significant": True},
            {"feature": "response_lag_mean", "label": "Mean response lag (min)", "esc_median": 41.622, "non_median": 31.650, "rank_biserial": -0.076, "p_value": 6.703e-11, "significant": True},
            {"feature": "out_we_ratio", "label": "We-not-I ratio (agent)", "esc_median": 0.375, "non_median": 0.250, "rank_biserial": 0.033, "p_value": 1.972e-03, "significant": False},
            {"feature": "out_formality_mean", "label": "Formality F-score (agent)", "esc_median": 49.444, "non_median": 50.0, "rank_biserial": -0.013, "p_value": 0.275, "significant": False},
        ],
        "distributions": [
            {"feature": "agent_sim", "label": "Agent template similarity", "esc": {"q1": 0.0, "median": 0.141, "q3": 0.42, "mean": 0.22, "lo": 0.0, "hi": 0.78}, "non": {"q1": 0.0, "median": 0.0, "q3": 0.0, "mean": 0.05, "lo": 0.0, "hi": 0.31}},
            {"feature": "response_lag_max", "label": "Max response lag (min)", "esc": {"q1": 38.0, "median": 118.0, "q3": 360.0, "mean": 290.0, "lo": 0.0, "hi": 900.0}, "non": {"q1": 12.0, "median": 41.8, "q3": 140.0, "mean": 120.0, "lo": 0.0, "hi": 520.0}},
            {"feature": "inb_negation_ratio", "label": "Negation ratio (customer)", "esc": {"q1": 0.0, "median": 0.333, "q3": 0.6, "mean": 0.34, "lo": 0.0, "hi": 1.0}, "non": {"q1": 0.0, "median": 0.0, "q3": 0.33, "mean": 0.17, "lo": 0.0, "hi": 1.0}},
        ],
        "eval": {"y_true": y_true, "y_proba": y_proba, "auc": 0.8918},
        "shap": None,
    }


_BUNDLED_DASH = None
_DASH_FILE = os.path.join(ART, "dashboard_data.json")

@app.route("/api/dashboard")
def api_dashboard():
    global _BUNDLED_DASH
    if os.path.exists(_DASH_FILE):
        try:
            with open(_DASH_FILE) as f:
                return jsonify(json.load(f))
        except Exception as e:
            print(f"[dashboard] read failed, bundled: {e}")
    if _BUNDLED_DASH is None:
        _BUNDLED_DASH = _build_bundled_dashboard()
    return jsonify(_BUNDLED_DASH)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
