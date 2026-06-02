"""
app.py — Flask backend for the escalation predictor.

Design goal: the server starts and the predictor page is FUNCTIONAL even before
the notebook export scripts have been run. Heavy / optional pieces degrade:

  - spaCy + Cardiff sentiment model: required for scoring. Loaded LAZILY on the
    first /api/score call (so startup is instant and the download only happens
    when you actually score something).
  - XGBoost model + scaler (Layer B): OPTIONAL. If artifacts/ is missing them,
    scoring still returns the transparent Layer-A read; Layer B stays quiet.
  - predictor_threads.json (dataset roots / middle threads): OPTIONAL. If absent,
    /api/root and /api/middle serve a small bundled sample so the page works out
    of the box. Run export_threads.py in your notebook to use the real dataset.

Routes:
  GET  /            -> redirect to /predictor
  GET  /predictor   -> the predictor page
  GET  /dashboard   -> placeholder until the dashboard page is built
  GET  /api/root    -> a random thread opener (Mode 1)
  GET  /api/middle  -> a random >=4-turn thread + continuation + outcome (Mode 2)
  POST /api/score   -> {turns, start_hour} -> Layer-A read (+ Layer B if >=4 turns)
"""

import json
import os
import random
import re

from flask import Flask, jsonify, redirect, render_template, request

import features
import scoring

ART = "artifacts"
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Optional artifacts loaded at startup (None if absent)
# ---------------------------------------------------------------------------
MODEL = None
SCALER = None
FEATURE_COLUMNS = None

try:
    import joblib
    from xgboost import XGBClassifier
    if os.path.exists(os.path.join(ART, "model.json")):
        MODEL = XGBClassifier()
        MODEL.load_model(os.path.join(ART, "model.json"))
        SCALER = joblib.load(os.path.join(ART, "scaler.pkl"))
        with open(os.path.join(ART, "feature_columns.json")) as f:
            FEATURE_COLUMNS = json.load(f)
        print(f"[startup] model loaded ({len(FEATURE_COLUMNS)} features) — Layer B enabled")
    else:
        print("[startup] no model artifacts — Layer B disabled (run export_artifacts.py)")
except Exception as e:
    print(f"[startup] model load skipped: {e}")

# ---------------------------------------------------------------------------
# Thread data: real export if present, else a small bundled sample
# ---------------------------------------------------------------------------
_BUNDLED = {
    "roots": [
        {"opening_text": "I've been charged twice for my subscription this month. Can you sort this out?"},
        {"opening_text": "My delivery still hasn't arrived and it's been two weeks now."},
        {"opening_text": "The app keeps logging me out every few minutes, it's driving me mad."},
        {"opening_text": "Why has my account been suspended with no warning?"},
    ],
    "middles": [
        {
            "prompt_turns": [
                {"inbound": True,  "text": "I've been charged twice for my subscription and I want a refund."},
                {"inbound": False, "text": "We're sorry to hear that. Could you confirm the email on your account?"},
                {"inbound": True,  "text": "It's the same one I'm messaging from. This has happened twice now, it's not good enough."},
            ],
            "real_continuation": [
                {"inbound": False, "text": "Unfortunately refunds are handled by our billing department. Kindly contact them via the form and they'll be in touch."},
                {"inbound": True,  "text": "Unbelievable. I'm cancelling."},
            ],
            "escalated": 1,
        },
        {
            "prompt_turns": [
                {"inbound": True,  "text": "My order arrived damaged. What can you do?"},
                {"inbound": False, "text": "Oh no, sorry about that! I can get a replacement shipped today, no charge."},
                {"inbound": True,  "text": "Oh great, thank you. How long will it take?"},
            ],
            "real_continuation": [
                {"inbound": False, "text": "It'll be with you in 2 working days and I've added tracking to your account now."},
                {"inbound": True,  "text": "Perfect, thanks for sorting it so fast!"},
            ],
            "escalated": 0,
        },
    ],
}

THREADS = _BUNDLED
USING_BUNDLED = True
_p = os.path.join(ART, "predictor_threads.json")
if os.path.exists(_p):
    try:
        with open(_p, encoding="utf-8") as f:
            THREADS = json.load(f)
        USING_BUNDLED = False
        print(f"[startup] dataset threads loaded: "
              f"{len(THREADS.get('roots', []))} roots, {len(THREADS.get('middles', []))} middles")
    except Exception as e:
        print(f"[startup] failed to read predictor_threads.json, using bundled sample: {e}")
else:
    print("[startup] no predictor_threads.json — using bundled sample (run export_threads.py)")

# ---------------------------------------------------------------------------
# Lazy sentiment model
# ---------------------------------------------------------------------------
_SENT = {}

def _ensure_sentiment():
    if _SENT:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    print("[lazy] loading Cardiff sentiment model (first scoring call)...")
    name = "cardiffnlp/twitter-roberta-base-sentiment"
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForSequenceClassification.from_pretrained(name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev).eval()
    _SENT.update(tok=tok, mdl=mdl, dev=dev, torch=torch)


def score_sentiment(text):
    if not text or not text.strip():
        return 0.0, 1.0, 0.0
    _ensure_sentiment()
    torch = _SENT["torch"]
    t = re.sub(r"http\S+", "http", text)
    t = re.sub(r"@\w+", "@user", t).strip()
    inputs = _SENT["tok"](t, return_tensors="pt", truncation=True,
                          max_length=128, padding=True).to(_SENT["dev"])
    with torch.no_grad():
        logits = _SENT["mdl"](**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
    return float(probs[0]), float(probs[1]), float(probs[2])


def _model_proba(feat):
    """Layer-B probability, or None if the model isn't loaded."""
    if MODEL is None:
        return None
    import pandas as pd
    row = pd.DataFrame([feat]).reindex(columns=FEATURE_COLUMNS, fill_value=0.0)
    row_scaled = SCALER.transform(row)
    return float(MODEL.predict_proba(row_scaled)[:, 1][0])


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return redirect("/predictor")


@app.route("/predictor")
def predictor():
    return render_template("predictor.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ---------------------------------------------------------------------------
# Dashboard data: real export if present, else a bundled set built from the
# project's actual step-9 / step-12 numbers (only the per-point test predictions
# are synthesised, since those raw arrays aren't hardcodable).
# ---------------------------------------------------------------------------
def _build_bundled_dashboard():
    import numpy as np
    rng = np.random.default_rng(42)
    n_non, n_esc = 3387, 579           # test-set class sizes (step 12)
    p_non = rng.beta(2.0, 6.0, n_non)  # resolved -> lower scores
    p_esc = rng.beta(6.0, 2.5, n_esc)  # escalated -> higher scores
    y_true = [0] * n_non + [1] * n_esc
    y_proba = [round(float(v), 4) for v in np.concatenate([p_non, p_esc])]

    return {
        "source": "bundled",
        "kpis": {"threads": 19828, "escalated": 2893, "esc_rate": 0.146, "auc": 0.8918},
        "binary_drivers": [
            {"feature": "out_deflection_any", "label": "Deflection language (agent)",
             "esc_rate": 0.315, "non_rate": 0.172, "cramers_v": 0.127, "p_value": 9.468e-72, "significant": True},
            {"feature": "out_bureaucratic_any", "label": "Bureaucratic language (agent)",
             "esc_rate": 0.207, "non_rate": 0.103, "cramers_v": 0.114, "p_value": 9.503e-58, "significant": True},
            {"feature": "inb_frustration_any", "label": "Frustration terms (customer)",
             "esc_rate": 0.112, "non_rate": 0.044, "cramers_v": 0.106, "p_value": 8.273e-51, "significant": True},
            {"feature": "inb_intent_any", "label": "Intent terms (customer)",
             "esc_rate": 0.118, "non_rate": 0.050, "cramers_v": 0.102, "p_value": 1.788e-46, "significant": True},
            {"feature": "out_formal_vocab_any", "label": "Formal vocabulary (agent)",
             "esc_rate": 0.047, "non_rate": 0.019, "cramers_v": 0.064, "p_value": 2.807e-19, "significant": True},
            {"feature": "inb_neg_emoji_any", "label": "Negative emoji (customer)",
             "esc_rate": 0.052, "non_rate": 0.041, "cramers_v": 0.018, "p_value": 9.375e-03, "significant": False},
        ],
        "continuous_drivers": [
            {"feature": "agent_sim", "label": "Agent template similarity",
             "esc_median": 0.141, "non_median": 0.000, "rank_biserial": -0.675, "p_value": 0.0, "significant": True},
            {"feature": "response_lag_max", "label": "Max response lag (min)",
             "esc_median": 118.033, "non_median": 41.817, "rank_biserial": -0.250, "p_value": 6.471e-103, "significant": True},
            {"feature": "inb_negation_ratio", "label": "Negation ratio (customer)",
             "esc_median": 0.333, "non_median": 0.000, "rank_biserial": -0.206, "p_value": 2.830e-91, "significant": True},
            {"feature": "inb_caps_mean", "label": "Caps ratio (customer)",
             "esc_median": 0.042, "non_median": 0.035, "rank_biserial": -0.107, "p_value": 1.614e-21, "significant": True},
            {"feature": "out_passive_ratio", "label": "Passive voice ratio (agent)",
             "esc_median": 0.000, "non_median": 0.000, "rank_biserial": -0.103, "p_value": 3.917e-45, "significant": True},
            {"feature": "response_lag_mean", "label": "Mean response lag (min)",
             "esc_median": 41.622, "non_median": 31.650, "rank_biserial": -0.076, "p_value": 6.703e-11, "significant": True},
            {"feature": "out_we_ratio", "label": "We-not-I ratio (agent)",
             "esc_median": 0.375, "non_median": 0.250, "rank_biserial": 0.033, "p_value": 1.972e-03, "significant": False},
            {"feature": "out_formality_mean", "label": "Formality F-score (agent)",
             "esc_median": 49.444, "non_median": 50.000, "rank_biserial": -0.013, "p_value": 0.275, "significant": False},
        ],
        "distributions": [
            {"feature": "agent_sim", "label": "Agent template similarity",
             "esc": {"q1": 0.0, "median": 0.141, "q3": 0.42, "mean": 0.22, "lo": 0.0, "hi": 0.78},
             "non": {"q1": 0.0, "median": 0.0, "q3": 0.0, "mean": 0.05, "lo": 0.0, "hi": 0.31}},
            {"feature": "response_lag_max", "label": "Max response lag (min)",
             "esc": {"q1": 38.0, "median": 118.0, "q3": 360.0, "mean": 290.0, "lo": 0.0, "hi": 900.0},
             "non": {"q1": 12.0, "median": 41.8, "q3": 140.0, "mean": 120.0, "lo": 0.0, "hi": 520.0}},
            {"feature": "inb_negation_ratio", "label": "Negation ratio (customer)",
             "esc": {"q1": 0.0, "median": 0.333, "q3": 0.6, "mean": 0.34, "lo": 0.0, "hi": 1.0},
             "non": {"q1": 0.0, "median": 0.0, "q3": 0.33, "mean": 0.17, "lo": 0.0, "hi": 1.0}},
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
            print(f"[dashboard] failed to read export, using bundled: {e}")
    if _BUNDLED_DASH is None:
        _BUNDLED_DASH = _build_bundled_dashboard()
    return jsonify(_BUNDLED_DASH)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.route("/api/root")
def api_root():
    roots = THREADS.get("roots") or _BUNDLED["roots"]
    return jsonify(random.choice(roots) | {"bundled": USING_BUNDLED})


@app.route("/api/middle")
def api_middle():
    middles = THREADS.get("middles") or _BUNDLED["middles"]
    m = random.choice(middles)
    return jsonify(m | {"bundled": USING_BUNDLED})


@app.route("/api/score", methods=["POST"])
def api_score():
    payload = request.get_json(force=True)
    turns = payload.get("turns", [])
    start_hour = int(payload.get("start_hour", 12))

    if not any((t.get("text") or "").strip() for t in turns):
        return jsonify({"error": "No text to score yet."}), 400

    feat = features.aggregate_thread(turns, score_sentiment, start_hour=start_hour)
    n_turns = len(turns)
    n_agent = sum(1 for t in turns if not t.get("inbound"))

    proba = _model_proba(feat) if n_turns >= 4 else None
    result = scoring.score_thread(feat, n_turns, n_agent, model_proba=proba)
    result["model_loaded"] = MODEL is not None
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)