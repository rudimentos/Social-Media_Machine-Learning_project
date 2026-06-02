"""
scoring.py — escalation evaluation for the predictor (coaching tool).

Two layers, kept separate on purpose:

  Layer A (core, always shown): a TRANSPARENT additive "escalation pressure"
    read built from the step-9 driver findings. Every point traces to an effect
    size and an empirical escalation rate. This is NOT a probability and is
    deliberately not called one, so it can't be confused with the model output.

  Layer B (secondary, only when the thread has >= 4 turns AND a model is loaded):
    the XGBoost calibrated probability. Below 4 turns the thread is outside the
    model's training distribution (agent_sim and the thread aggregates are not
    meaningfully defined), so B is withheld and a note explains why.

The numbers in DRIVERS come from STEP 9 of the project:
  out_deflection_any    esc 31.5% vs non 17.2%   Cramer's V 0.127
  out_bureaucratic_any  esc 20.7% vs non 10.3%   Cramer's V 0.114
  inb_intent_any        esc 11.8% vs non  5.0%   Cramer's V 0.102
  inb_frustration_any   esc 11.2% vs non  4.4%   Cramer's V 0.106
  out_formal_vocab_any  esc  4.7% vs non  1.9%   Cramer's V 0.064
  out_passive_ratio     rank-biserial 0.103 (continuous)
  response_lag_max      rank-biserial 0.250 (continuous, minutes)
  agent_sim             rank-biserial 0.675 (continuous; needs >=2 agent turns)
  out_formality_mean    NON-significant (p=0.275) — included at weight 0
"""

# ---------------------------------------------------------------------------
# Driver registry. weight = step-9 effect-size magnitude used as the additive
# coefficient. side = who controls it. kind = how the feature value is read.
# Each driver carries the empirical escalation rates so the UI can cite them.
# ---------------------------------------------------------------------------
DRIVERS = [
    # ---- agent-side (the coaching targets) -------------------------------
    {
        "key": "out_deflection_any", "side": "reply", "kind": "binary",
        "weight": 0.127, "esc_rate": 0.315, "non_rate": 0.172,
        "label": "Deflection language",
        "advice": "Avoid 'unable', 'cannot', 'policy', 'contact another department'. "
                  "Threads with deflection language escalated almost twice as often.",
    },
    {
        "key": "out_bureaucratic_any", "side": "reply", "kind": "binary",
        "weight": 0.114, "esc_rate": 0.207, "non_rate": 0.103,
        "label": "Bureaucratic language",
        "advice": "Avoid 'kindly', 'we appreciate your patience', 'we have flagged this'. "
                  "Formulaic reassurance is associated with double the escalation rate.",
    },
    {
        "key": "out_passive_ratio", "side": "reply", "kind": "ratio",
        "weight": 0.103, "esc_rate": None, "non_rate": None,
        "label": "Passive voice",
        "advice": "Prefer active voice ('we will refund you') over passive "
                  "('a refund will be processed'). Passive constructions are "
                  "associated with escalated threads.",
    },
    {
        "key": "out_formal_vocab_any", "side": "reply", "kind": "binary",
        "weight": 0.064, "esc_rate": 0.047, "non_rate": 0.019,
        "label": "Overly formal vocabulary",
        "advice": "Plain wording reads as more human than heavy formal vocabulary.",
    },
    {
        "key": "response_lag_max", "side": "reply", "kind": "lag",
        "weight": 0.250, "esc_rate": None, "non_rate": None,
        "label": "Slow response",
        "advice": "Reply quickly. Escalated threads had a median worst-case wait of "
                  "~118 min vs ~42 min for resolved threads.",
        # scaled by how far the max gap exceeds the resolved-thread median (~42 min)
        "lag_pivot_min": 42.0, "lag_full_min": 118.0,
    },
    {
        "key": "agent_sim", "side": "reply", "kind": "simrange",
        "weight": 0.675, "esc_rate": None, "non_rate": None,
        "label": "Templated / repetitive replies",
        "advice": "Vary your wording across turns. Copy-paste style replies "
                  "(high similarity between agent turns) were the single strongest "
                  "correlate of unresolved threads.",
        # esc median agent_sim ~0.141, non ~0.000; treat >0.30 as full pressure
        "sim_pivot": 0.05, "sim_full": 0.30,
    },
    {
        "key": "out_formality_mean", "side": "reply", "kind": "noneffect",
        "weight": 0.0, "esc_rate": None, "non_rate": None,
        "label": "Formality score (F-score)",
        "advice": "Tested but NOT a significant predictor (p=0.275). Shown for "
                  "completeness; carries no weight in the score.",
    },
    # ---- customer-side (incoming context, not the agent's fault) ---------
    {
        "key": "inb_intent_any", "side": "incoming", "kind": "binary",
        "weight": 0.102, "esc_rate": 0.118, "non_rate": 0.050,
        "label": "Escalation-intent terms",
        "advice": "Customer used terms like 'cancel', 'refund', 'lawyer', 'switching'. "
                  "These raise baseline risk — handle with extra care.",
    },
    {
        "key": "inb_frustration_any", "side": "incoming", "kind": "binary",
        "weight": 0.106, "esc_rate": 0.112, "non_rate": 0.044,
        "label": "Frustration terms",
        "advice": "Customer used strong negative terms ('unacceptable', 'useless'). "
                  "Baseline risk is elevated.",
    },
]

# weight totals per side, for normalising each side to 0-100
_REPLY_MAX = sum(d["weight"] for d in DRIVERS if d["side"] == "reply")
_INCOMING_MAX = sum(d["weight"] for d in DRIVERS if d["side"] == "incoming")


def _driver_activation(d, feat):
    """Return activation in [0,1] for one driver given the thread feature dict."""
    v = feat.get(d["key"], 0) or 0
    kind = d["kind"]
    if kind == "binary":
        return 1.0 if v else 0.0
    if kind == "ratio":                 # 0..1 already (proportion of agent turns)
        return max(0.0, min(1.0, float(v)))
    if kind == "noneffect":
        return 0.0
    if kind == "lag":
        pivot, full = d["lag_pivot_min"], d["lag_full_min"]
        if v <= pivot:
            return 0.0
        return max(0.0, min(1.0, (float(v) - pivot) / (full - pivot)))
    if kind == "simrange":
        pivot, full = d["sim_pivot"], d["sim_full"]
        if v <= pivot:
            return 0.0
        return max(0.0, min(1.0, (float(v) - pivot) / (full - pivot)))
    return 0.0


def score_thread(feat, n_turns, n_agent_turns, model_proba=None):
    """Evaluate a thread-so-far.

    feat: the aggregated thread feature dict (from features.aggregate_thread).
    n_turns / n_agent_turns: turn counts, used to gate agent_sim and Layer B.
    model_proba: optional Layer-B probability already computed by the caller.

    Returns a dict the route can hand straight to the template.
    """
    contributions = []
    reply_raw = 0.0
    incoming_raw = 0.0

    for d in DRIVERS:
        # agent_sim is undefined with fewer than 2 agent turns — skip cleanly
        if d["key"] == "agent_sim" and n_agent_turns < 2:
            continue
        act = _driver_activation(d, feat)
        contrib = act * d["weight"]
        if d["side"] == "reply":
            reply_raw += contrib
        elif d["side"] == "incoming":
            incoming_raw += contrib

        if act > 0 or d["kind"] == "noneffect":
            contributions.append({
                "key": d["key"],
                "label": d["label"],
                "side": d["side"],
                "activation": round(act, 3),
                "points": round(contrib, 4),
                "esc_rate": d["esc_rate"],
                "non_rate": d["non_rate"],
                "advice": d["advice"],
                "value": round(float(feat.get(d["key"], 0) or 0), 4),
            })

    reply_pressure = round(100 * reply_raw / _REPLY_MAX, 1) if _REPLY_MAX else 0.0
    incoming_pressure = round(100 * incoming_raw / _INCOMING_MAX, 1) if _INCOMING_MAX else 0.0

    # overall read: reply pressure is what the agent controls, so it leads;
    # incoming pressure shades the baseline. simple documented blend.
    overall = round(0.65 * reply_pressure + 0.35 * incoming_pressure, 1)

    if overall >= 55:
        band = "high"
    elif overall >= 25:
        band = "moderate"
    else:
        band = "low"

    contributions.sort(key=lambda c: c["points"], reverse=True)

    layer_b = None
    if n_turns >= 4:
        if model_proba is not None:
            layer_b = {
                "available": True,
                "probability": round(float(model_proba), 4),
                "note": "Secondary signal from the trained XGBoost model. "
                        "Indicative only — the model was trained on completed "
                        "threads, so treat as supporting evidence, not the verdict.",
            }
        else:
            layer_b = {
                "available": False,
                "note": "Trained model not loaded — run export_artifacts.py in the "
                        "notebook to enable the secondary model probability.",
            }
    else:
        layer_b = {
            "available": False,
            "note": "Model probability withheld until the thread reaches 4 turns. "
                    "Below that, thread-level features (e.g. agent_sim) are not "
                    "meaningfully defined and the model would be extrapolating.",
        }

    return {
        "overall_pressure": overall,
        "band": band,
        "reply_pressure": reply_pressure,
        "incoming_pressure": incoming_pressure,
        "contributions": contributions,
        "layer_b": layer_b,
        "blend_note": "Overall = 0.65 x reply pressure + 0.35 x incoming pressure. "
                      "A transparent weighted index of step-9 drivers, NOT a probability.",
    }