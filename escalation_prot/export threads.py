"""
export_threads.py — run in your training notebook to produce the JSON the
predictor page loads. Needs the tweet-level frame WITH thread structure in scope
(the one that has thread_root, turn_index, inbound, text) and df_threads (for the
escalated label). Adjust the two variable names at the top if yours differ.

Writes artifacts/predictor_threads.json with two random samples:

  "roots"  : opening customer tweets, for Mode 1 ("from start"). Each is just an
             inbound root the user can start from.
  "middles": threads with >= 4 turns, for Mode 2 ("in the middle"). Each gives the
             first 3 turns (the prompt), the real continuation, and the escalated
             label (the reveal).

Random sampling, fixed seed, no curation — defensible and reproducible.
"""

import json
import os

# ---- adjust to your notebook variable names ------------------------------
tweets = tweets_scored      # tweet-level frame: thread_root, turn_index, inbound, text
threads = df_threads        # thread-level frame: thread_root, escalated
# --------------------------------------------------------------------------

N_ROOTS = 300
N_MIDDLES = 300
SEED = 42
OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

label_by_root = dict(zip(threads["thread_root"], threads["escalated"]))

# group tweets into ordered turn lists
grouped = {}
for tid, g in tweets.sort_values("turn_index").groupby("thread_root"):
    grouped[tid] = [
        {"inbound": bool(r.inbound), "text": str(r.text)}
        for r in g.itertuples()
    ]

all_ids = [t for t in grouped if t in label_by_root]

import random
random.seed(SEED)

# Mode 1: roots whose first turn is inbound (a real customer opener)
root_candidates = [
    t for t in all_ids
    if grouped[t] and grouped[t][0]["inbound"]
]
roots_sample = random.sample(root_candidates, min(N_ROOTS, len(root_candidates)))
roots = [
    {
        "thread_root": t,
        "opening_text": grouped[t][0]["text"],
    }
    for t in roots_sample
]

# Mode 2: threads with >= 4 turns following customer -> company -> customer -> ...
middle_candidates = []
for t in all_ids:
    turns = grouped[t]
    if len(turns) < 4:
        continue
    # require the canonical opening pattern: inbound, outbound, inbound
    if turns[0]["inbound"] and (not turns[1]["inbound"]) and turns[2]["inbound"]:
        middle_candidates.append(t)

middles_sample = random.sample(middle_candidates, min(N_MIDDLES, len(middle_candidates)))
middles = [
    {
        "thread_root": t,
        "prompt_turns": grouped[t][:3],          # what the user sees
        "real_continuation": grouped[t][3:],     # the reveal
        "escalated": int(label_by_root[t]),      # the outcome
    }
    for t in middles_sample
]

payload = {"roots": roots, "middles": middles}
with open(os.path.join(OUT, "predictor_threads.json"), "w") as f:
    json.dump(payload, f, ensure_ascii=False, indent=2)

print(f"roots: {len(roots)} (from {len(root_candidates)} candidates)")
print(f"middles: {len(middles)} (from {len(middle_candidates)} candidates)")
print("wrote artifacts/predictor_threads.json")