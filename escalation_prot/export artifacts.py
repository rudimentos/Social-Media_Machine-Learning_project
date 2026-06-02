"""
export_artifacts.py — run ONCE at the end of your training notebook (after STEP 11,
once model_a and scaler exist) to write the three files the Flask app loads.

The feature_columns.json is the contract between training and serving: the app
reindexes its computed features to exactly this order before scaling and scoring.
Saving it from the live X guarantees the order matches, rather than hardcoding it.
"""

import json
import os

import joblib

# Expecting these to already exist in your notebook namespace:
#   model_a  -> trained XGBClassifier (scale_pos_weight version, carried forward)
#   scaler   -> the StandardScaler fitted on X_train
#   X        -> the feature DataFrame BEFORE scaling (defines column order)

OUT = "artifacts"
os.makedirs(OUT, exist_ok=True)

# XGBoost native format is version-robust and reloads cleanly into XGBClassifier.
model_a.save_model(os.path.join(OUT, "model.json"))

# StandardScaler has no native format; joblib is standard for sklearn objects.
joblib.dump(scaler, os.path.join(OUT, "scaler.pkl"))

with open(os.path.join(OUT, "feature_columns.json"), "w") as f:
    json.dump(list(X.columns), f, indent=2)

print(f"wrote {len(X.columns)} feature columns")
print("artifacts/model.json, artifacts/scaler.pkl, artifacts/feature_columns.json")