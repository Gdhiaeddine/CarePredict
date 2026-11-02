# -*- coding: utf-8 -*-
"""
Predict readmission risk for NEW patients from a CSV. The target and the
operating thresholds come from readmission_model.joblib
(target_mode = "any": OUI if readmitted <30 OR >30 days).

Usage:
    python predict.py patients.csv          # scores the given rows
    python predict.py diabetic_data.csv 50  # scores the first 50 rows of a file

The input CSV must contain the original columns of diabetic_data.csv
(the rows are scored exactly as patients, first encounter per patient_nb).

Output: <input>_predictions.csv with added columns:
    risk_prob        - predicted probability of readmission
    flag_accuracy    - prediction at the accuracy-tuned threshold
    flag_f1          - prediction at the F1-optimal threshold
"""

import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
ARTIFACT_PATH = "readmission_model.joblib"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit("Error: give the path to a CSV of patients, e.g. "
                 "python predict.py patients.csv")
    path = sys.argv[1]
    n_rows = int(sys.argv[2]) if len(sys.argv) > 2 else None

    art = joblib.load(ARTIFACT_PATH)
    model = art["model"]
    thr_acc = art["threshold_accuracy"]
    thr_f1 = art["threshold_f1"]
    thresholds_extra = art.get("thresholds_extra", {})
    feature_cols = art["features"]
    category_map = art["category_map"]

    df_raw = pd.read_csv(path, na_values="?", low_memory=False)
    if n_rows:
        df_raw = df_raw.head(n_rows)

    # Each input row = a patient (no dedupe). Group rare specialties with the
    # training list so the feature space matches the model.
    if art.get("drop_late") and "readmitted" in df_raw.columns:
        df_raw = df_raw[df_raw["readmitted"].fillna("") != ">30"]
    brought, top_specialties = prepare_data(
        df_raw, top_specialties=art["top_specialties"],
        remove_deaths=False, keep_first_per_patient=False)

    missing = [c for c in feature_cols if c not in brought.columns]
    if missing:
        sys.exit(f"Error: input is missing required columns: {missing}. "
                 f"Provide the original diabetic_data.csv-style columns.")

    X_new = align_categories(brought[feature_cols], category_map)
    probs = model.predict_proba(X_new)[:, 1]

    assign_cols = {
        "risk_prob": np.round(probs, 4),
        "risk_percent": np.round(probs * 100, 1),
        "flag_accuracy": (probs >= thr_acc).astype(int),
        "flag_f1": (probs >= thr_f1).astype(int),
    }
    for name, thr in thresholds_extra.items():
        assign_cols[f"flag_{name}"] = (probs >= float(thr)).astype(int)
    out = df_raw.assign(**assign_cols)
    cols = list(df_raw.columns) + list(assign_cols.keys())
    out = out[cols]

    destination = os.path.splitext(path)[0] + "_predictions.csv"
    out.to_csv(destination, index=False)

    print(f"Scored {len(out):,} patients -> {destination}")
    print(f"  flagged (accuracy-thr {thr_acc:.3f}): "
          f"{int(out.flag_accuracy.sum()):,} patients")
    print(f"  flagged (F1-thr {thr_f1:.3f}):       "
          f"{int(out.flag_f1.sum()):,} patients")
    for name, thr in thresholds_extra.items():
        print(f"  flagged ({name} thr {float(thr):.3f}):  "
              f"{int(out[f'flag_{name}'].sum()):,} patients")
    print("\nTop 10 highest-risk patients:")
    show = out.sort_values("risk_prob", ascending=False).head(10)
    cols_show = [c for c in ["patient_nbr", "age", "race", "gender",
                             "target", "risk_percent", "flag_f1"]
                 if c in show.columns]
    print(show[cols_show].to_string(index=False))


if __name__ == "__main__":
    main()
