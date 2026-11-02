# -*- coding: utf-8 -*-
"""
Test / use the trained readmission model.

Run:  python test_model.py

Does three things:
 1. Re-splits the data EXACTLY like training and re-evaluates the saved model
    on the held-out test set (set sizes depend on the artifact's drop_late flag).
 2. Prints a demo: individual patient predictions (like a clinical scoring tool).
 3. Refreshes readmission_model.joblib with the category map needed for
    predicting new CSVs (see predict.py) — preserving all artifact keys.
"""

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)

from readmission_utils import prepare_data, align_categories, strip_late_rows

RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"


def split_like_training(df):
    """Reproduces the exact 70/15/15 patient-grouped split of the training run."""
    groups = df["patient_nbr"]
    feature_cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[feature_cols].copy(), df["target"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    val_rel, te_rel = next(gss2.split(X.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    te_idx = temp_idx[te_rel]

    return X, y, feature_cols, te_idx


def main():
    artifact = joblib.load(ARTIFACT_PATH)
    model = artifact["model"]
    thr_acc = artifact["threshold_accuracy"]
    thr_f1 = artifact["threshold_f1"]

    # ------------------------------------------------------- rebuild data
    target_mode = artifact.get("target_mode", "early")
    artifacts_need = artifact.get("drop_late", False)
    is_trajectory = artifact.get("model_type") == "longitudinal_trajectory_lgbm"
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, top_specialties = prepare_data(df_raw, target_mode=target_mode, keep_first_per_patient=not is_trajectory)
    if artifacts_need:
        df = strip_late_rows(df_raw, df)   # drop patients readmitted '>30'

    feature_cols = artifact["features"]
    X, y, all_cols, te_idx = split_like_training(df)
    X_te, y_te = X[feature_cols].iloc[te_idx], y[te_idx]

    # ------------------------------------------------- 1. hold-out metrics
    probs = model.predict_proba(X_te)[:, 1]
    from sklearn.metrics import roc_auc_score, average_precision_score
    auc = roc_auc_score(y_te, probs)
    pr_auc = average_precision_score(y_te, probs)

    preds_acc = (probs >= thr_acc).astype(int)
    preds_f1 = (probs >= thr_f1).astype(int)

    print("=" * 62)
    print(f"1) VERIFICATION on the held-out test set ({len(y_te):,} patients"
          + (" — patients '>30' exclus" if artifacts_need else "") + ")")
    print("=" * 62)
    print(f"Features in model                : {len(feature_cols)} variables")
    print(f"ROC-AUC                          : {auc:.4f}")
    print(f"PR-AUC                           : {pr_auc:.4f}")
    print(f"Accuracy @ accuracy-tuned thr {thr_acc:.3f} : "
          f"{accuracy_score(y_te, preds_acc):.4f}")
    print(f"Accuracy @ F1-optimal thr {thr_f1:.3f}     : "
          f"{accuracy_score(y_te, preds_f1):.4f}")

    print("\n-- classification report @ F1 threshold (usable operating point) --")
    print(classification_report(y_te, preds_f1, digits=3))
    print(confusion_matrix(y_te, preds_f1))

    # --------------------------------------- 2. demo: patient-level scoring
    print("=" * 62)
    print(f"2) DEMO - scoring individual patients (target: {target_mode} "
          f"readmission)")
    print("=" * 62)

    demo_idx = te_idx[:6]  # 6 test patients
    demo = df.iloc[demo_idx]
    X_demo = demo[feature_cols]
    p = model.predict_proba(X_demo)[:, 1]

    out = pd.DataFrame({
        f"true_readmit({target_mode})": demo["target"].values,
        "risk_prob": np.round(p, 3),
        "flag@acc-thr": (p >= thr_acc).astype(int),
        "flag@F1-thr": (p >= thr_f1).astype(int),
        "primary_diag": demo["diag1_group"].values,
        "age": demo["age"].values,
        "prev_inpatient": demo["number_inpatient"].values,
    })
    print(out.to_string(index=False))
    print("\nNote: @acc-thr optimise la precision de classification (accuracy). "
          "Le F1 threshold est le meilleur compromis pour signaler les patients.")

    # ------------------------- 3. sanity check on highest-risk patients top-1%
    n_top1 = max(1, len(y_te) // 100)
    top = np.argsort(-probs)[:n_top1]  # top 1% of test patients
    caught = int(y_te[top].sum())
    print(f"\nIf we flag the top 1% riskiest patients ({len(top)}), "
          f"{caught} truly readmitted early -> {caught/len(top):.0%} precision")
    n_top10 = max(1, len(y_te) // 10)
    top10 = np.argsort(-probs)[:n_top10]  # top 10%
    print(f"Top 10% riskiest ({len(top10)}): "
          f"{int(y_te[top10].sum())} true early readmissions "
          f"(base rate would give {(y_te.mean()*len(top10)):.0f})")


if __name__ == "__main__":
    main()
