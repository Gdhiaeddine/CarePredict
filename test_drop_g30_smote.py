# -*- coding: utf-8 -*-
"""
Test de 2 changements de données sur la cible '<30' (réadmission précoce):
  1. IGNORER les lignes '>30' (35,545 patients, négatifs 'faciles')
  2. SMOTE / class_weight pour équilibrer positif vs négatif

Variantes (mêmes split/seed/paramètres, évaluées sur le MÊME test set):
  A  baseline            : '>30' gardées, aucun rééquilibrage   [modèle déployé]
  A+SMOTE                : '>30' gardées, SMOTE 50/50 sur le train
  A+weights              : '>30' gardées, class_weight="balanced"
  B  drop >30            : '>30' JETÉES, aucun rééquilibrage
  B+SMOTE                : '>30' jetées, SMOTE 50/50 sur le train

Règles anti-fuite: SMOTE appliqué SUR LE TRAIN SEUL (pipeline imblearn);
test set jamais modifié; seuils detunés sur la validation.

Run:  python test_drop_g30_smote.py   (~5-8 min)
"""

import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from readmission_utils import prepare_data

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"

BASE_PARAMS = {
    "subsample": 1.0, "reg_lambda": 0, "reg_alpha": 0, "num_leaves": 15,
    "n_estimators": 800, "min_child_samples": 100, "max_depth": 8,
    "learning_rate": 0.03, "colsample_bytree": 0.6,
}


def make_splits(df):
    groups = df["patient_nbr"]
    cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[cols], df["target"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    v_rel, te_rel = next(gss2.split(X.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    return X, y, tr_idx, temp_idx[v_rel], temp_idx[te_rel]


def f1_threshold(y_val, probs):
    best_t, best_f = 0.5, -1
    for t in np.linspace(0.05, 0.95, 181):
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0
        if f1 > best_f:
            best_f, best_t = f1, t
    return best_t


def evaluate(tag, y_te, probs_te, y_val, probs_val):
    thr = f1_threshold(y_val, probs_val)
    pred = (probs_te >= thr).astype(int)
    n = len(y_te)
    top1 = int(n * 0.01)
    ordering = np.argsort(-probs_te)[:top1]
    prec_at1 = y_te[ordering].mean()
    row = {
        "variant": tag,
        "n_test": n,
        "ROC-AUC": round(roc_auc_score(y_te, probs_te), 4),
        "PR-AUC": round(average_precision_score(y_te, probs_te), 4),
        "acc@0.5": round(accuracy_score(y_te, probs_te >= 0.5), 4),
        "acc@opt": round(accuracy_score(y_te, (probs_te >= acc_thr(y_val, probs_val))
                                        .astype(int)), 4),
        "thr": round(thr, 3),
        "P(oui)": round(precision_score(y_te, pred), 4),
        "R(oui)": round(recall_score(y_te, pred), 4),
        "top1%prec": round(prec_at1, 4),
    }
    print(f"{tag:<16} AUC {row['ROC-AUC']:.4f} PR {row['PR-AUC']:.4f} "
          f"acc50 {row['acc@0.5']:.2%} accOpt {row['acc@opt']:.2%} "
          f"P {row['P(oui)']:.2%} R {row['R(oui)']:.2%} top1 {row['top1%prec']:.2%}")
    return row


def acc_thr(y_val, probs_val):
    thresholds = np.linspace(0.05, 0.95, 181)
    accs = [accuracy_score(y_val, (probs_val >= t).astype(int)) for t in thresholds]
    return thresholds[int(np.argmax(accs))]


def build_and_fit(tag, X_tr, y_tr, X_val, X_te, use_smote, weighted):
    num_cols = [c for c in X_tr.columns if X_tr[c].dtype != "category"]
    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == "category"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         cat_cols),
    ])
    model = LGBMClassifier(
        class_weight="balanced" if weighted else None,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1, **BASE_PARAMS)
    if use_smote:
        pipe = ImbPipeline([
            ("oh", pre),
            ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
            ("clf", model),
        ])
    else:
        pipe = ImbPipeline([("oh", pre), ("clf", model)])
    pipe.fit(X_tr, y_tr)
    pv = pipe.predict_proba(X_val)[:, 1]
    pt = pipe.predict_proba(X_te)[:, 1]
    return pv, pt


def main():
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df_all, _ = prepare_data(df_raw, target_mode="early")
    print(f"All rows (baseline A path): {len(df_all):,}")

    # ------- Path A: keep '>30' as negatives (current deployed settings)
    X, y, tr_idx, val_idx, te_idx = make_splits(df_all)
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]
    rows = []

    pv, pt = build_and_fit("A", X_tr, y_tr, X_val, X_te,
                           use_smote=False, weighted=False)
    rows.append(evaluate("A baseline", y_te, pt, y_val, pv))

    pv, pt = build_and_fit("A+SMOTE", X_tr, y_tr, X_val, X_te,
                           use_smote=True, weighted=False)
    rows.append(evaluate("A+SMOTE", y_te, pt, y_val, pv))

    pv, pt = build_and_fit("A+weights", X_tr, y_tr, X_val, X_te,
                           use_smote=False, weighted=True)
    rows.append(evaluate("A+weights", y_te, pt, y_val, pv))

    # ------- Path B: DROP the >30 rows entirely
    df_late_raw = df_raw[df_raw["readmitted"] == ">30"][["patient_nbr"]]
    df_b = (df_all.merge(df_late_raw.assign(_late=True), on="patient_nbr",
                         how="left"))
    df_b = df_b[df_b["_late"].isna()].drop(columns=["_late"])
    print(f"\nPath B (after dropping '>30'): {len(df_b):,} rows "
          f"(target dist: {df_b['target'].value_counts(normalize=True).to_dict()})")

    Xb, yb, trb, valb, teb = make_splits(df_b)
    Xb_tr, yb_tr = Xb.iloc[trb], yb[trb]
    Xb_val, yb_val = Xb.iloc[valb], yb[valb]
    Xb_te, yb_te = Xb.iloc[teb], yb[teb]

    pv, pt = build_and_fit("B drop>30", Xb_tr, yb_tr, Xb_val, Xb_te,
                           use_smote=False, weighted=False)
    rows.append(evaluate("B drop>30", yb_te, pt, yb_val, pv))

    pv, pt = build_and_fit("B+SMOTE", Xb_tr, yb_tr, Xb_val, Xb_te,
                           use_smote=True, weighted=False)
    rows.append(evaluate("B+SMOTE", yb_te, pt, yb_val, pv))

    res = pd.DataFrame(rows)
    print("\n=== TABLEAU COMPARATIF ===")
    print(res.to_string(index=False))
    res.to_csv("drop_g30_smote_results.csv", index=False)
    print("\n-> drop_g30_smote_results.csv")
    print("Rappel: SMOTE n'est appliqué QUE sur le train (fuite impossible).")


if __name__ == "__main__":
    main()
