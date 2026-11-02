# -*- coding: utf-8 -*-
"""
L'anomalie detection est-elle efficace pour la réadmission <30 jours ?

Compare, sur le MEME test set (split identique à l'entraînement) :
  - IsolationForest (sur tout le train)
  - IsolationForest one-class (entraîné sur les patients NON réadmis seulement)
  - SGD One-Class SVM (nu = taux de réadmission ~9%)
  - Local Outlier Factor (novelty=True, sous-échantillon)
  - Erreur de reconstruction PCA
  - LightGBM supervisé (référence, readmission_model.joblib)

Réponse quantifiée : ROC-AUC, PR-AUC, precision@top1/5/10%, meilleur seuil F1.

Run:  python test_anomaly_detection.py   (~2-3 min)
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import SGDOneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score)

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"
LOF_SUBSAMPLE = 8000  # LOF cost is quadratic -> subsample train set


def split(df):
    groups = df["patient_nbr"]
    cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[cols], df["target"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    v_rel, te_rel = next(gss2.split(X.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    return X, y, cols, tr_idx, temp_idx[v_rel], temp_idx[te_rel]


def f1_thr(y_val, s_val):
    best_t, best_f = np.median(s_val), -1
    for t in np.quantile(s_val, np.linspace(0.005, 0.30, 120)):
        pred = (s_val >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        if f1 > best_f:
            best_f, best_t = f1, t
    return best_t


def report(name, s_val, y_val, s_te, y_te):
    """s = anomaly score, higher = more anomalous."""
    auc = roc_auc_score(y_te, s_te)
    pr = average_precision_score(y_te, s_te)
    n_te = len(y_te)
    line = (f"{name:<34} AUC {auc:.4f}  PR-AUC {pr:.4f}")
    for k in (0.01, 0.05, 0.10):
        n = max(1, int(n_te * k))
        top = np.argsort(-s_te)[:n]
        line += f"  | P@{int(k * 100)}% {y_te[top].mean():.2%}"
    t = f1_thr(y_val, s_val)
    pred = (s_te >= t).astype(int)
    line += f"  | F1(thr) precision {precision_score(y_te, pred, zero_division=False):.2%} recall {recall_score(y_te, pred, zero_division=False):.2%} flag {pred.mean():.1%}"
    print(line)
    return auc


def main():
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, _ = prepare_data(df_raw)
    X, y, feat_cols, tr_idx, val_idx, te_idx = split(df)
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    # ----------------------- one-hot pipeline for unsupervised detectors
    pre = ColumnTransformer([
        ("num", StandardScaler(), [c for c in feat_cols
                                   if X_tr[c].dtype != "category"]),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         [c for c in feat_cols if X_tr[c].dtype == "category"]),
    ])
    X_tr_p = pre.fit_transform(X_tr).astype(np.float32)
    X_val_p = pre.transform(X_val).astype(np.float32)
    X_te_p = pre.transform(X_te).astype(np.float32)
    print(f"Encoded features for anomaly models: {X_tr_p.shape}\n")

    results = {}

    # ------------------------------------------------ supervised reference
    art = joblib.load(ARTIFACT_PATH)
    X_te_sup = align_categories(X_te[art["features"]], art["category_map"])
    X_val_sup = align_categories(X_val[art["features"]], art["category_map"])
    s_val = art["model"].predict_proba(X_val_sup)[:, 1]
    s_te = art["model"].predict_proba(X_te_sup)[:, 1]
    rs = report("SUPERVISED LightGBM (reference)", s_val, y_val, s_te, y_te)
    results["LightGBM supervisé (ref)"] = rs

    # ------------------------------------------------------- detectors
    det = {}

    iforest = IsolationForest(n_estimators=300, contamination=0.09,
                              random_state=RANDOM_STATE, n_jobs=-1).fit(X_tr_p)
    det["IsolationForest (tout le train)"] = (
        -iforest.score_samples(X_val_p), -iforest.score_samples(X_te_p))

    mask = y_tr == 0  # robust variant: negatives only
    iforest_neg = IsolationForest(n_estimators=300, contamination="auto",
                                  random_state=RANDOM_STATE,
                                  n_jobs=-1).fit(X_tr_p[mask])
    det["IsolationForest one-class (neg)"] = (
        -iforest_neg.score_samples(X_val_p), -iforest_neg.score_samples(X_te_p))

    ocsvm = SGDOneClassSVM(nu=0.09, random_state=RANDOM_STATE).fit(X_tr_p)
    det["One-Class SVM (SGD)"] = (
        -ocsvm.decision_function(X_val_p), -ocsvm.decision_function(X_te_p))

    rng = np.random.RandomState(RANDOM_STATE)
    lof_idx = rng.choice(len(X_tr_p), size=LOF_SUBSAMPLE, replace=False)
    lof = LocalOutlierFactor(n_neighbors=50, novelty=True,
                             contamination=0.09, n_jobs=-1).fit(X_tr_p[lof_idx])
    det["Local Outlier Factor (subsample)"] = (
        -lof.decision_function(X_val_p), -lof.decision_function(X_te_p))

    pca = PCA(n_components=30, random_state=RANDOM_STATE).fit(X_tr_p)
    def pca_err(M):
        rec = pca.inverse_transform(pca.transform(M))
        return np.mean((M - rec) ** 2, axis=1)
    det["PCA reconstruction error"] = (pca_err(X_val_p), pca_err(X_te_p))

    for name, (sv, st) in det.items():
        results[name] = report(name, sv, y_val, st, y_te)

    # --------------------------------------------------------- verdict
    print("\n=== VERDICT ===")
    best_unsup = max((k, v) for k, v in results.items() if "supervisé" not in k)
    print(f"Meilleur superviseur : AUC {results['LightGBM supervisé (ref)']:.4f}")
    print(f"Meilleur anomaly detector : {best_unsup[0]}  AUC {best_unsup[1]:.4f}")
    gap = results['LightGBM supervisé (ref)'] - best_unsup[1]
    print(f"Ecart d'AUC décision supervisée - non supervisée : {gap:.4f}")
    print("Interpretation: une AUC < 0.60 veut dire que le detector classe les")
    print("patients presque au hasard pour la re admission. Efficace ? -> non,")
    print("comparer aux thresholds du modele supervisé (improve_precision.py).")


if __name__ == "__main__":
    main()
