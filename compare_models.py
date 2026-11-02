# -*- coding: utf-8 -*-
"""
Compare Logistic Regression / Random Forest / XGBoost / LightGBM(ensemble)
on the SAME prepared data and yield the train (val) threshold F1, test
metrics for each.

Run:  python compare_models.py   (~3-5 min)
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score)

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
BASE_PARAMS = {  # same tuned LightGBM params as select_features_retrain.py
    "subsample": 1.0, "reg_lambda": 0, "reg_alpha": 0, "num_leaves": 15,
    "n_estimators": 800, "min_child_samples": 100, "max_depth": 8,
    "learning_rate": 0.03, "colsample_bytree": 0.6, "class_weight": None,
}


def split(df):
    groups = df["patient_nbr"]
    cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[cols], df["target"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    v_rel, te_rel = next(gss2.split(X.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    return X, y, cols, tr_idx, temp_idx[v_rel], temp_idx[te_rel]


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


def main():
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, _ = prepare_data(df_raw, target_mode="any")
    X, y, feat_cols, tr_idx, val_idx, te_idx = split(df)
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    # ---------------- shared preprocessing for the classic (dense) models
    num_cols = [c for c in feat_cols if X_tr[c].dtype != "category"]
    cat_cols = [c for c in feat_cols if X_tr[c].dtype == "category"]
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]), cat_cols),
    ])
    X_tr_p = pre.fit_transform(X_tr)          # fit on TRAIN only
    X_val_p = pre.transform(X_val)
    X_te_p = pre.transform(X_te)
    print(f"Encoded for classic models: {X_tr_p.shape}")

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000, random_state=RANDOM_STATE, n_jobs=-1),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=400, learning_rate=0.05,
                                 max_depth=6, random_state=RANDOM_STATE,
                                 n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                                   verbosity=-1, **BASE_PARAMS),
    }

    print(f"{'Model':<22}{'ROC-AUC':>9}{'PR-AUC':>9}{'Acc@0.5':>9}"
          f"{'Acc(thr)':>10}{'P(oui)':>9}{'R(oui)':>9}{'F1(oui)':>9}")
    rows = []
    for name, m in models.items():
        dense = name != "LightGBM"
        m.fit(X_tr_p if dense else align_categories(X_tr, art_cat()),
              y_tr)
        pv_ = m.predict_proba(X_val_p if dense
                              else align_categories(X_val, art_cat()))[:, 1]
        thr = f1_threshold(y_val, pv_)
        pt_ = m.predict_proba(X_te_p if dense
                              else align_categories(X_te, art_cat()))[:, 1]
        pred = (pt_ >= thr).astype(int)
        row = dict(model=name, auc=roc_auc_score(y_te, pt_),
                   pr_auc=average_precision_score(y_te, pt_),
                   acc05=accuracy_score(y_te, pt_ >= 0.5),
                   thr=thr, fp=precision_score(y_te, pred),
                   rc=recall_score(y_te, pred), f1=f1_score(y_te, pred))
        rows.append(row)
        print(f"{name:<22}{row['auc']:>9.4f}{row['pr_auc']:>9.4f}"
              f"{row['acc05']:>9.2%}{row['thr']:>10.3f}"
              f"{row['fp']:>9.2%}{row['rc']:>9.2%}{row['f1']:>9.2%}")

    summary = pd.DataFrame(rows).sort_values("auc", ascending=False)
    summary.to_csv("model_comparison.csv", index=False)
    print("\nsaved -> model_comparison.csv")


def art_cat():
    """category_map for LightGBM native path (loaded lazily)"""
    global _ART
    if _ART is None:
        _ART = joblib.load("readmission_model.joblib")
    return _ART["category_map"]


_ART = None

if __name__ == "__main__":
    main()
