# -*- coding: utf-8 -*-
"""
DEPLOY the validated B+SMOTE model:
  - cible '<30' (early)
  - '>30' jetées (négatifs toxiques) -> drop_late=True
  - one-hot + SMOTE(train only) + LightGBM (pipeline imblearn)

Fit sur le split seed=42 validé, seuils tunés sur la validation,
métriques finales imprimées sur le test HORS-production.
Sauvegarde: readmission_model.joblib (clés compatibles test_model.py /
predict.py / app_streamlit.py / app_fastapi.py).

Run:  python deploy_b_smote.py   (~2-3 min)
"""

import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             accuracy_score, precision_score, recall_score,
                             classification_report, confusion_matrix)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from readmission_utils import prepare_data, strip_late_rows

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"


def split(df, seed):
    groups = df["patient_nbr"]
    cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[cols], df["target"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    v_rel, te_rel = next(gss2.split(X.iloc[temp_idx],
                                    groups=groups.iloc[temp_idx]))
    return X, y, cols, tr_idx, temp_idx[v_rel], temp_idx[te_rel]


def thresholds_from_val(y_val, probs):
    t = np.linspace(0.05, 0.95, 181)

    accs = [accuracy_score(y_val, (probs >= x).astype(int)) for x in t]
    thr_acc = t[int(np.argmax(accs))]

    f1s = []
    for x in t:
        pred = (probs >= x).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    thr_f1 = t[int(np.argmax(f1s))]
    return thr_acc, thr_f1


def main():
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, top_specialties = prepare_data(df_raw, target_mode="early")
    df_b = strip_late_rows(df_raw, df)
    print(f"A: {len(df):,} rows -> B (sans '>30'): {len(df_b):,} rows")
    print(f"B target distribution: "
          f"{df_b['target'].value_counts(normalize=True).round(4).to_dict()}")

    X, y, cols, tr, va, te = split(df_b, RANDOM_STATE)
    X_tr, y_tr = X.iloc[tr], y[tr]
    X_va, y_va = X.iloc[va], y[va]
    X_te, y_te = X.iloc[te], y[te]

    num = [c for c in cols if X_tr[c].dtype != "category"]
    cat = [c for c in cols if X_tr[c].dtype == "category"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    clf = LGBMClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0,
                         num_leaves=15, n_estimators=800,
                         min_child_samples=100, max_depth=8,
                         learning_rate=0.03, colsample_bytree=0.6,
                         random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1)
    pipe = ImbPipeline([("pre", pre),
                        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)),
                        ("clf", clf)])
    print("\nFitting B+SMOTE (SMOTE sur le TRAIN seul)...")
    pipe.fit(X_tr, y_tr)

    pv = pipe.predict_proba(X_va)[:, 1]
    pt = pipe.predict_proba(X_te)[:, 1]
    thr_acc, thr_f1 = thresholds_from_val(y_va, pv)

    acc_a = accuracy_score(y_te, (pt >= thr_acc).astype(int))
    pred_f1 = (pt >= thr_f1).astype(int)
    top1 = np.argsort(-pt)[:max(1, int(len(y_te) * 0.01))]

    print("\n=== METRIQUES FINALES (test: 10,496->" f"{len(y_te):,} patients sans '>30') ===")
    print(f"ROC-AUC            : {roc_auc_score(y_te, pt):.4f}")
    print(f"PR-AUC             : {average_precision_score(y_te, pt):.4f}")
    print(f"Accuracy @thr {thr_acc:.3f}  : {acc_a:.2%}")
    print(f"Precision @F1 {thr_f1:.3f} : {precision_score(y_te, pred_f1):.2%} "
          f"| recall {recall_score(y_te, pred_f1):.2%}")
    print(f"Precision top-1%   : {y_te[top1].mean():.2%} "
          f"({int(y_te[top1].sum())}/{len(top1)})")
    print("\n" + classification_report(y_te, pred_f1, digits=3))
    print(confusion_matrix(y_te, pred_f1))

    joblib.dump({"model": pipe,
                 "features": cols,
                 "category_map": {},          # not needed (pipeline embeds one-hot)
                 "top_specialties": top_specialties,
                 "target_mode": "early",
                 "drop_late": True,
                 "threshold_accuracy": float(thr_acc),
                 "threshold_f1": float(thr_f1)},
                ARTIFACT_PATH)
    print("\nreadmission_model.joblib remplacé: B+SMOTE (drop_late=True).")
    print("-> test_model.py / predict.py / app_streamlit.py / app_fastapi.py "
          "adoptent le nouveau modele tel quel.")


if __name__ == "__main__":
    main()
