# -*- coding: utf-8 -*-
"""
Validation rigoureuse de l'optimisation proposée (B+SMOTE vs A baseline).

Méthode:
  - 3 splits différents (seeds 42/7/123) pour éliminer le hasard d'un split
  - à chaque split, les DEUX modèles prédisent les MÊMES lignes de test B
    (les patients NO / <30) => métriques strictement comparables
  - métriques: ROC-AUC, PR-AUC, précision top-1%, accuracy@0.5, P/R @F1-thr
  - conclusion: somme des victoires + moyenne ± écart-type

Run:  python validate_b_smote.py   (~8-12 min)
"""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from readmission_utils import prepare_data
from test_drop_g30_smote import BASE_PARAMS, f1_threshold

warnings.filterwarnings("ignore")
DATA_PATH = "diabetic_data.csv"
SEEDS = [42, 7, 123]


def build_b(df_raw, df_all):
    df_late = df_raw[df_raw["readmitted"] == ">30"][["patient_nbr"]]
    df_b = (df_all.merge(df_late.assign(_late=True), on="patient_nbr",
                         how="left"))
    return df_b[df_b["_late"].isna()].drop(columns=["_late"])


def split_for(df, seed):
    groups = df["patient_nbr"]
    cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[cols], df["target"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    v_rel, te_rel = next(gss2.split(X.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    val_idx, te_idx = temp_idx[v_rel], temp_idx[te_rel]
    return cols, tr_idx, val_idx, te_idx


def fit_A(X_tr, y_tr, X_val, y_val, seed):
    """Baseline: LightGBM native categoricals (deployed pipeline)."""
    m = LGBMClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0, num_leaves=15,
                       n_estimators=800, min_child_samples=100, max_depth=8,
                       learning_rate=0.03, colsample_bytree=0.6,
                       random_state=seed, n_jobs=-1, verbosity=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def import_lgb():
    global lgb
    import lightgbm as lgb


def fit_B(X_tr, y_tr, X_val, y_val, seed):
    """B+SMOTE: one-hot + SMOTE(train only) + LightGBM."""
    num = [c for c in X_tr.columns if X_tr[c].dtype != "category"]
    cat = [c for c in X_tr.columns if X_tr[c].dtype == "category"]
    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    clf = LGBMClassifier(subsample=1.0, reg_lambda=0, reg_alpha=0,
                         num_leaves=15, n_estimators=800,
                         min_child_samples=100, max_depth=8,
                         learning_rate=0.03, colsample_bytree=0.6,
                         random_state=seed, n_jobs=-1, verbosity=-1)
    pipe = ImbPipeline([("pre", pre),
                        ("smote", SMOTE(random_state=seed, k_neighbors=5)),
                        ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    return pipe


def metrics_on(y_true, probs, y_val, probs_val, tag):
    thr = f1_threshold(y_val, probs_val)
    pred = (probs >= thr).astype(int)
    n = len(y_true)
    top1 = np.argsort(-probs)[:max(1, int(n * 0.01))]
    return {
        "variant": tag,
        "ROC-AUC": roc_auc_score(y_true, probs),
        "acc@0.5": accuracy_score(y_true, probs >= 0.5),
        "P@F1": precision_score(y_true, pred),
        "R@F1": recall_score(y_true, pred),
        "top1%prec": y_true[top1].mean(),
    }


def main():
    import lightgbm as lgb  # needed for early stopping callback in fit_A
    globals()["lgb"] = lgb

    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df_all, _ = prepare_data(df_raw, target_mode="early")
    df_b = build_b(df_raw, df_all)
    print(f"A data: {len(df_all):,} rows | B data (sans '>30'): {len(df_b):,} rows\n")

    rows = []
    for seed in SEEDS:
        cols, tr, va, te = split_for(df_all, seed)
        X_tr, y_tr = df_all.iloc[tr][cols], df_all.iloc[tr]["target"].values
        X_va, y_va = df_all.iloc[va][cols], df_all.iloc[va]["target"].values
        X_te, y_te = df_all.iloc[te][cols], df_all.iloc[te]["target"].values

        cols_b, trb, vab, teb = split_for(df_b, seed)

        mA = fit_A(X_tr, y_tr, X_va, y_va, seed)
        pva = mA.predict_proba(X_va)[:, 1]

        # B model fits on B's train, but is evaluated on the SAME B-test rows
        Xb_tr, yb_tr = df_b.iloc[trb][cols_b], df_b.iloc[trb]["target"].values
        Xb_va, yb_va = df_b.iloc[vab][cols_b], df_b.iloc[vab]["target"].values
        Xb_te, yb_te = df_b.iloc[teb][cols_b], df_b.iloc[teb]["target"].values
        mB = fit_B(Xb_tr, yb_tr, Xb_va, yb_va, seed)
        pvb = mB.predict_proba(Xb_va)[:, 1]

        # --- comparaison sur les MÊMES lignes (B-test rows, subset de A-test)
        mask = X_te.index.isin(Xb_te.index)
        y_same = y_te[mask]
        X_same = X_te[mask]
        pA_same = mA.predict_proba(X_same)[:, 1]
        pB_same = mB.predict_proba(X_same)[:, 1]

        rA = metrics_on(y_same, pA_same, y_va, pva, f"A baseline")
        rB = metrics_on(y_same, pB_same, yb_va, pvb, f"B+SMOTE")
        rA["seed"] = rB["seed"] = seed
        rA["n_same"] = rB["n_same"] = int(mask.sum())
        rows += [rA, rB]
        print(f"seed {seed} (n_same={mask.sum():,}):")
        print(f"   A baseline: AUC {rA['ROC-AUC']:.4f} PRtop? {rA['top1%prec']:.2%} "
              f"acc {rA['acc@0.5']:.2%} P {rA['P@F1']:.2%} R {rA['R@F1']:.2%} top1 {rA['top1%prec']:.2%}")
        print(f"   B+SMOTE   : AUC {rB['ROC-AUC']:.4f} "
              f"acc {rB['acc@0.5']:.2%} P {rB['P@F1']:.2%} R {rB['R@F1']:.2%} top1 {rB['top1%prec']:.2%}\n")

    res = pd.DataFrame(rows)
    agg = (res.groupby("variant")[["ROC-AUC", "acc@0.5", "P@F1", "R@F1",
                                   "top1%prec"]]
           .agg(["mean", "std"]))
    print("=== MOYENNE ± ECART-TYPE (3 seeds, mêmes lignes de test) ===")
    print(agg.round(4).to_string())

    wins = sum(
        (res[res.variant == "B+SMOTE"][m].values >
         res[res.variant == "A baseline"][m].values).sum()
        for m in ["ROC-AUC", "P@F1", "R@F1", "top1%prec"])
    print(f"\nVictoires B+SMOTE sur 12 comparaisons : {wins}/12")
    if wins >= 9:
        print("VERDICT: OPTIMISATION CONFIRMEE - déployer B+SMOTE")
    elif wins >= 7:
        print("VERDICT: gain marginalement réel - à tester avant déploiement")
    else:
        print("VERDICT: PAS de gain robuste - garder le baseline")


if __name__ == "__main__":
    main()
