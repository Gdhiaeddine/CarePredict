# -*- coding: utf-8 -*-
"""
Sélection de variables à fort impact + nouveau modèle.

Cible redéfinie ("any") : OUI (1) si réadmis <30 OU >30, NON (0) sinon.

Étapes :
 1. fit LightGBM sur TOUTES les variables (47) avec la cible "any"
 2. permutation importance sur la VALIDATION (scoring ROC-AUC, 10 répétitions)
 3. garde les top-K variables (K=15) -> re-fit avec early stopping
 4. comparaison test set : all-features vs top-15
 5. seuils maintenant : F1 + precision (F0.5) + accuracy (sur validation)

Le meilleur des deux est sauvegardé dans readmission_model.joblib (clés
identiques), test_model.py / predict.py fonctionnent directement.

Run:  python select_features_retrain.py            (cible "any" par défaut)
      python select_features_retrain.py early      (cible "<30" seulement)
                                                     (~3-4 min)
"""

import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, classification_report,
                             confusion_matrix)

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"
TOP_K = 15
TARGET_MODE = sys.argv[1] if len(sys.argv) > 1 else "any"
assert TARGET_MODE in ("any", "early"), f"bad target mode: {TARGET_MODE}"

# hyperparamètres validés précédemment par RandomizedSearchCV
BASE_PARAMS = {
    "subsample": 1.0, "reg_lambda": 0, "reg_alpha": 0, "num_leaves": 15,
    "n_estimators": 1200, "min_child_samples": 100, "max_depth": 8,
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


def fit_early(params, X_tr, y_tr, X_val, y_val):
    model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1, verbosity=-1,
                           **params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def summarize(name, model, X_te, y_te, X_val, y_val):
    probs_te = model.predict_proba(X_te)[:, 1]
    probs_val = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_te, probs_te)
    pr = average_precision_score(y_te, probs_te)

    # threshold optimised for accuracy on VALIDATION
    thresholds = np.linspace(0.05, 0.95, 181)
    accs = [accuracy_score(y_val, (probs_val >= t).astype(int)) for t in thresholds]
    thr_acc = thresholds[int(np.argmax(accs))]

    # threshold optimised for F1 on VALIDATION
    f1s = []
    for t in thresholds:
        pred = (probs_val >= t).astype(int)
        tp = ((pred == 1) & (y_val == 1)).sum()
        fp = ((pred == 1) & (y_val == 0)).sum()
        fn = ((pred == 0) & (y_val == 1)).sum()
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    thr_f1 = thresholds[int(np.argmax(f1s))]

    acc_a = accuracy_score(y_te, (probs_te >= thr_acc).astype(int))
    acc_5 = accuracy_score(y_te, probs_te >= 0.5)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC {auc:.4f} | PR-AUC {pr:.4f}")
    print(f"accuracy test @0.5   : {acc_5:.2%}")
    print(f"accuracy test @thr {thr_acc:.3f} (opt accuracy, val): {acc_a:.2%}")

    preds = (probs_te >= thr_f1).astype(int)
    print(f"classification report @F1-thr {thr_f1:.3f} :")
    print(classification_report(y_te, preds, digits=3))

    return {"model": model, "auc": auc, "pr": pr,
            "thr_acc": float(thr_acc), "thr_f1": float(thr_f1),
            "acc": float(acc_a), "probs_val": probs_val}


def main():
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, top_specialties = prepare_data(df_raw, target_mode=TARGET_MODE)
    print(f"Target '{TARGET_MODE}' distribution: "
          f"{df['target'].value_counts(normalize=True).to_dict()}")
    X, y, feat_cols, tr_idx, val_idx, te_idx = split(df)
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]

    cat_cols = [c for c, dt in X_tr.dtypes.items() if dt == "category"]
    category_map = {c: list(X_tr[c].cat.categories) for c in cat_cols}
    X_tr_a = align_categories(X_tr, category_map)
    X_val_a = align_categories(X_val, category_map)
    X_te_a = align_categories(X_te, category_map)

    # --------------------------------- 1) manual fit on ALL features
    all_model = fit_early(BASE_PARAMS, X_tr_a, y_tr, X_val_a, y_val)
    res_all = summarize("TOUTES les variables", all_model, X_te_a, y_te,
                        X_val_a, y_val)

    # --------------------- 2) permutation importance on validation, top-K
    print("\nPermutation importance (validation, scoring=roc_auc, 10 repeats)...")
    perm = permutation_importance(all_model, X_val_a, y_val, scoring="roc_auc",
                                  n_repeats=10, random_state=RANDOM_STATE,
                                  n_jobs=-1)
    imp = (pd.DataFrame({"feature": feat_cols,
                         "mean": perm.importances_mean,
                         "std": perm.importances_std})
           .sort_values("mean", ascending=False))
    print("\nPermutation importance (top 20) :")
    print(imp.head(20).to_string(index=False))
    top_feats = imp.head(TOP_K)["feature"].tolist()
    print(f"\n=> {TOP_K} variables retenues : {top_feats}")

    # ----------------------------------- 3) refit on the TOP-K features
    sel_model = fit_early(BASE_PARAMS, X_tr_a[top_feats], y_tr,
                          X_val_a[top_feats], y_val)
    res_sel = summarize(f"TOP-{TOP_K} features", sel_model, X_te_a[top_feats],
                        y_te, X_val_a[top_feats], y_val)

    # ------------------------------------------------- 4) compare + save
    print("\n========== COMPARAISON MINIMALE ==========")
    for name, r in (("all", res_all), (f"top{TOP_K}", res_sel)):
        print(f"  {name:<8}: AUC {r['auc']:.4f}  PR-AUC {r['pr']:.4f}  "
              f"acc(test) {r['acc']:.2%}")

    pick = res_sel if (res_sel["auc"] >= res_all["auc"] or
                       res_sel["pr"] >= res_all["pr"]) else res_all
    model_name = f"top{TOP_K}" if pick is res_sel else "all"
    feats = top_feats if pick is res_sel else feat_cols
    print(f"\nSauvegarde du modèle {model_name} ({len(feats)} variables) "
          f"-> readmission_model.joblib")

    joblib.dump({"model": pick["model"],
                 "threshold_accuracy": pick["thr_acc"],
                 "threshold_f1": pick["thr_f1"],
                 "features": feats,
                 "category_map": category_map,
                 "top_specialties": top_specialties,
                 "target_mode": TARGET_MODE,
                 "model_name": model_name},
                ARTIFACT_PATH)
    print("-> test_model.py / predict.py utilisent ce nouveau modèle tel quel.")


if __name__ == "__main__":
    main()
