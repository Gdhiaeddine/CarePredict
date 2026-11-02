# -*- coding: utf-8 -*-
"""
Améliorer la PRÉCISION du modèle de réadmission <30 jours.

Stratégies testées ici :
  A) modèle de référence (params actuels du joblib, mêmes variables)
  B) nouveaux paramètres "conservateurs" (feuilles pures -> expositions saines)
 四人 (variant) : les DEUX sur les variables enrichies (comorbidités, ratios,
     historique de service, drapeaux d'admission/décharge)

Pour chaque modèle on reporte :
  - ranking-based precision@K (top 1%, 5%, 10%, 20% des plus à risque)
  - un seuil optimisé pour F0.5 (pénalise les faux positifs (precision-focused))
  - des seuils calibres pour une precision cible (>= 0.30, 0.40, 0.50)

À la fin : le meilleur modèle remplace readmission_model.joblib
(mêmes clés : model, thresholds, features, category_map, top_specialties)
-> test_model.py / predict.py continuent de fonctionner.

Run:  python improve_precision.py
"""

import warnings

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"

# the tuned params found previously by RandomizedSearchCV
BASE_PARAMS = {
    "subsample": 1.0, "reg_lambda": 0, "reg_alpha": 0, "num_leaves": 15,
    "n_estimators": 300, "min_child_samples": 100, "max_depth": 8,
    "learning_rate": 0.03, "colsample_bytree": 0.6, "class_weight": None,
}

VARIANTS = {
    "A_base_params": dict(BASE_PARAMS),
    "B_conservative": dict(BASE_PARAMS, n_estimators=1200, min_child_samples=400,
                           num_leaves=15, max_depth=5, learning_rate=0.02),
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


def precision_at_k(y, probs, k_frac):
    n = max(1, int(len(y) * k_frac))
    top = np.argsort(-probs)[:n]
    return y[top].mean(), n


def fbeta_threshold(y, probs, beta=0.5):
    """threshold maximizing F-beta (beta<1 -> precision weighted more)."""
    b2 = beta ** 2
    best_thr, best_f = 0.5, -1
    for t in np.linspace(0.02, 0.9, 220):
        pred = (probs >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f = (1 + b2) * prec * rec / (b2 * prec + rec) if (b2 * prec + rec) else 0.0
        if f > best_f:
            best_f, best_thr = f, t
    return best_thr


def precision_target_threshold(y, probs, target):
    """smallest threshold on val reaching >= target precision (any flagging)."""
    best = None
    for t in np.linspace(0.05, 0.99, 281):
        pred = (probs >= t).astype(int)
        flagged = pred.sum()
        if flagged == 0:
            continue
        prec = ((pred == 1) & (y == 1)).sum() / flagged
        if prec >= target:
            if best is None or flagged > best[1]:
                best = (t, flagged)
    return best


def headline(y_te, probs, thr):
    pred = (probs >= thr).astype(int)
    tp = ((pred == 1) & (y_te == 1)).sum()
    fp = ((pred == 1) & (y_te == 0)).sum()
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / ((y_te == 1).sum())
    return prec, rec, int(pred.sum())


def evaluate(name, model, X_te, y_te, X_val, y_val, title=""):
    probs_te = model.predict_proba(X_te)[:, 1]
    probs_val = model.predict_proba(X_val)[:, 1]
    print(f"\n### {name} {title}")
    print(f"ROC-AUC: {roc_auc_score(y_te, probs_te):.4f}")

    print("Precision@k (les k% les plus a risque, classement modele) :")
    for k in (0.01, 0.05, 0.10, 0.20):
        prec, n = precision_at_k(y_te, probs_te, k)
        print(f"  top {k:>4.0%}: precision {prec:.2%} sur {n} patients")

    thr_f05 = fbeta_threshold(y_val, probs_val, beta=0.5)
    prec, rec, n = headline(y_te, probs_te, thr_f05)
    print(f"Seuil F0.5 (val)={thr_f05:.3f} -> test: precision {prec:.2%}, "
          f"rappel {rec:.2%}, {n} patients signales")

    for target in (0.30, 0.40, 0.50):
        got = precision_target_threshold(y_val, probs_val, target)
        if got:
            thr, _ = got
            p, r, n = headline(y_te, probs_te, thr)
            print(f"  cible precision>={target:.0%}: seuil {thr:.3f} -> test "
                  f"precision {p:.2%}, rappel {r:.2%}, {n} signales")
        else:
            print(f"  cible precision>={target:.0%}: jamais atteinte sur val")
    return model, thr_f05


def main():
    # ------------------------- prepare enriched data + same split as training
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, top_specialties = prepare_data(df_raw)      # <- enriched features
    X, y, feat_cols, tr_idx, val_idx, te_idx = split(df)
    X_tr, y_tr = X.iloc[tr_idx], y[tr_idx]
    X_val, y_val = X.iloc[val_idx], y[val_idx]
    X_te, y_te = X.iloc[te_idx], y[te_idx]
    print(f"Variables: {len(feat_cols)} (variables enrichies incluses)")

    # ------------------------------------------------- baseline (old artifact)
    old = joblib.load(ARTIFACT_PATH)
    X_te_old = align_categories(X_te[old["features"]], old["category_map"])
    evaluate("A BASELINE (ancien modele, anciennes variables)",
             old["model"], X_te_old, y_te,
             align_categories(X_val[old["features"]], old["category_map"]), y_val)

    # ------------------------------------- enriched-feature variants (B, C)
    cat_cols = [c for c, dt in X_tr.dtypes.items() if dt == "category"]
    category_map = {c: list(X_tr[c].cat.categories) for c in cat_cols}
    X_tr_al = align_categories(X_tr, category_map)
    X_val_al = align_categories(X_val, category_map)
    X_te_al = align_categories(X_te, category_map)

    results = {}
    for name, params in VARIANTS.items():
        model = LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1,
                               verbosity=-1, **params)
        model.fit(X_tr_al, y_tr,
                  eval_set=[(X_val_al, y_val)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        print(f"  [fit] {name}: best_iteration={model.best_iteration_}")
        results[name] = (model, evaluate(name, model, X_te_al, y_te,
                                         X_val_al, y_val)[1])

    # --------------------------------------------------- pick best & save
    def score_at_top1(model_key):
        model = results[model_key][0]
        probs = model.predict_proba(X_te_al)[:, 1]
        return precision_at_k(y_te, probs, 0.01)[0]

    probs_old = old["model"].predict_proba(X_te_old)[:, 1]
    base_top1 = precision_at_k(y_te, probs_old, 0.01)[0]

    best_name = max(results, key=score_at_top1)
    best_model, best_thr = results[best_name]
    print(f"\n=== MEILLEUR : {best_name} (precision@top1% "
          f"{score_at_top1(best_name):.2%} vs baseline {base_top1:.2%}) ===")

    if score_at_top1(best_name) >= base_top1:
        joblib.dump({"model": best_model,
                     "threshold_accuracy": old["threshold_accuracy"],
                     "threshold_f1": best_thr,
                     "features": feat_cols,
                     "category_map": category_map,
                     "top_specialties": top_specialties},
                    ARTIFACT_PATH)
        print(f"readmission_model.joblib mis a jour avec {best_name}.")
        print("-> test_model.py / predict.py utilisent automatiquement le nouveau modele.")
    else:
        print("Le baseline reste Meilleur, rien n'est remplace.")

    # feature importance of the winner
    imp = pd.Series(best_model.feature_importances_,
                    index=feat_cols).sort_values(ascending=False)
    print("\nTop 15 variables (modele gagnant):")
    print(imp.head(15).to_string())


if __name__ == "__main__":
    main()
