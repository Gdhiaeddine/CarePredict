# -*- coding: utf-8 -*-
"""
SHAP explainability via LightGBM's native TreeSHAP (pred_contrib=True).

Produces:
  shap_summary.png        - global SHAP importance (mean |SHAP|) top-20
  shap_beeswarm.png       - summary scatter (top-10 features, sample)
  shap_patient_<i>.png    - per-plot contribution bars for 2 demo patients
  shap_importance.csv     - global importance table
Prints a short text explanation for 3 demo patients.

Run:  python shap_explain.py
"""

import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"


def split(df):
    groups = df["patient_nbr"]
    cols = [c for c in df.columns if c not in ("patient_nbr", "target")]
    X, y = df[cols], df["target"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    tr_idx, temp_idx = next(gss.split(X, y, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    v_rel, te_rel = next(gss2.split(X.iloc[temp_idx], groups=groups.iloc[temp_idx]))
    return X, y, cols, tr_idx, temp_idx[v_rel], temp_idx[te_rel]


def main():
    art = joblib.load(ARTIFACT_PATH)
    model = art["model"]
    feats = art["features"]

    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    df, _ = prepare_data(df_raw)
    X, y, cols, tr_idx, val_idx, te_idx = split(df)
    X_te = align_categories(X.iloc[te_idx][feats], art["category_map"])
    y_te = y[te_idx]

    # --------------------------------------------------- TreeSHAP values
    contrib = model.predict(X_te, pred_contrib=True)  # shape (n, k+1)
    shap_vals = contrib[:, :-1]   # per-feature SHAP
    base = contrib[:, -1].mean()  # expected value (log-odds)
    print(f"SHAP matrix: {shap_vals.shape}, expected value (log-odds) "
          f"{base:.3f} -> baseline prob {1/(1+np.exp(-base)):.3f}")

    # --------------------------------------------- 1. global importance
    glob = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-glob)[:20]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(range(len(order)), glob[order][::-1])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([feats[i] for i in order][::-1])
    ax.set_xlabel("mean |SHAP value| (impact on prediction)")
    ax.set_title("Global feature importance (TreeSHAP)")
    fig.tight_layout()
    fig.savefig("shap_summary.png", dpi=140)
    plt.close(fig)

    pd.DataFrame({"feature": [feats[i] for i in np.argsort(-glob)],
                  "mean_abs_shap": glob[np.argsort(-glob)]}
                 ).to_csv("shap_importance.csv", index=False)
    print("-> shap_summary.png / shap_importance.csv")

    # ------------------------------------------------ 2. beeswarm top-10
    order10 = np.argsort(-glob)[:10]
    rng = np.random.RandomState(RANDOM_STATE)
    n = min(len(X_te), 2000)
    idx = rng.choice(len(X_te), size=n, replace=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for rank, fi in enumerate(order10):
        vals = shap_vals[idx, fi]
        rangex = np.random.RandomState(fi).normal(0, 0.12, n) + (9 - rank)
        ax.scatter(vals, rangex, s=6, alpha=0.5)
    ax.set_yticks(range(10))
    ax.set_yticklabels([feats[i] for i in order10][::-1])
    ax.set_xlabel("SHAP value (x unit: log-odds of OUI readmission)")
    ax.set_title("SHAP summary (strip-approximation, top 10)")
    ax.axvline(0, color="k", lw=0.8)
    fig.tight_layout()
    fig.savefig("shap_beeswarm.png", dpi=140)
    plt.close(fig)
    print("-> shap_beeswarm.png")

    # ----------------------------------- 3. per-patient explanation bars
    opos = np.argsort(-y_te[y_te == 1].shape[0] * 0)  # noop placeholder
    prob_te = model.predict_proba(X_te)[:, 1]
    top_pos = np.argsort(-prob_te)[:2]   # 2 highest risk flagged patients
    figs = []
    for i, patient in enumerate(top_pos):
        sv = shap_vals[patient]
        o = np.argsort(-np.abs(sv))[:10]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#d9534f" if v > 0 else "#5bc0de" for v in sv[o]]
        ax.barh(range(len(o)), sv[o][::-1] / np.log(10) / len(str(o)),  # scale
                color=colors[::-1])
        # redraw properly scaled bars
        ax.cla()
        ax.barh(range(len(o)), sv[o][::-1], color=colors[::-1])
        ax.set_yticks(range(len(o)))
        ax.set_yticklabels([feats[j] for j in o][::-1], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("SHAP contribution (log-odds)")
        ax.set_title(f"Patient #{patient} - risk {prob_te[patient]:.0%} "
                     f"(true: {'OUI' if y_te[patient] else 'NON'})")
        fig.tight_layout()
        path = f"shap_patient_{i}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        figs.append(path)
    print("-> " + ", ".join(figs))

    # --------------------------------------------- 4. text explanations
    print("\n=== Text explanation for the 3 most-at-risk test patients ===")
    for patient in np.argsort(-prob_te)[:3]:
        sv = shap_vals[patient]
        o = np.argsort(-np.abs(sv))[:5]
        print(f"\npatient #{patient}: P(readmit)={prob_te[patient]:.2%} "
              f"(true label {int(y_te[patient])})")
        for j in o:
            direction = "PUSHD risk UP" if sv[j] > 0 else "pulls risk DOWN"
            feat_val = X_te.iloc[patient, j]
            print(f"  {feats[j]:<26} = {str(feat_val):<14} -> {sv[j]:+.3f} "
                  f"({direction})")


if __name__ == "__main__":
    main()
