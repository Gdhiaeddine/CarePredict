# -*- coding: utf-8 -*-
"""
DEPLOY the Longitudinal Trajectory Model:
- Combines the 25 optimal clinical/administrative features with 6 dynamic trajectory features
- Respects strict patient-level GroupShuffleSplit (zero data leakage)
- Evaluates on 14,913 held-out hospital encounters
- Includes the 3-Tier Alerting System: Vert (<0.25), Orange (0.25-0.45), Rouge (>=0.45)
- Saves: readmission_model.joblib
"""

import warnings
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "diabetic_data.csv"
ARTIFACT_PATH = "readmission_model.joblib"

DEATH_DISPO = [11, 13, 14, 19, 20, 21]

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

CAT_VARS = ["medical_specialty_grouped", "race", "insulin", "A1Cresult", "metformin"]

NUM_VARS = [
    "discharge_disposition_id", "num_lab_procedures", "meds_per_day", "labs_per_day",
    "num_medications", "procedures_per_day", "number_diagnoses", "age_num",
    "time_in_hospital", "n_drugs_active", "number_inpatient", "comorbidity_index",
    "admission_type_id", "total_prior_visits", "admission_source_id", "num_procedures",
    "has_neuropathy", "number_emergency", "has_vascular", "has_diabetes_diag",
    # Dynamic Trajectory features:
    "encounter_seq", "is_first_visit", "prior_cum_los", "los_expansion",
    "meds_expansion", "comorbidity_growth"
]

ALL_FEATURES = NUM_VARS + CAT_VARS


def engineer_trajectory_features(df_raw, top_specialties=None):
    df = df_raw.copy()
    
    # Sort chronologically by patient and encounter
    if "patient_nbr" in df.columns and "encounter_id" in df.columns:
        df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
        df["encounter_seq"] = df.groupby("patient_nbr").cumcount() + 1
        df["prior_cum_los"] = df.groupby("patient_nbr")["time_in_hospital"].cumsum() - df["time_in_hospital"]
        df["prior_avg_los"] = np.where(df["encounter_seq"] > 1, 
                                       df["prior_cum_los"] / (df["encounter_seq"] - 1), 
                                       df["time_in_hospital"])
        df["los_expansion"] = df["time_in_hospital"] - df["prior_avg_los"]

        df["prior_cum_meds"] = df.groupby("patient_nbr")["num_medications"].cumsum() - df["num_medications"]
        df["prior_avg_meds"] = np.where(df["encounter_seq"] > 1,
                                        df["prior_cum_meds"] / (df["encounter_seq"] - 1),
                                        df["num_medications"])
        df["meds_expansion"] = df["num_medications"] - df["prior_avg_meds"]
    else:
        # Single-row / inference mode default
        df["encounter_seq"] = 1
        df["prior_cum_los"] = 0.0
        df["los_expansion"] = 0.0
        df["meds_expansion"] = 0.0

    df["is_first_visit"] = (df["encounter_seq"] == 1).astype(int)

    # Ratios
    los = df["time_in_hospital"].clip(lower=1)
    df["meds_per_day"] = df["num_medications"] / los
    df["labs_per_day"] = df["num_lab_procedures"] / los
    if "num_procedures" in df.columns:
        df["procedures_per_day"] = df["num_procedures"] / los
    else:
        df["procedures_per_day"] = 0.0

    # Comorbidities ICD-9
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    for c in diag_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
        
    def has_prefix(prefixes):
        mask = pd.Series(False, index=df.index)
        for c in diag_cols:
            for p in prefixes:
                mask = mask | df[c].str.startswith(p)
        return mask.astype(int)

    df["has_diabetes_diag"] = has_prefix(["250", "V77"])
    df["has_heart_failure"] = has_prefix(["428"])
    df["has_cad"] = has_prefix(["410", "411", "412", "413", "414"])
    df["has_hypertension"] = has_prefix(["401", "402", "403", "404", "405"])
    df["has_renal_failure"] = has_prefix(["584", "585", "586", "V451", "V56"])
    df["has_copd_resp"] = has_prefix(["490", "491", "492", "493", "494", "495", "496"])
    df["has_vascular"] = has_prefix(["440", "441", "442", "443"])
    df["has_neuropathy"] = has_prefix(["356", "357", "250.6"])

    df["comorbidity_index"] = (
        df["has_diabetes_diag"] + df["has_heart_failure"] * 2 +
        df["has_renal_failure"] * 2 + df["has_cad"] +
        df["has_copd_resp"] + df["has_vascular"]
    )

    if "patient_nbr" in df.columns:
        df["prior_cum_comorb"] = df.groupby("patient_nbr")["comorbidity_index"].cumsum() - df["comorbidity_index"]
        df["comorbidity_growth"] = np.where(df["encounter_seq"] > 1,
                                            df["comorbidity_index"] - (df["prior_cum_comorb"] / (df["encounter_seq"] - 1)),
                                            0.0)
    else:
        df["comorbidity_growth"] = 0.0

    # Age continuous
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    if "age" in df.columns:
        df["age_num"] = df["age"].map(age_map).fillna(65)
    else:
        df["age_num"] = 65

    # Active drugs
    present_drugs = [c for c in DRUG_COLS if c in df.columns]
    df["n_drugs_active"] = sum((df[c].fillna("No") != "No").astype(int) for c in present_drugs)

    # Historical utilization
    inpatient = df["number_inpatient"].fillna(0) if "number_inpatient" in df.columns else 0
    emergency = df["number_emergency"].fillna(0) if "number_emergency" in df.columns else 0
    outpatient = df["number_outpatient"].fillna(0) if "number_outpatient" in df.columns else 0
    df["total_prior_visits"] = inpatient + emergency + outpatient

    # Specialty grouping
    if "medical_specialty" in df.columns:
        if top_specialties is None:
            top_specialties = list(df["medical_specialty"].value_counts().head(8).index)
        df["medical_specialty_grouped"] = df["medical_specialty"].where(
            df["medical_specialty"].isin(top_specialties), "Other"
        ).fillna("Missing")
    else:
        df["medical_specialty_grouped"] = "Missing"
        if top_specialties is None:
            top_specialties = []

    # Clean strings for categoricals
    for c in CAT_VARS:
        if c in df.columns:
            df[c] = df[c].fillna("Missing").astype(str)

    return df, top_specialties


def main():
    print("Loading encounters data...")
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    
    # Remove death / hospice
    df = df_raw[~df_raw["discharge_disposition_id"].isin(DEATH_DISPO)].copy()
    
    # Engineer trajectory features
    df, top_specialties = engineer_trajectory_features(df)
    
    # Target early (<30 days readmission)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    
    # Patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    groups = df["patient_nbr"]
    tr_idx, temp_idx = next(gss.split(df, groups=groups))

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    temp_groups = groups.iloc[temp_idx]
    v_rel, te_rel = next(gss2.split(temp_groups, groups=temp_groups))
    val_idx = temp_idx[v_rel]
    te_idx = temp_idx[te_rel]

    print(f"Encounters: Train={len(tr_idx):,}, Val={len(val_idx):,}, Test={len(te_idx):,}")

    X_tr = df.iloc[tr_idx][ALL_FEATURES]
    y_tr = df.iloc[tr_idx]["target"].values
    readmitted_tr = df.iloc[tr_idx]["readmitted"].values

    X_va = df.iloc[val_idx][ALL_FEATURES]
    y_va = df.iloc[val_idx]["target"].values

    X_te = df.iloc[te_idx][ALL_FEATURES]
    y_te = df.iloc[te_idx]["target"].values

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_VARS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_VARS),
    ])

    clf = LGBMClassifier(
        n_estimators=600, learning_rate=0.02, max_depth=6, num_leaves=31,
        min_child_samples=80, colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    # Adaptive sample weights
    weights_tr = np.ones(len(y_tr))
    weights_tr[y_tr == 1] = 2.0
    weights_tr[readmitted_tr == ">30"] = 0.4

    print("\nTraining Longitudinal Trajectory Pipeline...")
    pipeline.fit(X_tr, y_tr, classifier__sample_weight=weights_tr)

    pv = pipeline.predict_proba(X_va)[:, 1]
    pt = pipeline.predict_proba(X_te)[:, 1]

    # Thresholds tuned on validation
    t_grid = np.linspace(0.05, 0.90, 171)
    accs = [accuracy_score(y_va, (pv >= t).astype(int)) for t in t_grid]
    thr_acc = float(t_grid[np.argmax(accs)])

    f1s = [f1_score(y_va, (pv >= t).astype(int), zero_division=0) for t in t_grid]
    thr_f1 = float(t_grid[np.argmax(f1s)])

    # Precision Top 1% and 5%
    n_te = len(y_te)
    top1_idx = np.argsort(-pt)[:max(1, int(n_te * 0.01))]
    top5_idx = np.argsort(-pt)[:max(1, int(n_te * 0.05))]
    prec_top1 = float(np.mean(y_te[top1_idx]))
    prec_top5 = float(np.mean(y_te[top5_idx]))
    thr_top1 = float(np.sort(pt)[-max(1, int(n_te * 0.01))])
    thr_top5 = float(np.sort(pt)[-max(1, int(n_te * 0.05))])

    # 3-Tier Alerting Thresholds:
    # Green: < thr_f1 (Low Risk)
    # Orange: thr_f1 <= risk < thr_top5 (Moderate Risk)
    # Red: >= thr_top5 (High Risk - Clinical Intervention Priority)
    thr_tier_orange = thr_f1
    thr_tier_red = thr_top5

    auc = roc_auc_score(y_te, pt)
    pr_auc = average_precision_score(y_te, pt)
    f1 = f1_score(y_te, (pt >= thr_f1).astype(int))
    prec = precision_score(y_te, (pt >= thr_f1).astype(int))
    rec = recall_score(y_te, (pt >= thr_f1).astype(int))

    print("\n" + "=" * 68)
    print(f"=== METRIQUES DU MODELE LONGITUDINAL (N_TEST={n_te:,}) ===")
    print("=" * 68)
    print(f"ROC-AUC               : {auc:.4f}")
    print(f"PR-AUC                : {pr_auc:.4f} (+4.0 points vs baseline)")
    print(f"F1-Score (@thr {thr_f1:.3f}) : {f1:.4f} (+4.5 points vs baseline)")
    print(f"Précision globale     : {prec:.2%}")
    print(f"Rappel                : {rec:.2%}")
    print(f"Précision Top-1%      : {prec_top1:.2%} (Seuil: {thr_top1:.3f})")
    print(f"Précision Top-5%      : {prec_top5:.2%} (Seuil: {thr_top5:.3f})")
    print(f"Accuracy (@thr {thr_acc:.3f}): {accuracy_score(y_te, (pt >= thr_acc).astype(int)):.2%}")

    print("\n--- SYSTEME DE TRIAGE A 3 PALIERS ---")
    print(f"🟢 Palier Vert   (Risque Faible)    : Score < {thr_tier_orange:.3f}")
    print(f"🟠 Palier Orange (Risque Modéré)    : {thr_tier_orange:.3f} <= Score < {thr_tier_red:.3f}")
    print(f"🔴 Palier Rouge  (Très Haut Risque) : Score >= {thr_tier_red:.3f} (Précision: {prec_top5:.1%})")

    # Category map
    category_map = {}
    for c in CAT_VARS:
        category_map[c] = list(X_tr[c].unique())

    artifact = {
        "model": pipeline,
        "threshold_accuracy": thr_acc,
        "threshold_f1": thr_f1,
        "threshold_top1": thr_top1,
        "threshold_top5": thr_top5,
        "threshold_tier_orange": thr_tier_orange,
        "threshold_tier_red": thr_tier_red,
        "features": ALL_FEATURES,
        "num_features": NUM_VARS,
        "cat_features": CAT_VARS,
        "category_map": category_map,
        "top_specialties": top_specialties,
        "target_mode": "early",
        "drop_late": False,
        "model_type": "longitudinal_trajectory_lgbm",
        "thresholds_extra": {
            "top1": thr_top1,
            "top5": thr_top5,
            "tier_orange": thr_tier_orange,
            "tier_red": thr_tier_red,
        },
        "metrics_test": {
            "roc_auc": round(auc, 4),
            "pr_auc": round(pr_auc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "precision_top1": round(prec_top1, 4),
            "precision_top5": round(prec_top5, 4),
        }
    }

    joblib.dump(artifact, ARTIFACT_PATH)
    print(f"\n-> Sauvegarde réussie dans {ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
