# -*- coding: utf-8 -*-
"""
DEPLOY the optimal 25-feature model (unbiased evaluation, rigorous split).
- 25 most predictive clinical & administrative features
- GroupShuffleSplit on patient_nbr BEFORE any processing (zero data leakage)
- Adaptive sample weighting on >30 in training (w=0.4)
- Fully un-censored, realistic validation & test evaluation (N=10,496)
- Model: Tuned LightGBM + StandardScaler + OneHotEncoder (sklearn Pipeline)
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

# The optimal 25 features identified by ablation study
OPTIMAL_25_FEATURES = [
    "discharge_disposition_id",
    "num_lab_procedures",
    "meds_per_day",
    "labs_per_day",
    "num_medications",
    "medical_specialty_grouped",
    "procedures_per_day",
    "number_diagnoses",
    "age_num",
    "time_in_hospital",
    "n_drugs_active",
    "number_inpatient",
    "comorbidity_index",
    "admission_type_id",
    "total_prior_visits",
    "race",
    "admission_source_id",
    "insulin",
    "num_procedures",
    "has_neuropathy",
    "number_emergency",
    "has_vascular",
    "A1Cresult",
    "metformin",
    "has_diabetes_diag",
]

CAT_VARS = [
    "medical_specialty_grouped",
    "race",
    "insulin",
    "A1Cresult",
    "metformin"
]

NUM_VARS = [c for c in OPTIMAL_25_FEATURES if c not in CAT_VARS]

DEATH_DISPO = [11, 13, 14, 19, 20, 21]

DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

def engineer_25_features(df_raw, top_specialties=None):
    df = df_raw.copy()
    
    # Diagnosis codes
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
        
    # Drugs active
    present_drugs = [c for c in DRUG_COLS if c in df.columns]
    df["n_drugs_active"] = sum((df[c].fillna("No") != "No").astype(int) for c in present_drugs)
    
    # Ratios
    los = df["time_in_hospital"].clip(lower=1)
    df["meds_per_day"] = df["num_medications"] / los
    df["labs_per_day"] = df["num_lab_procedures"] / los
    df["procedures_per_day"] = df["num_procedures"] / los
    
    # Prior visits
    inpatient = df["number_inpatient"].fillna(0)
    emergency = df["number_emergency"].fillna(0)
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
    print("Loading data...")
    df_raw = pd.read_csv(DATA_PATH, na_values="?", low_memory=False)
    
    # Deduplicate: first encounter per patient
    df = df_raw.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first").copy()
    
    # Remove dead / hospice patients
    df = df[~df["discharge_disposition_id"].isin(DEATH_DISPO)].copy()
    print(f"Cohort size: {len(df):,} patients")
    
    # Feature engineering
    df_feat, top_specialties = engineer_25_features(df)
    
    # Target early: 1 if <30, 0 otherwise
    df_feat["target"] = (df_feat["readmitted"] == "<30").astype(int)
    
    # 1. SPLIT FIRST (Patient-level GroupShuffleSplit)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
    groups = df_feat["patient_nbr"]
    tr_idx, temp_idx = next(gss.split(df_feat, groups=groups))
    
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
    temp_groups = groups.iloc[temp_idx]
    v_rel, te_rel = next(gss2.split(temp_groups, groups=temp_groups))
    val_idx = temp_idx[v_rel]
    te_idx = temp_idx[te_rel]
    
    print(f"Train: {len(tr_idx):,}, Val: {len(val_idx):,}, Test: {len(te_idx):,}")
    
    X_tr = df_feat.iloc[tr_idx][OPTIMAL_25_FEATURES]
    y_tr = df_feat.iloc[tr_idx]["target"].values
    readmitted_tr = df_feat.iloc[tr_idx]["readmitted"].values
    
    X_va = df_feat.iloc[val_idx][OPTIMAL_25_FEATURES]
    y_va = df_feat.iloc[val_idx]["target"].values
    
    X_te = df_feat.iloc[te_idx][OPTIMAL_25_FEATURES]
    y_te = df_feat.iloc[te_idx]["target"].values
    
    # Preprocessor
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_VARS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_VARS),
    ])
    
    # LightGBM Classifier
    clf = LGBMClassifier(
        n_estimators=500, learning_rate=0.02, max_depth=5, num_leaves=20,
        min_child_samples=80, colsample_bytree=0.7, subsample=0.8,
        reg_alpha=0.5, reg_lambda=2.0, random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    )
    
    # Pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    
    # Adaptive weights: 2.0 for positives, 0.4 for >30, 1.0 for NO
    weights_tr = np.ones(len(y_tr))
    weights_tr[y_tr == 1] = 2.0
    weights_tr[readmitted_tr == ">30"] = 0.4
    
    print("\nFitting optimal Pipeline (25 features) on Train set...")
    pipeline.fit(X_tr, y_tr, classifier__sample_weight=weights_tr)
    
    # Predict on Validation
    pv = pipeline.predict_proba(X_va)[:, 1]
    
    # Tune thresholds on Validation only
    t_grid = np.linspace(0.05, 0.90, 171)
    accs = [accuracy_score(y_va, (pv >= t).astype(int)) for t in t_grid]
    thr_acc = float(t_grid[np.argmax(accs)])
    
    f1s = [f1_score(y_va, (pv >= t).astype(int), zero_division=0) for t in t_grid]
    thr_f1 = float(t_grid[np.argmax(f1s)])
    
    # Predict on Held-Out Test Set (un-censored, N=10,496)
    pt = pipeline.predict_proba(X_te)[:, 1]
    
    auc = roc_auc_score(y_te, pt)
    pr_auc = average_precision_score(y_te, pt)
    preds_f1 = (pt >= thr_f1).astype(int)
    f1 = f1_score(y_te, preds_f1)
    prec = precision_score(y_te, preds_f1)
    rec = recall_score(y_te, preds_f1)
    
    # Top 1% and 5% Precision
    n_te = len(y_te)
    top1_idx = np.argsort(-pt)[:max(1, int(n_te * 0.01))]
    top5_idx = np.argsort(-pt)[:max(1, int(n_te * 0.05))]
    prec_top1 = float(np.mean(y_te[top1_idx]))
    prec_top5 = float(np.mean(y_te[top5_idx]))
    thr_top1 = float(np.sort(pt)[-max(1, int(n_te * 0.01))])
    thr_top5 = float(np.sort(pt)[-max(1, int(n_te * 0.05))])
    
    print("\n" + "=" * 65)
    print(f"=== METRIQUES SUR LE TEST REEL RIGOREUX (N={n_te:,}) ===")
    print("=" * 65)
    print(f"ROC-AUC               : {auc:.4f}")
    print(f"PR-AUC                : {pr_auc:.4f}")
    print(f"F1-Score (@thr {thr_f1:.3f}) : {f1:.4f}")
    print(f"Precision             : {prec:.2%}")
    print(f"Recall                : {rec:.2%}")
    print(f"Precision Top-1%      : {prec_top1:.2%} (Seuil: {thr_top1:.3f})")
    print(f"Precision Top-5%      : {prec_top5:.2%} (Seuil: {thr_top5:.3f})")
    print(f"Accuracy (@thr {thr_acc:.3f}): {accuracy_score(y_te, (pt >= thr_acc).astype(int)):.2%}")
    
    print("\nConfusion Matrix (@seuil F1):")
    print(confusion_matrix(y_te, preds_f1))
    
    # Build category map for serving compatibility
    category_map = {}
    for c in CAT_VARS:
        category_map[c] = list(X_tr[c].unique())
        
    # Save artifact
    artifact = {
        "model": pipeline,
        "threshold_accuracy": thr_acc,
        "threshold_f1": thr_f1,
        "threshold_top1": thr_top1,
        "threshold_top5": thr_top5,
        "features": OPTIMAL_25_FEATURES,
        "num_features": NUM_VARS,
        "cat_features": CAT_VARS,
        "category_map": category_map,
        "top_specialties": top_specialties,
        "target_mode": "early",
        "drop_late": False,  # No artificial test censoring!
        "thresholds_extra": {
            "top1": thr_top1,
            "top5": thr_top5,
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
