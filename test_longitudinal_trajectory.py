import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

print("Loading raw encounters...")
df_raw = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)

# Remove deaths
DEATH_DISPO = [11, 13, 14, 19, 20, 21]
df = df_raw[~df_raw["discharge_disposition_id"].isin(DEATH_DISPO)].copy()

# Sort by patient and encounter_id to respect chronological order
df = df.sort_values(["patient_nbr", "encounter_id"]).reset_index(drop=True)
print(f"Total valid encounters: {len(df):,} across {df['patient_nbr'].nunique():,} unique patients")

# 1. Feature Engineering: Longitudinal & Trajectory Features
print("Computing longitudinal & trajectory features...")
df["encounter_seq"] = df.groupby("patient_nbr").cumcount() + 1  # 1st, 2nd, 3rd visit...
df["is_first_visit"] = (df["encounter_seq"] == 1).astype(int)

# Cumulative days spent in hospital across prior visits
df["prior_cum_los"] = df.groupby("patient_nbr")["time_in_hospital"].cumsum() - df["time_in_hospital"]
df["prior_avg_los"] = np.where(df["encounter_seq"] > 1, 
                               df["prior_cum_los"] / (df["encounter_seq"] - 1), 
                               df["time_in_hospital"])
df["los_expansion"] = df["time_in_hospital"] - df["prior_avg_los"]  # Is this stay longer than their normal?

# Prior cumulative medications and labs
df["prior_cum_meds"] = df.groupby("patient_nbr")["num_medications"].cumsum() - df["num_medications"]
df["prior_avg_meds"] = np.where(df["encounter_seq"] > 1,
                                df["prior_cum_meds"] / (df["encounter_seq"] - 1),
                                df["num_medications"])
df["meds_expansion"] = df["num_medications"] - df["prior_avg_meds"]

# Clinical ratio features
los = df["time_in_hospital"].clip(lower=1)
df["meds_per_day"] = df["num_medications"] / los
df["labs_per_day"] = df["num_lab_procedures"] / los
df["procedures_per_day"] = df["num_procedures"] / los

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

# Cumulative comorbidity trend
df["prior_cum_comorb"] = df.groupby("patient_nbr")["comorbidity_index"].cumsum() - df["comorbidity_index"]
df["comorbidity_growth"] = np.where(df["encounter_seq"] > 1,
                                    df["comorbidity_index"] - (df["prior_cum_comorb"] / (df["encounter_seq"] - 1)),
                                    0)

# Age
age_map = {
    "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
    "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
    "[80-90)": 85, "[90-100)": 95
}
df["age_num"] = df["age"].map(age_map).fillna(65)

# Active drugs
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]
present_drugs = [c for c in DRUG_COLS if c in df.columns]
df["n_drugs_active"] = sum((df[c].fillna("No") != "No").astype(int) for c in present_drugs)

# Utilization
inpatient = df["number_inpatient"].fillna(0)
emergency = df["number_emergency"].fillna(0)
outpatient = df["number_outpatient"].fillna(0)
df["total_prior_visits"] = inpatient + emergency + outpatient

# Specialty
top_spec = list(df["medical_specialty"].value_counts().head(8).index)
df["medical_specialty_grouped"] = df["medical_specialty"].where(
    df["medical_specialty"].isin(top_spec), "Other"
).fillna("Missing")

# Target
df["target"] = (df["readmitted"] == "<30").astype(int)

# Categoricals to string
CAT_VARS = ["medical_specialty_grouped", "race", "insulin", "A1Cresult", "metformin"]
for c in CAT_VARS:
    df[c] = df[c].fillna("Missing").astype(str)

NUM_VARS = [
    "discharge_disposition_id", "num_lab_procedures", "meds_per_day", "labs_per_day",
    "num_medications", "procedures_per_day", "number_diagnoses", "age_num",
    "time_in_hospital", "n_drugs_active", "number_inpatient", "comorbidity_index",
    "admission_type_id", "total_prior_visits", "admission_source_id", "num_procedures",
    "has_neuropathy", "number_emergency", "has_vascular", "has_diabetes_diag",
    # Trajectory features:
    "encounter_seq", "is_first_visit", "prior_cum_los", "los_expansion",
    "meds_expansion", "comorbidity_growth"
]

ALL_FEATURES = NUM_VARS + CAT_VARS
print(f"Total features with trajectory: {len(ALL_FEATURES)}")

# 2. Strict GroupSplit on patient_nbr (Zero Leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
groups = df["patient_nbr"]
tr_idx, temp_idx = next(gss.split(df, groups=groups))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
temp_groups = groups.iloc[temp_idx]
v_rel, te_rel = next(gss2.split(temp_groups, groups=temp_groups))
val_idx = temp_idx[v_rel]
te_idx = temp_idx[te_rel]

print(f"Encounters: Train={len(tr_idx):,}, Val={len(val_idx):,}, Test={len(te_idx):,}")
print(f"Test readmitted rate: {df['target'].iloc[te_idx].mean():.2%}")

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

X_tr_enc = preprocessor.fit_transform(X_tr)
X_va_enc = preprocessor.transform(X_va)
X_te_enc = preprocessor.transform(X_te)

# Weights
weights_tr = np.ones(len(y_tr))
weights_tr[y_tr == 1] = 2.0
weights_tr[readmitted_tr == ">30"] = 0.4

print("\nTraining LightGBM with Longitudinal Trajectory...")
clf = LGBMClassifier(n_estimators=600, learning_rate=0.02, max_depth=6, num_leaves=31,
                     min_child_samples=80, colsample_bytree=0.7, subsample=0.8,
                     reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1)
clf.fit(X_tr_enc, y_tr, sample_weight=weights_tr)

pv = clf.predict_proba(X_va_enc)[:, 1]
pt = clf.predict_proba(X_te_enc)[:, 1]

auc = roc_auc_score(y_te, pt)
pr_auc = average_precision_score(y_te, pt)

# Find optimal threshold on validation
t_grid = np.linspace(0.05, 0.90, 171)
f1s = [f1_score(y_va, (pv >= t).astype(int), zero_division=0) for t in t_grid]
best_thr = t_grid[np.argmax(f1s)]

preds = (pt >= best_thr).astype(int)
f1 = f1_score(y_te, preds)
prec = precision_score(y_te, preds)
rec = recall_score(y_te, preds)

n_te = len(y_te)
top1_idx = np.argsort(-pt)[:max(1, int(n_te * 0.01))]
top5_idx = np.argsort(-pt)[:max(1, int(n_te * 0.05))]
prec_top1 = np.mean(y_te[top1_idx])
prec_top5 = np.mean(y_te[top5_idx])

print("\n" + "="*70)
print(f"=== RESULTATS MODELE LONGITUDINAL TRAJECTOIRE (N_TEST={n_te:,}) ===")
print("="*70)
print(f"ROC-AUC               : {auc:.4f}")
print(f"PR-AUC                : {pr_auc:.4f}")
print(f"F1-Score (@thr {best_thr:.3f}) : {f1:.4f}")
print(f"Précision globale     : {prec:.2%}")
print(f"Rappel                : {rec:.2%}")
print(f"Précision Top-1%      : {prec_top1:.2%}")
print(f"Précision Top-5%      : {prec_top5:.2%}")

# Also test on repeat encounters only
mask_repeats = (df.iloc[te_idx]["encounter_seq"] > 1).values
if mask_repeats.sum() > 0:
    auc_rep = roc_auc_score(y_te[mask_repeats], pt[mask_repeats])
    pr_rep = average_precision_score(y_te[mask_repeats], pt[mask_repeats])
    print(f"\nSur les patients à visites répétées (N={mask_repeats.sum():,} encounters) :")
    print(f"  ROC-AUC : {auc_rep:.4f}")
    print(f"  PR-AUC  : {pr_rep:.4f}")
