import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMClassifier
import joblib

# 1. Load data
print("Loading data...")
df_raw = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)

# Keep first encounter per patient
df = df_raw.sort_values("encounter_id").drop_duplicates(subset="patient_nbr", keep="first").copy()

# Remove deaths
DEATH_DISPO = [11, 13, 14, 19, 20, 21]
df = df[~df["discharge_disposition_id"].isin(DEATH_DISPO)].copy()

print(f"Total cohort: {len(df):,} patients")

# 2. Enhanced Feature Engineering
def engineer_clinical_features(df):
    df = df.copy()
    
    # Comorbidity ICD-9 flags across diag_1, diag_2, diag_3
    diag_cols = [c for c in ["diag_1", "diag_2", "diag_3"] if c in df.columns]
    for c in diag_cols:
        df[c] = df[c].fillna("Unknown").astype(str)
        
    def has_code(df, prefixes):
        mask = pd.Series(False, index=df.index)
        for c in diag_cols:
            for p in prefixes:
                mask = mask | df[c].str.startswith(p)
        return mask.astype(int)
    
    # Clinical comorbidities linked to diabetes readmission
    df["has_diabetes_diag"] = has_code(df, ["250", "V77"])
    df["has_heart_failure"] = has_code(df, ["428"])
    df["has_cad"] = has_code(df, ["410", "411", "412", "413", "414"])
    df["has_hypertension"] = has_code(df, ["401", "402", "403", "404", "405"])
    df["has_renal_failure"] = has_code(df, ["584", "585", "586", "V451", "V56"])
    df["has_copd_resp"] = has_code(df, ["490", "491", "492", "493", "494", "495", "496"])
    df["has_vascular"] = has_code(df, ["440", "441", "442", "443"])
    df["has_neuropathy"] = has_code(df, ["356", "357", "250.6"])
    
    # Comorbidity index sum
    df["comorbidity_index"] = (df["has_diabetes_diag"] + df["has_heart_failure"]*2 + 
                               df["has_renal_failure"]*2 + df["has_cad"] + 
                               df["has_copd_resp"] + df["has_vascular"])
    
    # Age to integer mid-point
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    df["age_num"] = df["age"].map(age_map).fillna(65)
    df["is_elderly"] = (df["age_num"] >= 75).astype(int)
    
    # Drug columns
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
    df["n_drugs_dosage_changed"] = sum(df[c].fillna("No").isin(["Up", "Down"]).astype(int) for c in present_drugs)
    df["insulin_treatment"] = (df["insulin"].fillna("No") != "No").astype(int)
    df["insulin_modified"] = df["insulin"].fillna("No").isin(["Up", "Down"]).astype(int)
    
    # Clinical ratios & hospital stay metrics
    los = df["time_in_hospital"].clip(lower=1)
    df["meds_per_day"] = df["num_medications"] / los
    df["labs_per_day"] = df["num_lab_procedures"] / los
    df["procedures_per_day"] = df["num_procedures"] / los
    df["high_med_intensity"] = (df["num_medications"] > 15).astype(int)
    
    # Historical healthcare utilization
    df["prior_inpatient"] = df["number_inpatient"].fillna(0)
    df["prior_emergency"] = df["number_emergency"].fillna(0)
    df["prior_outpatient"] = df["number_outpatient"].fillna(0)
    df["total_prior_visits"] = df["prior_inpatient"] + df["prior_emergency"] + df["prior_outpatient"]
    df["frequent_inpatient"] = (df["prior_inpatient"] >= 2).astype(int)
    df["any_prior_hospitalization"] = (df["prior_inpatient"] > 0).astype(int)
    
    # Interaction: prior inpatient + high comorbidity
    df["high_risk_profile"] = ((df["prior_inpatient"] > 0) & (df["comorbidity_index"] >= 2)).astype(int)
    
    # Lab & test indicators
    df["a1c_tested"] = (df["A1Cresult"].fillna("None") != "None").astype(int)
    df["a1c_uncontrolled"] = (df["A1Cresult"].fillna("None") == ">8").astype(int)
    df["glu_uncontrolled"] = (df["max_glu_serum"].fillna("None") == ">300").astype(int)
    
    # Specialty top 8
    top_spec = list(df["medical_specialty"].value_counts().head(8).index)
    df["medical_specialty_grouped"] = df["medical_specialty"].where(
        df["medical_specialty"].isin(top_spec), "Other"
    ).fillna("Missing")
    
    # Admission & discharge categoricals
    df["emergency_admission"] = df["admission_type_id"].isin([1, 2, 7]).astype(int)
    df["discharged_to_home"] = df["discharge_disposition_id"].isin([1]).astype(int)
    df["discharged_to_facility"] = df["discharge_disposition_id"].isin([3, 4, 5, 6]).astype(int)
    
    return df

df_feat = engineer_clinical_features(df)

# Target: 1 if <30, 0 otherwise
df_feat["target_early"] = (df_feat["readmitted"] == "<30").astype(int)
# 3-class target: 0 = NO, 1 = >30, 2 = <30
class_map = {"NO": 0, ">30": 1, "<30": 2}
df_feat["target_3class"] = df_feat["readmitted"].map(class_map)

# Feature columns
drop_cols = ["encounter_id", "patient_nbr", "weight", "payer_code", "age",
             "diag_1", "diag_2", "diag_3", "readmitted", "medical_specialty",
             "target_early", "target_3class"]
feature_cols = [c for c in df_feat.columns if c not in drop_cols]

# 3. SPLIT FIRST (GroupShuffleSplit on patient_nbr - zero leakage)
gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
groups = df_feat["patient_nbr"]
tr_idx, temp_idx = next(gss.split(df_feat, groups=groups))

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
temp_groups = groups.iloc[temp_idx]
v_rel, te_rel = next(gss2.split(temp_groups, groups=temp_groups))
val_idx = temp_idx[v_rel]
te_idx = temp_idx[te_rel]

print(f"Splits: Train={len(tr_idx):,}, Val={len(val_idx):,}, Test={len(te_idx):,}")
print(f"Test class balance: {pd.Series(df_feat['target_early'].iloc[te_idx]).value_counts(normalize=True).to_dict()}")

X_tr = df_feat.iloc[tr_idx][feature_cols].copy()
y_tr = df_feat.iloc[tr_idx]["target_early"].values
readmitted_tr = df_feat.iloc[tr_idx]["readmitted"].values

X_va = df_feat.iloc[val_idx][feature_cols].copy()
y_va = df_feat.iloc[val_idx]["target_early"].values

X_te = df_feat.iloc[te_idx][feature_cols].copy()
y_te = df_feat.iloc[te_idx]["target_early"].values

# Categorical & Numerical handling
cat_cols = [c for c in feature_cols if X_tr[c].dtype.kind in ('O', 'S', 'U') or isinstance(X_tr[c].dtype, pd.StringDtype) or X_tr[c].dtype.name == 'category']
num_cols = [c for c in feature_cols if c not in cat_cols]

for col in cat_cols:
    X_tr[col] = X_tr[col].fillna("Missing").astype(str)
    X_va[col] = X_va[col].fillna("Missing").astype(str)
    X_te[col] = X_te[col].fillna("Missing").astype(str)

from sklearn.preprocessing import OrdinalEncoder
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])

print(f"Number of num features: {len(num_cols)}, cat features: {len(cat_cols)}")
print("Fitting preprocessor...")
X_tr_enc = preprocessor.fit_transform(X_tr)
X_va_enc = preprocessor.transform(X_va)
X_te_enc = preprocessor.transform(X_te)
print(f"Total encoded features: {X_tr_enc.shape[1]}")

def evaluate_predictions(name, y_true, probs, y_val_true=None, probs_val=None):
    auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    
    # Threshold from validation if available, else standard grid
    if y_val_true is not None and probs_val is not None:
        t_grid = np.linspace(0.02, 0.90, 177)
        f1s = [f1_score(y_val_true, (probs_val >= t).astype(int), zero_division=0) for t in t_grid]
        best_thr = t_grid[np.argmax(f1s)]
    else:
        best_thr = 0.15
        
    preds = (probs >= best_thr).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    prec = precision_score(y_true, preds, zero_division=0)
    rec = recall_score(y_true, preds, zero_division=0)
    
    # Top-1% and Top-5% precision (lift)
    n = len(y_true)
    top1_idx = np.argsort(-probs)[:max(1, int(n * 0.01))]
    top5_idx = np.argsort(-probs)[:max(1, int(n * 0.05))]
    prec_top1 = np.mean(y_true[top1_idx])
    prec_top5 = np.mean(y_true[top5_idx])
    
    return {
        "Model": name,
        "ROC-AUC": round(auc, 4),
        "PR-AUC": round(pr_auc, 4),
        "Threshold (Val)": round(best_thr, 3),
        "F1": round(f1, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "Top-1% Prec": round(prec_top1, 4),
        "Top-5% Prec": round(prec_top5, 4),
    }

results = []

# --- EXPERIMENT 1: Baseline LightGBM on Full Train (target early) ---
print("Running Exp 1: Standard LGBM...")
clf1 = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31,
                      min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
                      random_state=42, n_jobs=-1, verbosity=-1)
clf1.fit(X_tr_enc, y_tr)
pv1 = clf1.predict_proba(X_va_enc)[:, 1]
pt1 = clf1.predict_proba(X_te_enc)[:, 1]
results.append(evaluate_predictions("1. LGBM standard (tout le train)", y_te, pt1, y_va, pv1))

# --- EXPERIMENT 2: LGBM with class_weight='balanced' ---
print("Running Exp 2: Balanced LGBM...")
clf2 = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31,
                      min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
                      class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
clf2.fit(X_tr_enc, y_tr)
pv2 = clf2.predict_proba(X_va_enc)[:, 1]
pt2 = clf2.predict_proba(X_te_enc)[:, 1]
results.append(evaluate_predictions("2. LGBM class_weight balanced", y_te, pt2, y_va, pv2))

# --- EXPERIMENT 3: Soft sample weights for >30 (sample_weight: 1 for <30, 1 for NO, 0.3 for >30) ---
print("Running Exp 3: Weighted >30...")
weights = np.ones(len(y_tr))
weights[readmitted_tr == ">30"] = 0.3
clf3 = LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=6, num_leaves=31,
                      min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
                      random_state=42, n_jobs=-1, verbosity=-1)
clf3.fit(X_tr_enc, y_tr, sample_weight=weights)
pv3 = clf3.predict_proba(X_va_enc)[:, 1]
pt3 = clf3.predict_proba(X_te_enc)[:, 1]
results.append(evaluate_predictions("3. LGBM pondération douce >30 (w=0.3)", y_te, pt3, y_va, pv3))

# --- EXPERIMENT 4: Drop >30 ONLY from Train (train purely on <30 vs NO) ---
print("Running Exp 4: Pure Train sans >30...")
mask_tr = (readmitted_tr != ">30")
X_tr_pure_enc = X_tr_enc[mask_tr]
y_tr_pure = y_tr[mask_tr]
clf4 = LGBMClassifier(n_estimators=400, learning_rate=0.03, max_depth=6, num_leaves=31,
                      min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
                      scale_pos_weight=2.0, random_state=42, n_jobs=-1, verbosity=-1)
clf4.fit(X_tr_pure_enc, y_tr_pure)
pv4 = clf4.predict_proba(X_va_enc)[:, 1]
pt4 = clf4.predict_proba(X_te_enc)[:, 1]
results.append(evaluate_predictions("4. LGBM train pur sans >30 + scale_pos", y_te, pt4, y_va, pv4))

# --- EXPERIMENT 5: 3-Class classification (0=NO, 1=>30, 2=<30), evaluate P(class=2) ---
print("Running Exp 5: Multiclass...")
y_tr_3 = df_feat.iloc[tr_idx]["target_3class"].values
clf5 = LGBMClassifier(n_estimators=300, learning_rate=0.03, max_depth=6, num_leaves=31,
                      min_child_samples=50, colsample_bytree=0.8, subsample=0.8,
                      objective="multiclass", num_class=3, random_state=42, n_jobs=-1, verbosity=-1)
clf5.fit(X_tr_enc, y_tr_3)
pv5 = clf5.predict_proba(X_va_enc)[:, 2]
pt5 = clf5.predict_proba(X_te_enc)[:, 2]
results.append(evaluate_predictions("5. LGBM Multi-classes (P(<30j))", y_te, pt5, y_va, pv5))

# --- EXPERIMENT 6: Tuned LightGBM with Regularization & Adaptive Weights ---
print("Running Exp 6: Fine-tuned Regularized...")
clf6 = LGBMClassifier(n_estimators=600, learning_rate=0.02, max_depth=5, num_leaves=24,
                      min_child_samples=100, colsample_bytree=0.6, subsample=0.8,
                      reg_alpha=0.5, reg_lambda=2.0,
                      random_state=42, n_jobs=-1, verbosity=-1)
weights6 = np.ones(len(y_tr))
weights6[y_tr == 1] = 2.5
weights6[readmitted_tr == ">30"] = 0.5
clf6.fit(X_tr_enc, y_tr, sample_weight=weights6)
pv6 = clf6.predict_proba(X_va_enc)[:, 1]
pt6 = clf6.predict_proba(X_te_enc)[:, 1]
results.append(evaluate_predictions("6. LGBM Optimisé (Régul + Poids adaptatifs)", y_te, pt6, y_va, pv6))

# --- EXPERIMENT 7: CatBoost / Gradient Boosting Alternative ---
from sklearn.ensemble import HistGradientBoostingClassifier
print("Running Exp 7: HistGradientBoosting...")
hgb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, max_depth=6,
                                     min_samples_leaf=50, random_state=42)
hgb.fit(X_tr_enc, y_tr)
pvh = hgb.predict_proba(X_va_enc)[:, 1]
pth = hgb.predict_proba(X_te_enc)[:, 1]
results.append(evaluate_predictions("7. HistGradientBoosting", y_te, pth, y_va, pvh))

# Print summary
res_df = pd.DataFrame(results)
print("\n" + "="*85)
print("=== RÉSULTATS COMPARATIFS SUR TEST RIGOREUX NON-BIAISÉ (N=10,265) ===")
print("="*85)
print(res_df.to_string(index=False))
res_df.to_csv("optimization_benchmarks.csv", index=False)
