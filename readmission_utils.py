# -*- coding: utf-8 -*-
"""
Shared preprocessing for the diabetic readmission project.
Used by optimize_readmission.py (training) and test_model.py / predict.py (inference).

IMPORTANT: keep this in sync with the training preprocessing - the model expects
exactly these features, in this exact form.
"""

import numpy as np
import pandas as pd

# The 23 medication columns in diabetic_data.csv
DRUG_COLS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone",
    "metformin-rosiglitazone", "metformin-pioglitazone",
]

# discharge_disposition_id values that mean death / hospice (can never be readmitted)
DEATH_DISPO = [11, 13, 14, 19, 20, 21]

TARGET_COL = "target"
ID_COLS = ["encounter_id", "patient_nbr"]


# ICD-9 code -> diagnosis group (same mapping used at training time)
def map_diagnosis(code):
    if pd.isna(code):
        return "Unknown"
    code = str(code)
    if code.startswith(("250", "V77")):
        return "Diabetes"
    if code.startswith(("E", "V")):
        return "Other"
    try:
        c = float(code.split("-")[0])
    except ValueError:
        return "Other"
    if 1 <= c < 140:
        return "Infectious"
    if 140 <= c < 240:
        return "Neoplasms"
    if 240 <= c < 280:
        return "Endocrine"
    if 280 <= c < 290:
        return "Blood"
    if 290 <= c < 320:
        return "Mental"
    if 320 <= c < 390:
        return "NervousSystem"
    if 390 <= c < 460:
        return "Circulatory"
    if 460 <= c < 520:
        return "Respiratory"
    if 520 <= c < 580:
        return "Digestive"
    if 580 <= c < 630:
        return "Genitourinary"
    if 630 <= c < 680:
        return "Pregnancy"
    if 680 <= c < 710:
        return "Skin"
    if 710 <= c < 740:
        return "Musculoskeletal"
    if 740 <= c < 760:
        return "Congenital"
    if 760 <= c < 780:
        return "Perinatal"
    if 780 <= c < 800:
        return "IllDefined"
    if 800 <= c <= 1000:
        return "Injury"
    return "Other"


def _engineer_features(df):
    """Row-level feature engineering: diagnosis groups, drug stats, age int."""
    df = df.copy()
    
    # Specific ICD-9 diagnosis flags
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

    for c, name in [("diag_1", "diag1_group"), ("diag_2", "diag2_group"),
                    ("diag_3", "diag3_group")]:
        if c in df.columns:
            df[name] = df[c].apply(map_diagnosis)
        else:
            df[name] = "Unknown"

    drugs = [c for c in DRUG_COLS if c in df.columns]
    df["n_drugs_used"] = sum((df[c].fillna("No") != "No").astype(int) for c in drugs)
    df["n_drugs_active"] = df["n_drugs_used"]
    df["n_drug_changes"] = sum(df[c].fillna("No").isin(["Up", "Down"]).astype(int)
                               for c in drugs)
    
    age_map = {
        "[0-10)": 5, "[10-20)": 15, "[20-30)": 25, "[30-40)": 35,
        "[40-50)": 45, "[50-60)": 55, "[60-70)": 65, "[70-80)": 75,
        "[80-90)": 85, "[90-100)": 95
    }
    if "age" in df.columns:
        df["age_num"] = df["age"].map(age_map).fillna(65)
        df["is_senior"] = df["age"].isin(["[70-80)", "[80-90)",
                                          "[90-100)"]).astype(int)
    else:
        df["age_num"] = 65
        df["is_senior"] = 0

    # ratios inside the stay
    los = df["time_in_hospital"].clip(lower=1)
    df["meds_per_day"] = df["num_medications"] / los
    df["labs_per_day"] = df["num_lab_procedures"] / los
    if "num_procedures" in df.columns:
        df["procedures_per_day"] = df["num_procedures"] / los
    else:
        df["procedures_per_day"] = 0.0

    # service utilization history
    inpatient = df["number_inpatient"].fillna(0) if "number_inpatient" in df.columns else 0
    emergency = df["number_emergency"].fillna(0) if "number_emergency" in df.columns else 0
    outpatient = df["number_outpatient"].fillna(0) if "number_outpatient" in df.columns else 0
    df["total_previous_visits"] = inpatient + emergency + outpatient
    df["total_prior_visits"] = df["total_previous_visits"]
    df["had_prior_inpatient"] = (inpatient > 0).astype(int)
    df["had_emergency"] = (emergency > 0).astype(int)

    # comorbidity count: distinct diagnosis groups across diag_1/2/3
    group_cols = ["diag1_group", "diag2_group", "diag3_group"]
    def _n_groups(row):
        g = [row[c] for c in group_cols if c in row.index]
        g = [x for x in g if x not in ("Unknown", "Other")]
        return len(set(g))
    if all(c in df.columns for c in group_cols):
        df["comorbidity_count"] = df.apply(_n_groups, axis=1)

    # emergency / urgent admission flag
    if "admission_type_id" in df.columns:
        df["emergency_admission"] = df["admission_type_id"].isin(
            [1, 2, 7]).astype(int)

    # Dynamic trajectory features
    if "patient_nbr" in df.columns and "encounter_id" in df.columns and len(df) > 1:
        df = df.sort_values(["patient_nbr", "encounter_id"])
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
        df["prior_cum_comorb"] = df.groupby("patient_nbr")["comorbidity_index"].cumsum() - df["comorbidity_index"]
        df["comorbidity_growth"] = np.where(df["encounter_seq"] > 1,
                                            df["comorbidity_index"] - (df["prior_cum_comorb"] / (df["encounter_seq"] - 1)),
                                            0.0)
    else:
        df["encounter_seq"] = 1
        df["prior_cum_los"] = 0.0
        df["los_expansion"] = 0.0
        df["meds_expansion"] = 0.0
        df["comorbidity_growth"] = 0.0

    df["is_first_visit"] = (df["encounter_seq"] == 1).astype(int)

    # discharged to skilled nursing facility / rehab
    if "discharge_disposition_id" in df.columns:
        df["discharged_to_snf"] = df["discharge_disposition_id"].isin(
            [3, 4]).astype(int)

    return df


def prepare_data(df, top_specialties=None, remove_deaths=True,
                 keep_first_per_patient=True, target_mode="early"):
    """Full preprocessing used for training (top_specialties=None -> full dataset).

    target_mode:
      "early" : target = 1 if readmitted within 30 days ("<30"), else 0
      "any"   : target = 1 if readmitted at all ("<30" OR ">30"), else 0

    For predicting new patients, pass the top_specialties list that was computed
    on the TRAINING data (saved in readmission_model.joblib) so the grouping
    of medical_specialty stays identical.
    """
    df = df.copy()
    if keep_first_per_patient:
        df = df.sort_values("encounter_id").drop_duplicates(
            subset="patient_nbr", keep="first")
    if target_mode == "early":
        if "readmitted" in df.columns:
            df["target"] = (df["readmitted"] == "<30").astype(int)
        else:
            df["target"] = -1  # inference-only payload: no label column
    elif target_mode == "any":
        if "readmitted" in df.columns:
            df["target"] = df["readmitted"].isin(["<30", ">30"]).astype(int)
        else:
            df["target"] = -1
    else:
        raise ValueError(f"unknown target_mode: {target_mode}")

    rows_removed_death = 0
    if remove_deaths:
        rows_removed_death = df["discharge_disposition_id"].isin(DEATH_DISPO).sum()
        df = df[~df["discharge_disposition_id"].isin(DEATH_DISPO)].copy()

    df = _engineer_features(df)

    drop_cols = [c for c in ["encounter_id", "weight", "payer_code",
                             "diag_1", "diag_2", "diag_3", "readmitted"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Dropped at training: constant columns over the full dataset. During
    # inference the MODEL feature list (not nunique of the new data) decides
    # the kept columns, so we only need to know the constant columns list.
    # Here: recompute for training data, ignore for new patients.
    protected_cols = [TARGET_COL, "patient_nbr", "encounter_seq", "is_first_visit",
                      "prior_cum_los", "los_expansion", "meds_expansion", "comorbidity_growth"]
    if top_specialties is None:
        const_cols = [c for c in df.columns if df[c].nunique(dropna=False) <= 1
                      and c not in protected_cols]
        df = df.drop(columns=const_cols)

    if "medical_specialty" in df.columns:
        if top_specialties is None:
            top_specialties = list(df["medical_specialty"].value_counts().head(8).index)
        df["medical_specialty"] = df["medical_specialty"].where(
            df["medical_specialty"].isin(top_specialties) |
            df["medical_specialty"].isna(), "Other").fillna("Missing")
        df["medical_specialty_grouped"] = df["medical_specialty"]
    else:
        df["medical_specialty"] = "Missing"
        df["medical_specialty_grouped"] = "Missing"
        if top_specialties is None:
            top_specialties = []

    if "race" in df.columns:
        df["race"] = df["race"].fillna("Missing")
    else:
        df["race"] = "Missing"
    if "age" in df.columns:
        if df["age"].dtype == object or str(df["age"].dtype).startswith("str"):
            df["age"] = df["age"].astype(str).str.strip("[]()").str.split("-").str[0]
            df["age"] = pd.to_numeric(df["age"], errors="coerce").fillna(65).astype(int)

    # categorical dtype cleanup (same "Unknown" fill as training)
    cat_cols = df.select_dtypes(include=["object", "str"]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != TARGET_COL]
    for c in cat_cols:
        cats = pd.Categorical(df[c])
        if "Unknown" not in cats.categories:
            cats = cats.add_categories("Unknown")
        df[c] = cats.fillna("Unknown")

    if rows_removed_death:
        print(f"prepare_data: removed {rows_removed_death} death/hospice rows")
    return df, list(top_specialties)


def categorical_features(feature_cols, excluded):
    """Names of categorical features among feature_cols (numeric = excluded set)."""
    return [c for c in feature_cols if c not in excluded]


def strip_late_rows(df_raw, df_prepared):
    """Companion function for `drop_late` models: remove patients whose raw
    `readmitted` value was '>30' - they are late readmissions that pollute
    the negative class of the early-readmission task."""
    if "readmitted" not in df_raw.columns or "patient_nbr" not in df_raw.columns:
        return df_prepared
    late = df_raw[df_raw["readmitted"] == ">30"][["patient_nbr"]]
    merged = df_prepared.merge(late.assign(_late=True), on="patient_nbr",
                               how="left")
    return merged[merged["_late"].isna()].drop(columns=["_late"])


def align_categories(df, category_map):
    """Cast the categorical columns of df to exactly the training categories
    so LightGBM sees identical category codes."""
    df = df.copy()
    for col, cats in category_map.items():
        if col not in df.columns:
            continue
        cats = list(cats)
        if "Unknown" not in cats:
            cats.append("Unknown")
        df[col] = pd.Categorical(df[col].astype(str).replace("nan", "Unknown"),
                                 categories=cats)
    return df
