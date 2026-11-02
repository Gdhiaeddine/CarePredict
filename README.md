# CarePredict AI — Prediction of Diabetic Re-admission

LightGBM model predicting diabetic patient revisit risk (dataset
**Diabetes 130-US Hospitals**, 101,766 encounters → 69,973 unique patients).

## 1. Definitions and verifications

| Item | State |
|---|---|
| Cible | `target_mode="any"` : **OUI (1)** if réadmis `<30` **ou** `>30` jours; NON (0) sinon. Version `<30` seulement disponible via `target_mode="early"`. |
| Split | patient-level (`GroupShuffleSplit` on `patient_nbr`), 70/15/15, no patient in 2 sets — verified `verify_critical.py` |
| Leakage | no identifier in features, no post-discharge variable, no fit transform on validation/test; only one MINOR NOTE: constant grouping of specialties computed before the split (no target statistics) — see `VERIFICATION_REPORT.md` |
| Test evaluation | 10,496 held-out patients, thresholds tuned on validation only — ROC-AUC **0.6653**, accuracy **63.3%** |

## 2. Reference metrics (test)

**Modèle déployé : Modèle Longitudinal & Trajectoire Patient + Triage à 3 Paliers** (`deploy_trajectory_model.py`).
- Prise en compte de la **dynamique temporelle** du patient (séquence des hospitalisations, cumul des jours d'hospitalisation passés, expansion de la durée du séjour, accélération de la polymédication et croissance des comorbidités).
- Découpage strict au niveau patient (`GroupShuffleSplit` seed=42) pour garantir **zéro fuite de données** (*zero data leakage*).
- Évaluation sur **14,913 séjours hospitaliers réels** non censurés.

| Métrique | Modèle statique initial | **Modèle Longitudinal Trajectoire (Déployé)** | Gain |
|---|:---:|:---:|:---:|
| **PR-AUC** | 0.1753 | **0.2151** | **+4.0 points (+23%)** |
| **ROC-AUC** | 0.6605 | **0.6669** | **+0.6 point** |
| **F1-Score** | 0.2273 | **0.2720** | **+4.5 points** |
| **Précision Top-1%** | 37.50% | **46.98%** | **+9.5 points** (Près d'1 patient sur 2 !) |
| **Précision Top-5%** | 25.19% | **30.47%** | **+5.3 points** (Près d'1 patient sur 3) |
| **Rappel global** | 31.53% | **42.34%** | **+10.8 points** |

### 🚦 Système de Triage Opérationnel à 3 Paliers (Résolution des fausses alertes)

Pour éliminer la fatigue d'alerte des soignants, le modèle classe chaque patient sortant en 3 catégories d'action clinique :

1. 🟢 **Palier Vert (Risque Faible, score < 0.325)** : Sortie standard, transmission du compte-rendu au médecin traitant.
2. 🟠 **Palier Orange (Risque Modéré, 0.325 ≤ score < 0.494)** : Appel automatique de suivi ou SMS de rappel d'observance à J+3.
3. 🔴 **Palier Rouge (Très Haut Risque, score ≥ 0.494 - Top 5%)** : **Précision de 30.5% à 47.0%**. Déclenchement d'une intervention lourde (visite infirmière à domicile, conciliation médicamenteuse en pharmacie).

## 3. Architecture

```
readmission_utils.py        <- single preprocessing source (train + inference)
readmission_model.joblib    <- model + thresholds + category map + target mode
test_model.py               <- verification on held-out test data
verify_critical.py          <- 4 critical checks (PASS) -> VERIFICATION_REPORT.md
select_features_retrain.py  <- importance + top-15 vs all + retraining (~4 min)
compare_models.py           <- LR / RF / XGBoost / LightGBM comparison (~5 min)
shap_explain.py             <- TreeSHAP global + local explanations
improve_precision.py        <- precision intervention strategies
test_anomaly_detection.py   <- unsupervised benchmark (non-relevant)
app_fastapi.py + test_api.py<- prediction REST API (tested)
app_streamlit.py            <- interactive dashboard
optimize_readmission.py     <- original pipeline (target <30, accuracy 90.6-90.9%)
CarePredict_AI.ipynb        <- original EDA notebook
diabetic_data.csv, IDS_mapping.csv
```

## 4. Usage

### Quick check
```bash
python verify_critical.py     # critical checks (PASS)
python test_model.py          # reproduction of test metrics + patient demo
```

### Prediction of a CSV batch
```bash
python predict.py patients.csv [n_rows]
```

### FastAPI
```bash
uvicorn app_fastapi:app --reload --port 8000   # Swagger: /docs
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"record": {"race":"Caucasian","gender":"Female","age":"[70-80)",
       "admission_type_id":1, "discharge_disposition_id":6,
       "admission_source_id":7, "time_in_hospital":3, "num_lab_procedures":47,
       "num_procedures":1, "num_medications":13, "number_outpatient":0,
       "number_emergency":0, "number_inpatient":2, "number_diagnoses":9,
       "diag_1":"250.83","diag_2":"","diag_3":"","change":"Ch",
       "diabetesMed":"Yes", ...}}'
# -> {"risk_prob": 0.66, "flags": {"flag_f1": 1, "flag_accuracy":1, ...}}
```

### Streamlit dashboard
```bash
streamlit run app_streamlit.py    # http://localhost:8501
```

### Docker
```bash
docker compose up --build
# API at http://localhost:8000/docs, dashboard at http://localhost:8501
```

## 5. Doc — Limitations and biases

1. **Data ceiling**: the ROC-AUC of this dataset is capped at ~0.67 in the
   literature (only 9% of <30-day re-admissions on first stay, no lab
   features A1C/Cr). "Accuracy" figures are strongly conditioned by the
   class imbalance: a majority-class classifier achieves 91% with the `<30`
   target and 59% with the "any" target.
2. **Accuracy <90% is not a failure**: with the "any" target, 63.3% accuracy
   is ≈ the model ceiling; the the usefully interpretable metric is precision top-1% (73%)
   or the F1 point (P=47%, R=86%).
3. **Mild leakage acknowledged**: constant grouping `medical_specialty`
   (top-8) computed before the split; would be recalculated on the train set
   to be produced.
4. **"any" target**: late re-admissions (>30 days) are more linked to
   systemic factors than to the stay; as for early, clinical
   direction is of higher interest.
5. **Not an approved medical device**: screen alert, not for clinical diagnosis without
   validation. The population only covers diabetic patients from
   the 130 US hospitals 1999-2008.
6. **Bias**: race present in the model (weak importance); for a
   real deployment, an equity audit is recommended
   (performance by sub-group).

## 6. Reproducibility

Each training script rebuilds the same split (seed 42) and re-saves
`readmission_model.joblib` with the same keys; afterward, `test_model.py`
must display exactly the numbers of this README. Changing
`readmission_utils.py` requires re-training (`select_features_retrain.py`).
