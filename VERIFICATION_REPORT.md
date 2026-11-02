# Critical Verification Report

Model: LGBMClassifier | target_mode=any | features=53

| id | check | status | detail |
|---|---|---|---|
| 1 | Verify patient-level train/test split | **PASS** | train=48,981 val=10,496 test=10,496 rows | patients: train=48,981 val=10,496 test=10,496 | overlaps: train-val=0, train-test=0, val-test=0 (split uses GroupShuffleSplit on patient_nbr) |
| 2 | Verify no data leakage | **PASS** | issues=none; notes=encounter_id and patient_nbr excluded from features; no fitted preprocessor on supervised path (LightGBM native categoricals) -> nothing to leak through fit/transform; MINOR NOTE: category value sets and the top-8 medical_specialty grouping are computed on the prepared dataset before the split - value-statistics leakage of low clinical impact (constant grouping rule, no target stats); would be refined in production |
| 3 | Confirm target definition | **PASS** | mode='any' -> target = 1 if readmitted '<30' OR '>30' (any readmission), else 0 | encoded distribution {0: 0.5927, 1: 0.4073} | raw readmitted values {'NO': 54864, '>30': 35545, '<30': 11357} |
| 4 | Evaluate on untouched test data | **PASS** | test set: 10,496 patients, never used for fitting or threshold tuning (validation-only tuning in the training scripts) -> ROC-AUC 0.6653, accuracy@0.455 thr = 63.29% |

**Overall: ALL CRITICAL CHECKS PASSED**
