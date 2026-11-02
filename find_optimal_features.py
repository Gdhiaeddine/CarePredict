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

# Load base data and preprocess features
from experiment_optimization import (
    X_tr, y_tr, X_va, y_va, X_te, y_te, readmitted_tr,
    num_cols, cat_cols, preprocessor, X_tr_enc, X_va_enc, X_te_enc,
    evaluate_predictions
)

# Step 1: Feature Importance Ranking from the regularized LGBM model
weights = np.ones(len(y_tr))
weights[y_tr == 1] = 2.0
weights[readmitted_tr == ">30"] = 0.4

base_clf = LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=5, num_leaves=20,
                          min_child_samples=80, colsample_bytree=0.7, subsample=0.8,
                          reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1)
base_clf.fit(X_tr_enc, y_tr, sample_weight=weights)

# Map one-hot feature names back to original variables
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols)
all_encoded_names = list(num_cols) + list(cat_feature_names)

# Group importance by original column
feature_importances = base_clf.feature_importances_
var_importance = {}

for num_c in num_cols:
    idx = all_encoded_names.index(num_c)
    var_importance[num_c] = feature_importances[idx]

for cat_c in cat_cols:
    # Sum importances of all one-hot levels for this categorical variable
    cat_idxs = [i for i, name in enumerate(all_encoded_names) if name.startswith(cat_c + "_")]
    var_importance[cat_c] = sum(feature_importances[i] for i in cat_idxs)

ranking_df = pd.DataFrame([
    {"variable": k, "importance": v} for k, v in var_importance.items()
]).sort_values("importance", ascending=False).reset_index(drop=True)

print("=== CLASSEMENT DE TOUTES LES VARIABLES PAR IMPORTANCE ===")
print(ranking_df.to_string())

# Step 2: Test progressive subsets: Top 5, 10, 15, 20, 25, 30, 40, 50, all
subsets_to_test = [5, 10, 15, 20, 25, 30, 35, 40, 50, len(ranking_df)]
results = []

for k in subsets_to_test:
    selected_vars = ranking_df.head(k)["variable"].tolist()
    
    # Sub-columns
    sub_num = [c for c in num_cols if c in selected_vars]
    sub_cat = [c for c in cat_cols if c in selected_vars]
    
    sub_preprocessor = ColumnTransformer([
        ("num", StandardScaler(), sub_num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), sub_cat),
    ])
    
    X_tr_sub = sub_preprocessor.fit_transform(X_tr[selected_vars])
    X_va_sub = sub_preprocessor.transform(X_va[selected_vars])
    X_te_sub = sub_preprocessor.transform(X_te[selected_vars])
    
    clf = LGBMClassifier(n_estimators=500, learning_rate=0.02, max_depth=5, num_leaves=20,
                         min_child_samples=80, colsample_bytree=0.7, subsample=0.8,
                         reg_alpha=0.5, reg_lambda=2.0, random_state=42, n_jobs=-1, verbosity=-1)
    clf.fit(X_tr_sub, y_tr, sample_weight=weights)
    
    pv = clf.predict_proba(X_va_sub)[:, 1]
    pt = clf.predict_proba(X_te_sub)[:, 1]
    
    metrics = evaluate_predictions(f"Top {k} variables", y_te, pt, y_va, pv)
    metrics["Nb_Variables"] = k
    results.append(metrics)
    print(f"Tested Top {k} variables: ROC-AUC={metrics['ROC-AUC']}, PR-AUC={metrics['PR-AUC']}, F1={metrics['F1']}")

res_df = pd.DataFrame(results)
print("\n" + "="*85)
print("=== RÉSULTATS COMPARATIFS SELON LE NOMBRE DE VARIABLES (TEST RÉEL N=10,496) ===")
print("="*85)
print(res_df[["Nb_Variables", "ROC-AUC", "PR-AUC", "F1", "Precision", "Recall", "Top-1% Prec", "Top-5% Prec"]].to_string(index=False))

res_df.to_csv("feature_ablation_results.csv", index=False)
ranking_df.to_csv("all_variables_ranked.csv", index=False)
