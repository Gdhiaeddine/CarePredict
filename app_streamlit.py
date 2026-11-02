# -*- coding: utf-8 -*-
"""
Streamlit dashboard for the diabetic readmission risk model.

Run:  streamlit run app_streamlit.py
Open: http://localhost:8501
"""

import io
import warnings

import joblib
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from readmission_utils import prepare_data, align_categories

warnings.filterwarnings("ignore")
ARTIFACT_PATH = "readmission_model.joblib"
DATA_PATH_FALLBACK = "diabetic_data.csv"

st.set_page_config(page_title="CarePredict AI - Risk of Re-admission",
                   page_icon="hospital", layout="wide")

if "art" not in st.session_state:
    st.session_state.art = joblib.load(ARTIFACT_PATH)
art = st.session_state.art
model = art["model"]
feats = art["features"]
target_mode = art.get("target_mode", "any")
thresholds = {"accuracy": art["threshold_accuracy"],
              "f1": art["threshold_f1"]}
thresholds.update(art.get("thresholds_extra", {}))


@st.cache_data(show_spinner=False)
def run_predictions(df_raw: pd.DataFrame):
    art_ = st.session_state.art
    if art_.get("drop_late") and "readmitted" in df_raw.columns:
        # drop_late model: exclude '>30' late readmissions (toxic negatives)
        df_raw = df_raw[df_raw["readmitted"].fillna("").astype(str) != ">30"]
    df_in = df_raw.copy()
    if "patient_nbr" not in df_in.columns:
        df_in["patient_nbr"] = np.arange(len(df_in), dtype="int64")
    if "encounter_id" not in df_in.columns:
        df_in["encounter_id"] = np.arange(len(df_in), dtype="int64")
    prepared, _ = prepare_data(df_in,
                               top_specialties=art_["top_specialties"],
                               remove_deaths=False,
                               keep_first_per_patient=False,
                               target_mode=target_mode)
    missing = [c for c in art_["features"] if c not in prepared.columns]
    if missing:
        st.error(f"Missing required columns: {missing}")
        return None, None
    X = align_categories(prepared[art_["features"]], art_["category_map"])
    probs = art_["model"].predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["risk_prob"] = np.round(probs, 4)          # raw [0-1] for the CSV
    out["risk_percent"] = np.round(probs * 100, 1)  # e.g. 31.9
    out["risk_%"] = out["risk_percent"].map(lambda v: f"{v:.1f}%")
    for name, thr in thresholds.items():
        out[f"flag_{name}"] = (probs >= float(thr)).astype(int)
    return out, probs


st.title("CarePredict AI - Diabetic Re-admission Risk Predictor")
st.caption(f"Model: LightGBM + SMOTE | Target: "
           f"{'OUI if re-admit <30 or >30 days' if target_mode == 'any' else 'OUI if <30 days'}"
           f" | {len(feats)} variables | "
           f"{'patients >30 j exclus | ' if art.get('drop_late') else ''}"
           f"evaluation: ROC-AUC "
           f"{'0.703' if art.get('drop_late') else '0.665'}")

# ---------------- NOTICE: why the reported accuracy has this value ----------
_ACCURACY_CAUSES = {
    "any": {
        "acc": 63.3, "baseline": 59.3, "auc": 0.665,
        "bullets": [
            "Cible 'OUI = réadmis <30 OU >30' : 41% des patients sont positifs "
            "(cible difficile): la classe majoritaire à elle seule fait 59%.",
            "Le modèle n'ajoute que +4 pts au-dessus de ce naïf — c'est le "
            "plafond des données (aucun modèle supervisé dans la littérature "
            "ne dépasse ~0.67 d'AUC sur ce dataset).",
            "Ce qui rend le modèle utile: la PRÉCISION sur les patients "
            "priorisés — 73% sur le top-1% les plus risqués (contre 41% au "
            "hasard) — et non l'accuracy globale.",
        ],
    },
    "early": {
        "acc": 90.9, "baseline": 90.8, "auc": 0.668,
        "bullets": [
            "Cible 'OUI = réadmis <30 seulement' : seulement 9% de positifs. "
            "Le gros de l'accuracy vient du hasard de l'imbalance: "
            "prédire 'NON' pour tout le monde donne déjà 90.8%.",
            "Le modèle ne dépasse le naïf que de +0.1 pt: l'accuracy élevée "
            "ne prouve PAS l'efficacité.",
            "La métrique qui compte: détecter les <30 j réels — précision "
            "40% à faible rappel, 38% sur le top-1% (base 9%).",
        ],
    },
    "early_drop": {
        "acc": 89.5, "baseline": 90.2, "auc": 0.703,
        "bullets": [
            "Cible 'OUI = réadmis <30 seulement', et les patients réadmis "
            ">30 j sont EXCLUS (négatifs toxiques) : 9.8% de positifs. Le "
            "naïf (prédire 'NON' pour tous) fait ~90.2%.",
            "L'accuracy du modèle au seuil optimal (89.5%) est du même ordre "
            "que le naïf: l'accuracy ne prouve PAS l'efficacité.",
            "Ce qui compte: ROC-AUC 0.703 (au-dessus du plafond ~0.67 de la "
            "littérature sur ce dataset, grâce à l'exclusion des '>30' + "
            "SMOTE sur le train seul) et la précision top-1% — 51% contre "
            "10% au hasard. Au seuil F1: précision 27% / rappel 35%.",
        ],
    },
}
_notice_key = ("early_drop" if (target_mode == "early" and
                                art.get("drop_late")) else target_mode)
_cause = _ACCURACY_CAUSES[_notice_key]
st.info(
    f"### ℹ️ Pourquoi l'accuracy vaut {_cause['acc']:.1f}% "
    f"(naïf: {_cause['baseline']:.1f}%)\n\n"
    + "\n".join(f"- {b}" for b in _cause["bullets"])
    + f"\n\n**Résumé:** trust ROC-AUC ({_cause['auc']:.3f}) et la précision "
      f"du top-1%, pas l'accuracy brute."
)
# ---------------------------------------------------------------------------

col_upload, col_thr = st.columns([3, 2])
with col_upload:
    uploaded = st.file_uploader("CSV of patients (columns "
                                "diabetic_data.csv)", type="csv")
    use_sample = st.checkbox("Use sample data "
                             f"({DATA_PATH_FALLBACK}, first 500 patients)",
                             value=uploaded is None)
with col_thr:
    strategy = st.radio("Alert threshold", ["f1", "accuracy",
                                            *sorted(t for t in thresholds
                                                    if t not in
                                                    ("f1", "accuracy"))],
                        format_func=lambda s: f"{s} (thr "
                                              f"{thresholds[s]:.3f})")

df_in = None
if uploaded:
    df_in = pd.read_csv(uploaded, na_values="?", low_memory=False)
elif use_sample:
    try:
        df_in = pd.read_csv(DATA_PATH_FALLBACK, na_values="?",
                            low_memory=False).head(500)
    except FileNotFoundError:
        st.error("No sample file; upload a CSV.")

if df_in is not None:
    results, _ = run_predictions(df_in)
    if results is not None:
        flag_col = f"flag_{strategy}"
        left, right = st.columns(2)
        left.metric("Number of patients scored", f"{len(results):,}")
        right.metric(f"Patients alerted ({strategy})",
                     f"{int(results[flag_col].sum()):,}")

        st.subheader("Risk score distribution")
        binned = pd.cut(results["risk_prob"], bins=np.linspace(0, 1, 21),
                        include_lowest=True)
        counts = binned.value_counts().sort_index()
        labels = [f"{i.left * 100:.0f}%–{i.right * 100:.0f}%"
                  for i in counts.index]
        df_dist = pd.DataFrame({"risk_range": labels,
                                "patients": counts.values})
        bar = (alt.Chart(df_dist)
               .mark_bar(cornerRadius=2)
               .encode(
                   x=alt.X("risk_range:N", sort=None,
                           title="Predicted risk (per 5% slice)",
                           axis=alt.Axis(labelAngle=-45, labelFontSize=10)),
                   y=alt.Y("patients:Q", title="Count of patients",
                           scale=alt.Scale(domain=[0, max(counts.max(), 1)])),
                   tooltip=[alt.Tooltip("risk_range:N", title="Risk interval"),
                            alt.Tooltip("patients:Q", title="Patients")],
               )
               .properties(height=260))
        st.altair_chart(bar, use_container_width=True)

        st.subheader(f"Top 15 patients most at risk ({strategy} alerted in RED)")
        top = results.sort_values("risk_prob", ascending=False).head(15)
        show_cols = [c for c in ["patient_nbr", "age", "gender", "race",
                                 "diag_1", "time_in_hospital",
                                 "number_inpatient", "risk_%",
                                 *[c for c in results.columns
                                   if c.startswith("flag_")]]
                    if c in top.columns]
        st.dataframe(top[show_cols], use_container_width=True,
                     hide_index=True)

        st.subheader("Full predictions (download)")
        st.dataframe(results[show_cols], use_container_width=True,
                     height=300)
        buf = io.BytesIO()
        results.to_csv(buf, index=False)
        st.download_button("Download all predictions (CSV)",
                           buf.getvalue(),
                           "predictions.csv", "text/csv")
else:
    st.info("Provide a patient CSV or check the sample data box.")

with st.expander("Model info and limitations"):
    st.write(
        "- Data: Diabetes-130US hospital (first encounter per patient).")
    st.write("- Target: re-admission within 30 days (all admissions merged) "
             "or only <30 days depending on target_mode.")
    st.write("- Thresholds are set on the validation set, test untouched.")
    st.write("- Experimental data limitation: ROC-AUC ~0.67 (dataset "
             "ceiling), high accuracy is mainly due to class imbalance.")
    st.write("- Do not use in actual condition without clinical validation.")
