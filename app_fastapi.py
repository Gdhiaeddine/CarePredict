# -*- coding: utf-8 -*-
"""
FastAPI prediction endpoint for the diabetic readmission model.

Run (dev):     uvicorn app_fastapi:app --reload --port 8000
Swagger (UI):  http://localhost:8000/docs
Health check:  GET  http://localhost:8000/health
Model info:    GET  http://localhost:8000/model

POST /predict            single patient  (patient JSON object)
POST /predict/batch      list of patients ({ "patients": [ {...}, ... ] })

Expected fields: the ORIGINAL diabetic_data.csv columns for one encounter,
e.g. race, gender, age ("[70-80)"), admission_type_id,
discharge_disposition_id, admission_source_id, time_in_hospital,
num_lab_procedures, num_procedures, num_medications, number_outpatient,
number_emergency, number_inpatient, number_diagnoses, diag_1..diag_3,
max_glu_serum, A1Cresult, the 23 drug columns, change, diabetesMed.
('readmitted' is never required and never used at inference.)

Responses: risk_prob + flag_<strategy> + applied thresholds.
"""

import os
import sys
import threading
import time
import urllib.request

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from readmission_utils import prepare_data, align_categories

ARTIFACT_PATH = "readmission_model.joblib"

app = FastAPI(title="Diabetic Readmission Risk API",
              description="Prediction of Diabetic 30-day Readmission Risk + 3-Tier Clinical Triage",
              version="2.0.0")

# -------------------------------------------------------------
# Keep-Alive Self-Pinger (Prevents Render Free Tier from Sleeping)
# -------------------------------------------------------------
def _keep_awake_worker(url: str, interval_seconds: int = 300):
    target_url = url.rstrip("/") + "/health" if not url.endswith("/health") else url
    print(f"[Keep-Awake] Worker active for {target_url} (every {interval_seconds // 60} min)")
    time.sleep(30)  # Wait for startup
    while True:
        try:
            req = urllib.request.Request(target_url, headers={"User-Agent": "RenderKeepAlive/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                print(f"[Keep-Awake] Pinged {target_url} -> {resp.status}")
        except Exception as err:
            print(f"[Keep-Awake] Ping attempt notice: {err}")
        time.sleep(interval_seconds)

def init_keep_alive():
    app_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL")
    enabled = os.environ.get("KEEP_ALIVE", "true").lower() in ("true", "1", "yes")
    interval = int(os.environ.get("KEEP_ALIVE_INTERVAL", "300"))
    if app_url and enabled:
        t = threading.Thread(target=_keep_awake_worker, args=(app_url, interval), daemon=True)
        t.start()

@app.on_event("startup")
def on_startup():
    init_keep_alive()

try:
    ART = joblib.load(ARTIFACT_PATH)
    MODEL = ART["model"]
    FEATURES = ART["features"]
    CATEGORY_MAP = ART["category_map"]
    THRESHALLS = {"accuracy": ART["threshold_accuracy"],
                  "f1": ART["threshold_f1"]}
    THRESHALLS.update(ART.get("thresholds_extra", {}))
    TOP_SPECIALTIES = ART["top_specialties"]
    IP = "fastapi_import_ok"
except Exception as exc:  # fail fast on broken artifact
    sys.exit(f"cannot load {ARTIFACT_PATH}: {exc}")


class Patient(BaseModel):
    """Free-form patient record: any subset of diabetic_data.csv columns."""
    record: dict = Field(..., description="single encounter record fields")


class PatientBatch(BaseModel):
    patients: list[dict]


def _score(df_raw: pd.DataFrame) -> pd.DataFrame:
    if ART.get("drop_late") and "readmitted" in df_raw.columns:
        # drop_late model: exclude '>30' late readmissions (toxic negatives)
        df_raw = df_raw[df_raw["readmitted"].fillna("").astype(str) != ">30"]
    if "patient_nbr" in df_raw.columns:
        df_in = df_raw.copy()
        df_in["encounter_id"] = np.arange(len(df_in), dtype="int64")
    else:
        df_in = df_raw.copy()
        df_in["patient_nbr"] = np.arange(len(df_in), dtype="int64")
        df_in["encounter_id"] = np.arange(len(df_in), dtype="int64")
    prepared, _ = prepare_data(df_in, top_specialties=TOP_SPECIALTIES,
                               remove_deaths=False, keep_first_per_patient=False,
                               target_mode=ART.get("target_mode", "any"))
    missing = [c for c in FEATURES if c not in prepared.columns]
    if missing:
        raise HTTPException(
            422, {"error": "missing required fields", "missing": missing})
    X = align_categories(prepared[FEATURES], CATEGORY_MAP)
    probs = MODEL.predict_proba(X)[:, 1]
    out = df_raw.copy()
    out["risk_prob"] = np.round(probs, 4)
    for name, thr in THRESHALLS.items():
        out[f"flag_{name}"] = (probs >= float(thr)).astype(int)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "target_mode": ART.get("target_mode", "any"),
            "drop_late": bool(ART.get("drop_late", False)),
            "n_features": len(FEATURES)}


@app.get("/keep-alive")
def keep_alive_status():
    app_url = os.environ.get("RENDER_EXTERNAL_URL") or os.environ.get("APP_URL")
    return {
        "keep_alive_enabled": os.environ.get("KEEP_ALIVE", "true").lower() in ("true", "1", "yes"),
        "interval_seconds": int(os.environ.get("KEEP_ALIVE_INTERVAL", "300")),
        "detected_app_url": app_url or "Not set (set RENDER_EXTERNAL_URL or APP_URL in Render dashboard)",
        "mode": "auto-pinger every 5 minutes" if app_url else "standby (waiting for URL)"
    }


@app.get("/model")
def model_info():
    return {"model": type(MODEL).__name__,
            "features": FEATURES,
            "thresholds": THRESHALLS,
            "target_mode": ART.get("target_mode", "any"),
            "drop_late": bool(ART.get("drop_late", False))}


@app.post("/predict")
def predict_one(p: Patient):
    df = pd.DataFrame([p.record])
    got = _score(df)
    out = got.iloc[0]
    flags = {c: int(out[c]) for c in got.columns if c.startswith("flag_")}
    prob = float(out["risk_prob"])
    tier_orange = float(THRESHALLS.get("tier_orange", 0.325))
    tier_red = float(THRESHALLS.get("tier_red", 0.494))
    if prob >= tier_red:
        tier = "Rouge (Très Haut Risque - Action Prioritaire)"
    elif prob >= tier_orange:
        tier = "Orange (Risque Modéré - Suivi Ambulatoire)"
    else:
        tier = "Vert (Risque Faible - Sortie Standard)"
    return {"risk_prob": prob,
            "triage_tier": tier,
            "flags": flags,
            "required_features": FEATURES}


@app.post("/predict/batch")
def predict_batch(b: PatientBatch):
    if not b.patients:
        raise HTTPException(400, "empty 'patients' list")
    df = pd.DataFrame(b.patients)
    got = _score(df)
    flag_cols = [c for c in got.columns if c.startswith("flag_")]
    records = []
    for _, row in got.iterrows():
        records.append({"risk_prob": float(row["risk_prob"]),
                        "flags": {c: int(row[c]) for c in flag_cols}})
    return {"n": len(records), "results": records}
