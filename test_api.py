# -*- coding: utf-8 -*-
"""Smoke tests for app_fastapi.py using FastAPI TestClient."""
import pandas as pd
from fastapi.testclient import TestClient
import app_fastapi

client = TestClient(app_fastapi.app)

df = pd.read_csv("diabetic_data.csv", na_values="?", low_memory=False)
one = df.drop(columns=["readmitted"]).head(1).replace({pd.NA: None})
one = one.where(pd.notna(one), None).to_dict(orient="records")[0]

r = client.get("/health")
print("GET /health ->", r.status_code, r.json())

r = client.get("/model")
print("GET /model ->", r.status_code, "| n features:", len(r.json()["features"]))

r = client.post("/predict", json={"record": one})
print("POST /predict ->", r.status_code)
print("   ", r.json())

batch_df = df.drop(columns=["readmitted"]).head(5).astype(object)
batch_df = batch_df.where(pd.notna(batch_df), None)
batch = client.post("/predict/batch",
                    json={"patients": batch_df.to_dict(orient="records")})
print("POST /predict/batch ->", batch.status_code,
      "| n =", batch.json().get("n") if batch.status_code == 200 else "")


# check missing-field behaviour returns 422 with detail
if batch.status_code == 200:
    bad = dict(one)
    bad.pop("diag_1")
    r2 = client.post("/predict", json={"record": bad})
    print("POST /predict (champ manquant) ->", r2.status_code,
          "|", r2.json().get("detail", {}).get("error"))
