FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# the trained artifact must exist (readmission_model.joblib committed with the code)
EXPOSE 8000 8501

# default service: FastAPI. The dashboard is started by docker-compose.
CMD ["uvicorn", "app_fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
