import os
import json
import time
import numpy as np
import onnxruntime as ort
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Mental Health ONNX Model API")

# Configuración leída de las variables de entorno del despliegue
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "mi-proyecto-salud-mental-mlops")
ENVIRONMENT = os.getenv("ENV", "dev")  # Se configurará como 'dev' o 'prod'
MODEL_KEY = f"modelos/{ENVIRONMENT}/mental_health_model.onnx"
LOG_KEY = f"monitoreo/predicciones_{ENVIRONMENT}.txt"

LOCAL_MODEL_PATH = "/tmp/model.onnx"
s3_client = boto3.client("s3")

# Cargar el orden de las características
with open("model_features.json", "r") as f:
    FEATURE_ORDER = json.load(f)

CONDITIONS = ['Depression', 'Anxiety', 'Bipolar', 'ADHD', 'SubstanceRelated']

@app.on_event("startup")
def load_model():
    """Descarga el modelo desde el bucket S3 al iniciar el contenedor"""
    try:
        print(f"Descargando modelo desde S3: {MODEL_KEY}...")
        s3_client.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
        global session
        session = ort.InferenceSession(LOCAL_MODEL_PATH)
        print("Modelo ONNX cargado exitosamente en el contenedor.")
    except Exception as e:
        print(f"Error crítico al cargar el modelo: {str(e)}")
        raise RuntimeError("No se pudo inicializar la sesión del modelo ONNX.")

class PatientData(BaseModel):
    Age: float
    Sex: float
    insomnia: float
    fatigue: float
    panic: float
    panic_sleep: float
    social_withdrawal: float
    mood_swings: float
    anhedonia: float
    rumination: float
    appetite_change: float
    concentration_issues: float
    self_harm_ideation: float
    irritability: float
    withdrawal: float
    anxiety_general: float
    psychomotor: float
    suicidal_thoughts: float
    hopelessness: float
    panic_physical: float

def log_prediction_to_s3(input_data, predictions):
    """Requerimiento: Registra la petición y predicción en formato TXT en S3"""
    log_line = f"Timestamp: {time.time()} | Input: {input_data} | Output: {predictions}\n"
    try:
        try:
            existing_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=LOG_KEY)
            existing_content = existing_obj['Body'].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            existing_content = ""
        
        new_content = existing_content + log_line
        s3_client.put_object(Bucket=BUCKET_NAME, Key=LOG_KEY, Body=new_content.encode('utf-8'))
    except Exception as e:
        print(f"Error al escribir bitácora de monitoreo en S3: {e}")

@app.post("/predict")
def predict(data: PatientData):
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise HTTPException(status_code=500, detail="Modelo no disponible en el contenedor.")
    
    dict_data = data.dict()
    input_vector = [dict_data[feat] for feat in FEATURE_ORDER]
    input_array = np.array([input_vector], dtype=np.float32)
    
    input_name = session.get_inputs()[0].name
    raw_preds = session.run(None, {input_name: input_array})
    
    predictions = {}
    for idx, condition in enumerate(CONDITIONS):
        predictions[condition] = int(raw_preds[idx][0])
        
    log_prediction_to_s3(dict_data, predictions)
    return {"predictions": predictions}

@app.get("/health")
def health():
    return {"status": "healthy", "environment": ENVIRONMENT}