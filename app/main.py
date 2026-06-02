import os
import json
from datetime import datetime
import numpy as np
import onnxruntime as ort
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Inicialización de la API FastAPI
app = FastAPI(
    title="Mental Health MLOps API",
    description="API para la clasificación de condiciones de salud mental usando un modelo MultiOutput XGBoost en formato ONNX.",
    version="1.0.0"
)

# 1. Configuraciones dinámicas basadas en variables de entorno del contenedor
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "mi-proyecto-salud-mental-mlops")
ENVIRONMENT = os.getenv("ENV", "dev")  # Recibe 'dev' o 'prod' según la rama del pipeline

MODEL_KEY = f"modelos/{ENVIRONMENT}/mental_health_model.onnx"
LOG_KEY = f"monitoreo/predicciones_{ENVIRONMENT}.txt"

LOCAL_MODEL_PATH = "/tmp/model.onnx"
s3_client = boto3.client("s3")

# Variable global para mantener la sesión de inferencia ONNX activa en memoria
session = None

# Definición estricta de las condiciones (Salidas de tu modelo)
CONDITIONS = ['Depression', 'Anxiety', 'Bipolar', 'ADHD', 'SubstanceRelated']

# 2. Cargar el orden exacto de las características desde el JSON del entrenamiento
try:
    with open("model_features.json", "r") as f:
        FEATURE_ORDER = json.load(f)
except FileNotFoundError:
    # Fallback manual en caso de que el JSON no esté en la raíz del contenedor
    FEATURE_ORDER = [
        'Age', 'Sex', 'insomnia', 'fatigue', 'panic', 'panic_sleep', 
        'social_withdrawal', 'mood_swings', 'anhedonia', 'rumination', 
        'appetite_change', 'concentration_issues', 'self_harm_ideation', 
        'irritability', 'withdrawal', 'anxiety_general', 'psychomotor', 
        'suicidal_thoughts', 'hopelessness', 'panic_physical'
    ]


# 3. Evento de inicio del contenedor (Startup)
@app.on_event("startup")
def load_model_from_s3():
    """ Descarga el modelo ONNX desde S3 al almacenamiento temporal del contenedor e inicializa la sesión. """
    global session
    try:
        print(f"[STARTUP] Entorno detectado: {ENVIRONMENT.upper()}")
        print(f"[STARTUP] Descargando modelo desde s3://{BUCKET_NAME}/{MODEL_KEY}...")
        
        # Descarga el archivo del modelo
        s3_client.download_file(BUCKET_NAME, MODEL_KEY, LOCAL_MODEL_PATH)
        
        # Inicializa ONNX Runtime Session
        session = ort.InferenceSession(LOCAL_MODEL_PATH)
        print("[STARTUP] Modelo ONNX cargado exitosamente en memoria.")
    except Exception as e:
        print(f"[STARTUP] ERROR CRÍTICO al inicializar el contenedor: {str(e)}")
        # Provoca un fallo controlado para que el orquestador (ECS/App Runner) sepa que el contenedor no está listo
        raise RuntimeError("No se pudo inicializar la sesión del modelo ONNX.")


# 4. Estructura de validación para los datos de entrada usando Pydantic
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


# 5. Función de monitoreo para S3 (Escribir en el TXT correspondiente)
def log_prediction_to_s3(input_data, predictions):
    """
    REQUERIMIENTO DEL TALLER: Descarga el archivo predicciones_dev.txt o predicciones_prod.txt,
    le concatena la nueva predicción realizada y lo actualiza en el bucket S3.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Formateo estructurado de la nueva línea de la bitácora
    log_line = f"Timestamp: {timestamp} | Input: {json.dumps(input_data)} | Output: {json.dumps(predictions)}\n"
    
    try:
        # Intenta obtener el contenido actual del archivo TXT en S3
        try:
            existing_obj = s3_client.get_object(Bucket=BUCKET_NAME, Key=LOG_KEY)
            existing_content = existing_obj['Body'].read().decode('utf-8')
        except s3_client.exceptions.NoSuchKey:
            # Si el archivo no existe en el bucket (primera petición), inicia vacío
            existing_content = ""
        
        # Concatena la nueva línea al historial existente
        new_content = existing_content + log_line
        
        # Sube el archivo modificado de vuelta a S3 de forma síncrona
        s3_client.put_object(Bucket=BUCKET_NAME, Key=LOG_KEY, Body=new_content.encode('utf-8'))
    except Exception as e:
        # Registramos el error en los logs del contenedor para no interrumpir la respuesta al usuario final
        print(f"[MONITORING ERROR] No se pudo escribir la bitácora en S3: {str(e)}")


# 6. Endpoint Principal: Inferencia (/predict)
@app.post("/predict", summary="Realiza predicciones múltiples de condiciones de salud mental")
def predict(data: PatientData):
    """
    Recibe las características del paciente en JSON, ejecuta la inferencia sobre el modelo
    MultiOutput de XGBoost empaquetado en ONNX, guarda la traza en S3 y retorna las predicciones binarias.
    """
    # Verificación de seguridad de carga del modelo
    if session is None or not os.path.exists(LOCAL_MODEL_PATH):
        raise HTTPException(status_code=500, detail="El modelo ONNX no está disponible en este contenedor.")
    
    try:
        # Convertir los datos de Pydantic a diccionario de Python
        dict_data = data.dict()
        
        # Mapear y ordenar las características basándonos estrictamente en el FEATURE_ORDER del entrenamiento
        input_vector = [dict_data[feat] for feat in FEATURE_ORDER]
        
        # Convertir a matriz NumPy de tipo float32 bidimensional (1, 20) requerido por ONNX
        input_array = np.array([input_vector], dtype=np.float32)
        
        # Obtener el nombre del nodo de entrada en el modelo ONNX
        input_name = session.get_inputs()[0].name
        
        # Ejecutar inferencia en ONNX Runtime
        raw_preds = session.run(None, {input_name: input_array})
        
        # EXPLICACIÓN E