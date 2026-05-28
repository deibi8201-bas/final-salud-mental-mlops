import os
import sys
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from src.utils import log_prediction_to_gcs

# Inicializar la aplicación FastAPI adaptando el título al entorno dinámico
env_name = os.environ.get("ENV", "dev").upper()
app = FastAPI(title=f"Mental Health Multi-Label API - Entorno {env_name}")

MODEL_FILE = "src/mental_health_model.onnx"
CONDICIONES = ['Depression', 'Anxiety', 'Bipolar', 'ADHD', 'SubstanceRelated']

# Verificar que el modelo esté presente antes de levantar el servidor
if not os.path.exists(MODEL_FILE):
    print(f"ERROR CRÍTICO: No se encontró '{MODEL_FILE}'. Ejecute download_assets.py primero.")
    sys.exit(1)

# Cargar el modelo ONNX en memoria global una sola vez para máxima eficiencia
try:
    session = ort.InferenceSession(MODEL_FILE)
    input_name = session.get_inputs()[0].name
    print(f"¡Modelo ONNX '{MODEL_FILE}' cargado con éxito para la API!")
except Exception as e:
    print(f"Error fatal al inicializar la sesión del modelo ONNX: {e}")
    sys.exit(1)


class MentalHealthPayload(BaseModel):
    features: list[float]

    # Validar que ingresen exactamente las 20 características clínicas que el modelo requiere
    @field_validator('features')
    @classmethod
    def check_dimensions(cls, value):
        if len(value) != 20:
            raise ValueError(f"El modelo requiere exactamente 20 parámetros numéricos. Recibidos: {len(value)}")
        return value


@app.get("/")
def health_check():
    """Ruta básica para verificar que el contenedor responda en la nube."""
    return {"status": "healthy", "environment": os.environ.get("ENV", "dev")}


@app.post("/predict")
def predict_conditions(payload: MentalHealthPayload):
    try:
        # Convertir la lista a una matriz bidimensional (1, 20) de tipo Float32 requerida por ONNX
        input_matrix = np.array([payload.features], dtype=np.float32)
        
        # Ejecutar la inferencia en los árboles de decisión de XGBoost
        outputs = session.run(None, {input_name: input_matrix})
        
        # outputs[0][0] contiene el arreglo de tamaño 5 con las salidas binarias o continuas
        raw_predictions = outputs[0][0]
        
        # Mapear los resultados con sus respectivas etiquetas de diagnóstico
        diagnosticos_detectados = []
        for idx, valor in enumerate(raw_predictions):
            # Si el valor es 1 (o mayor a un umbral si el convertidor exportó probabilidades)
            if valor >= 1:
                diagnosticos_detectados.append(CONDICIONES[idx])
        
        # Si no se detectó ninguna de las 5 patologías, se considera un estado saludable
        if not diagnosticos_detectados:
            diagnosticos_detectados.append("Normal / Estable / Sin Riesgo Detectado")
            
        max_score = float(np.max(raw_predictions))
        
        # Guardar en segundo plano la persistencia del log de auditoría en Google Cloud Storage
        log_prediction_to_gcs(payload.features, diagnosticos_detectados, max_score)
        
        return {
            "diagnosticos": diagnosticos_detectados,
            "salida_cruda_modelo": raw_predictions.tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error durante el proceso de inferencia: {str(e)}")