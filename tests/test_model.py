import os
import json
import numpy as np
import onnxruntime as ort
import pandas as pd

# Definición de rutas (ajusta si tu pipeline las descarga en otra ubicación)
LOCAL_MODEL_PATH = "mental_health_model.onnx"
LOCAL_DATA_PATH = "test_data.csv"

# 1. Definición exacta de las condiciones (Salidas)
CONDITIONS = ['Depression', 'Anxiety', 'Bipolar', 'ADHD', 'SubstanceRelated']

# 2. Carga dinámica del orden de características (Inputs)
# Esto asegura que coincida al 100% con las 20 columnas del entrenamiento
try:
    with open("model_features.json", "r") as f:
        FEATURE_ORDER = json.load(f)
except FileNotFoundError:
    # Fallback con el orden exacto y minúsculas/mayúsculas correctas de tu dataset
    FEATURE_ORDER = [
        'Age', 'Sex', 'insomnia', 'fatigue', 'panic', 'panic_sleep', 
        'social_withdrawal', 'mood_swings', 'anhedonia', 'rumination', 
        'appetite_change', 'concentration_issues', 'self_harm_ideation', 
        'irritability', 'withdrawal', 'anxiety_general', 'psychomotor', 
        'suicidal_thoughts', 'hopelessness', 'panic_physical'
    ]


def test_model_response():
    """Prueba 1: Validar que el modelo responde con datos de entrada definidos."""
    assert os.path.exists(LOCAL_MODEL_PATH), "El archivo del modelo ONNX no está disponible para la prueba."

    session = ort.InferenceSession(LOCAL_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    # Crear una entrada simulada con ceros usando las 20 características
    dummy_input = np.zeros((1, len(FEATURE_ORDER)), dtype=np.float32)
    raw_preds = session.run(None, {input_name: dummy_input})

    # CORRECCIÓN: El MultiOutputClassifier consolida las 5 respuestas binarias en raw_preds[0][0]
    predicciones_binarias = raw_preds[0][0]
    
    # Validamos que el vector resultante contenga las 5 salidas mapeadas
    assert len(predicciones_binarias) == len(CONDITIONS), (
        f"Se esperaban {len(CONDITIONS)} predicciones, pero se obtuvieron {len(predicciones_binarias)}."
    )


def test_model_performance_drift():
    """Prueba 2: Probar que la métrica no es menor a un valor límite establecido (75%)."""
    assert os.path.exists(LOCAL_DATA_PATH), "El archivo de datos de prueba no fue descargado por el pipeline."

    session = ort.InferenceSession(LOCAL_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    # Leer el archivo CSV cargado
    df_test = pd.read_csv(LOCAL_DATA_PATH)
    
    # Extraer matrices X (features) e y (targets reales) usando el orden correcto
    X_test = df_test[FEATURE_ORDER].values.astype(np.float32)
    y_test = df_test[CONDITIONS].values

    correct_predictions = 0
    for i in range(len(X_test)):
        # Preparar la fila para ONNX Runtime (Agregar dimensión de batch)
        row = np.array([X_test[i]], dtype=np.float32)
        raw_preds = session.run(None, {input_name: row})
        
        # CORRECCIÓN: Extraer el vector fila de predicciones [0, 1, 0, 0, 0]
        y_pred_row = raw_preds[0][0]
        
        # Evaluar coincidencia exacta del subset (todas las etiquetas correctas para el paciente)
        if np.array_equal(y_pred_row, y_test[i]):
            correct_predictions += 1

    # Calcular Exact Match Ratio
    accuracy = correct_predictions / len(X_test)
    
    # Aserción final del Taller (Límite del 75%)
    assert accuracy >= 0.75, (
        f"El rendimiento del modelo cayó por debajo del límite permitido (75%). Accuracy actual: {accuracy:.2%}"
    )