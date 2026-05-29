import os
import json
import numpy as np
import pandas as pd
import onnxruntime as ort

LOCAL_MODEL_PATH = "mental_health_model.onnx"
LOCAL_DATA_PATH = "test_data.csv"

with open("model_features.json", "r") as f:
    FEATURE_ORDER = json.load(f)

CONDITIONS = ['Depression', 'Anxiety', 'Bipolar', 'ADHD', 'SubstanceRelated']

def test_model_response():
    """Prueba 1: Validar que el modelo responde con datos de entrada definidos."""
    assert os.path.exists(LOCAL_MODEL_PATH), "El archivo del modelo ONNX no está disponible para la prueba."

    session = ort.InferenceSession(LOCAL_MODEL_PATH)
    input_name = session.get_inputs()[0].name

    dummy_input = np.zeros((1, len(FEATURE_ORDER)), dtype=np.float32)
    raw_preds = session.run(None, {input_name: dummy_input})

    # CORRECCIÓN: Extraemos las probabilidades detalladas que están en el segundo output
    probabilities = raw_preds[1]
    
    assert len(probabilities) == len(CONDITIONS), "El número de salidas del modelo no coincide con las condiciones esperadas."

def test_model_performance_drift():
    """Prueba 2: Probar que la métrica no es menor a un valor límite establecido (75%)."""
    assert os.path.exists(LOCAL_DATA_PATH), "El archivo de datos de prueba no fue descargado por el pipeline."
    
    session = ort.InferenceSession(LOCAL_MODEL_PATH)
    input_name = session.get_inputs()[0].name
    
    df_test = pd.read_csv(LOCAL_DATA_PATH)
    X_test = df_test[FEATURE_ORDER].values.astype(np.float32)
    y_test = df_test[CONDITIONS].values
    
    correct_predictions = 0
    for i in range(len(X_test)):
        row = np.array([X_test[i]], dtype=np.float32)
        raw_preds = session.run(None, {input_name: row})
        
        # CORRECCIÓN: Tomamos las etiquetas directamente del primer output (raw_preds[0])
        # raw_preds[0] tiene la forma array([[0, 0, 1, 0, 0]]) -> extraemos la fila [0]
        y_pred_row = raw_preds[0][0] 
        
        # Comparamos el vector de predicciones binarias contra el vector real (y_test[i])
        if np.array_equal(y_pred_row, y_test[i]):
            correct_predictions += 1

    accuracy = correct_predictions / len(X_test)
    assert accuracy >= 0.75, f"El rendimiento del modelo cayó por debajo del límite permitido (75%). Accuracy actual: {accuracy:.2%}"