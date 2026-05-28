import os
import pytest
import numpy as np
import pandas as pd
import onnxruntime as ort
from sklearn.metrics import accuracy_score

MODEL_FILE = "src/mental_health_model.onnx"
DATA_FILE = "test_data.csv"

@pytest.fixture(scope="module")
def onnx_session():
    """Fixture que levanta la sesión del modelo ONNX una sola vez para los tests."""
    # Verificar que el modelo exista antes de iniciar las pruebas
    assert os.path.exists(MODEL_FILE), f"No se encontró el archivo {MODEL_FILE} para la prueba."
    return ort.InferenceSession("src/mental_health_model.onnx")

def test_model_dimensions_and_responsiveness(onnx_session):
    """
    PRUEBA 1: Verificar la consistencia de las dimensiones de entrada y
    que el modelo genere respuestas válidas (Multi-label array de tamaño 5).
    """
    input_meta = onnx_session.get_inputs()[0]
    input_name = input_meta.name
    
    # Simular una entrada dummy válida de 1 fila por 20 columnas (Float32)
    dummy_input = np.ones((1, 20), dtype=np.float32)
    
    # Ejecutar inferencia básica
    outputs = onnx_session.run(None, {input_name: dummy_input})
    
    # Validaciones rigurosas
    assert len(outputs) > 0, "El modelo no retornó ninguna salida."
    
    shape_salida = outputs[0].shape
    assert shape_salida == (1, 5), f"Se esperaba una salida de forma (1, 5), pero se obtuvo {shape_salida}"
    print(f"\n[TEST] Validación de dimensiones exitosa. Forma obtenida: {shape_salida}")

def test_model_performance_metric(onnx_session):
    """
    PRUEBA 2: Evaluar la métrica de rendimiento (Accuracy) usando los datos
    de prueba sintéticos para garantizar que el modelo mantiene su precisión.
    """
    # Verificar si el archivo de datos de prueba está presente
    assert os.path.exists(DATA_FILE), f"Falta el archivo {DATA_FILE} requerido para evaluar la métrica."
    
    # Cargar el dataset de prueba tabular
    df = pd.read_csv(DATA_FILE)
    
    # Separar las primeras 20 columnas (características) y las últimas 5 (condiciones reales)
    # De acuerdo al script de entrenamiento: X son 20 columnas, y son las últimas 5 correspondientes a CONDITIONS
    X_test = df.iloc[:, :20].values.astype(np.float32)
    y_true = df.iloc[:, 20:25].values.astype(np.int64)
    
    input_name = onnx_session.get_inputs()[0].name
    predictions = []
    
    # Ejecutar inferencia en bloque para cada fila del set de pruebas
    for row in X_test:
        out = onnx_session.run(None, {input_name: np.array([row])})
        # Guardar el vector binario [0, 1, 0, 0, 0] arrojado por el clasificador
        predictions.append(out[0][0])
        
    y_pred = np.array(predictions, dtype=np.int64)
    
    # Calcular el Subset Accuracy (Exact Match Ratio obligatorio en MultiOutput)
    exact_match_acc = accuracy_score(y_true, y_pred)
    
    # Definir el umbral mínimo exigido por el negocio/taller (75%)
    UMBRAL_MINIMO = 0.75
    
    print(f"\n[TEST] Subset Accuracy obtenido en el set de pruebas: {exact_match_acc:.2%}")
    assert exact_match_acc >= UMBRAL_MINIMO, (
        f"Alerta de degradación de modelo: Precisión de {exact_match_acc:.4f} "
        f"por debajo del umbral mínimo permitido ({UMBRAL_MINIMO})"
    )