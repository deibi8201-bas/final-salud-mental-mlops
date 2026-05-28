import os
from google.cloud import storage

def download_blob():
    # Detecta el bucket desde las variables de entorno que inyecta GitHub/Cloud Run
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    
    print(f"Conectando al bucket: {bucket_name} usando credenciales del entorno...")
    
    # Inicializa el cliente buscando las credenciales automáticas (OIDC)
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # ASEGURAR CARPETA: Crea la carpeta 'src' si no existe en el entorno del pipeline
    os.makedirs("src", exist_ok=True)
    
    # 1. Descargar el modelo ONNX real
    print("Descargando mental_health_model.onnx...")
    blob_modelo = bucket.blob("mental_health_model.onnx")
    blob_modelo.download_to_filename("src/mental_health_model.onnx")
    
    # 2. Descargar los datos de prueba (.csv) exigidos por el taller
    print("Descargando test_data.csv...")
    blob_datos = bucket.blob("test_data.csv")
    blob_datos.download_to_filename("test_data.csv")
    
    print("Descarga de assets completada con éxito.")

if __name__ == "__main__":
    download_blob()