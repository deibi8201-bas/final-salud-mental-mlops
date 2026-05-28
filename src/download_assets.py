import os
from google.cloud import storage

def download_blob():
    # Toma el nombre del bucket de tus secretos: mi-proyecto-mlops-mental-health
    bucket_name = os.getenv("GCP_BUCKET_NAME")
    
    print(f"Conectando al bucket: {bucket_name}...")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Crea la carpeta local 'src' si no existe en la máquina virtual de GitHub
    os.makedirs("src", exist_ok=True)
    
    # 1. CORRECCIÓN DE RUTA: Apuntar a 'models/dev/mental_health_model.onnx' en el bucket
    print("Descargando desde models/dev/mental_health_model.onnx...")
    blob_modelo = bucket.blob("models/dev/mental_health_model.onnx")
    blob_modelo.download_to_filename("src/mental_health_model.onnx")
    
    # 2. CORRECCIÓN DE RUTA: Asumiendo que test_data.csv está dentro de la carpeta 'data/'
    print("Descargando desde data/test_data.csv...")
    blob_datos = bucket.blob("data/test_data.csv")
    blob_datos.download_to_filename("test_data.csv")
    
    print("Descarga de assets estructurada completada con éxito.")

if __name__ == "__main__":
    download_blob()