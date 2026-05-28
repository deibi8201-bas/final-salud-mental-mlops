import os
from datetime import datetime
from google.cloud import storage

def log_prediction_to_gcs(features_list, diagnostics_list, max_confidence):
    """
    Registra de forma persistente cada consulta y su predicción en un archivo de texto
    almacenado dentro del Bucket de Google Cloud Storage.
    """
    env = os.environ.get("ENV", "dev")
    bucket_name = os.environ.get("GCP_BUCKET_NAME")
    
    # Si no hay bucket configurado (por ejemplo, en pruebas locales básicas), lo escribe local
    if not bucket_name:
        print("Advertencia: GCP_BUCKET_NAME no definido. Registrando log localmente.")
        with open(f"predicciones_{env}.txt", "a") as f:
            f.write(f"[{datetime.utcnow().isoformat()}] Features: {features_list} -> Pred: {diagnostics_list}\n")
        return

    blob_path = f"logs/predicciones_{env}.txt"
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        # Intentar descargar el historial de logs existente; si no existe, arranca en blanco
        if blob.exists():
            existing_content = blob.download_as_text()
        else:
            existing_content = ""
            
        # Construir la nueva línea de log estructurada
        timestamp = datetime.utcnow().isoformat()
        log_line = (
            f"[{timestamp}] "
            f"Inputs: {features_list} | "
            f"Diagnosticos: {diagnostics_list} | "
            f"Confianza_Max: {max_confidence:.4f}\n"
        )
        
        new_content = existing_content + log_line
        
        # Subir el archivo de texto actualizado a la nube
        blob.upload_from_string(new_content, content_type="text/plain")
        print(f"Log de monitoreo sincronizado exitosamente en GCS: {blob_path}")
        
    except Exception as e:
        print(f"Error al escribir el log de monitoreo en GCS: {e}")