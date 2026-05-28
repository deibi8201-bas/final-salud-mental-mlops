FROM python:3.10-slim

WORKDIR /app

# Instalar dependencias del sistema si son necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copiar el código de la aplicación
COPY src/ ./src/

EXoOSE 8080

# Comando para arrancar uvicorn o tu framework (ajusta app:app según corresponda)
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8080"]