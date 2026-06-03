# 🧠 Sistema de Producción MLOps para Clasificación Multietiqueta en Salud Mental

Este repositorio contiene la arquitectura completa para el entrenamiento, conversión, integración continua (CI/CD), contenedorización y despliegue en la nube de un modelo de Machine Learning multietiqueta enfocado en la detección simultánea de condiciones de salud mental.

El sistema expone una API de inferencia de alto rendimiento que evalúa perfiles de pacientes basándose en variables demográficas y escalas de severidad de síntomas (0-10) para diagnosticar en paralelo: **Depresión, Ansiedad, Trastorno Bipolar, TDAH y Trastornos relacionados con Sustancias**.

---

## 🏗️ 1. Arquitectura General y Flujo de Trabajo

El ciclo MLOps implementado garantiza que cualquier cambio en el modelo o en el código del servidor se evalúe, empaquete y despliegue automáticamente en la infraestructura de AWS a través de un aislamiento riguroso de entornos (**Desarrollo** y **Producción**).

┌─────────────────────────────────────────────────────────┐
               │    📦 CÓDIGO FUENTE & ARTEFACTOS DEL MODELO (.ONNX)    │
               └────────────────────────────┬────────────────────────────┘
                                            │
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │         🐙 REPOSITORIO REMOTO: GITHUB (MAIN)            │
               └────────────────────────────┬────────────────────────────┘
                                            │  (git push / triggers)
                                            ▼
               ┌─────────────────────────────────────────────────────────┐
               │       🤖 PIPELINE AUTOMATIZADO: GITHUB ACTIONS          │
               │         - Validaciones de Calidad y Pytest -            │
               └────────────────────────────┬────────────────────────────┘
                                            │
           ┌────────────────────────────────┴────────────────────────────────┐
           │                                                                 │
           ▼ (Ambiente: Desarrollo)                                          ▼ (Ambiente: Producción)
 ┌──────────────────────────┐                                      ┌──────────────────────────┐
 │ 🐳 Build & Tag Docker    │                                      │ 🐳 Build & Tag Docker    │
 └─────────┬────────────────┘                                      └─────────┬────────────────┘
           │                                                                 │
           ▼ (Secure Push)                                                   ▼ (Secure Push)
 ┌──────────────────────────┐                                      ┌──────────────────────────┐
 │ ☁️ AWS ECR Repository     │                                      │ ☁️ AWS ECR Repository     │
 │   [salud-mental-dev]     │                                      │   [salud-mental-prod]    │
 └─────────┬────────────────┘                                      └─────────┬────────────────┘
           │                                                                 │
           ▼ (Despliegue Fargate)                                            ▼ (Despliegue Fargate)
 ┌──────────────────────────┐                                      ┌──────────────────────────┐
 │ ⚙️ AWS ECS / Cluster      │                                      │ ⚙️ AWS ECS / Cluster      │
 │   - Task dev:1 (Active)  │                                      │   - Task prod:1 (Active) │
 └─────────┬────────────────┘                                      └─────────┬────────────────┘
           │ ⏬ (Descarga de artefacto)                                      │ ⏬ (Descarga de artefacto)
 ┌─────────┴────────────────┐                                      ┌─────────┴────────────────┐
 │ 🪣 AWS S3 Model Storage  │                                      │ 🪣 AWS S3 Model Storage  │
 │   📂 modelos/dev/*.onnx  │                                      │   📂 modelos/prod/*.onnx │
 └─────────┬────────────────┘                                      └─────────┬────────────────┘
           │                                                                 │
           ▼ (Exposición de Servicio)                                        ▼ (Exposición de Servicio)
 ┌──────────────────────────┐                                      ┌──────────────────────────┐
 │ 🚀 FastAPI REST Service  │                                      │ 🚀 FastAPI REST Service  │
 │   - URL: http://IP:8000  │                                      │   - URL: http://IP:8000  │
 └─────────┬────────────────┘                                      └─────────┬────────────────┘
           │                                                                 │
           ▼ (Trazabilidad Append-Only)                                      ▼ (Trazabilidad Append-Only)
 ┌──────────────────────────┐                                      ┌──────────────────────────┐
 │ 📝 AWS S3 Data Drift Logs│                                      │ 📝 AWS S3 Data Drift Logs│
 │   📄 predicciones_dev.txt│                                      │   📄 predicciones_prod.txt│
 └─────────▲────────────────┘                                      └─────────▲────────────────┘
           │                                                                 │
           └────────────────────────────────┬────────────────────────────────┘
                                            │ (Peticiones HTTP / JSON Payload)
                                            │
                                  ┌─────────┴────────────────┐
                                  │ 💻 INTERFAZ DE USUARIO   │
                                  │   - Streamlit Web App -  │
                                  │   - Local Port 8501 -    │
                                  └──────────────────────────┘

```text
 [ Código / Modelo ] ──> [ GitHub Repo ] ──> [ GitHub Actions (CI/CD) ]
                                                   │
                   ┌───────────────────────────────┴───────────────────────────────┐
                   ▼ (Pipeline DEV)                                                ▼ (Pipeline PROD)
         [ Build & Push Docker ]                                         [ Build & Push Docker ]
                   │                                                               │
                   ▼                                                               ▼
        [ AWS ECR: salud-mental-dev ]                                  [ AWS ECR: salud-mental-prod ]
                   │                                                               │
                   ▼                                                               ▼
       [ AWS ECS / Fargate Tareas ]                                    [ AWS ECS / Fargate Tareas ]
    (Carga de S3: modelos/dev/*.onnx)                              (Carga de S3: modelos/prod/*.onnx)
                   │                                                               │
                   ├───────────────────────────────┼───────────────────────────────┤
                   ▼                                                               ▼
          [ API REST: Puerto 8000 ]                                       [ API REST: Puerto 8000 ]
     (Logs S3: predicciones_dev.txt)                                 (Logs S3: predicciones_prod.txt)
                   ▲                                                               ▲
                   └───────────────────────────────┼───────────────────────────────┘
                                                   │
                                        [ Interfaz Local Streamlit ]

```

## 🛠️ 2. Componentes del Repositorio
El proyecto se encuentra organizado bajo una estructura modular que separa el núcleo del servicio web, las pruebas automatizadas, la infraestructura de despliegue y la interfaz de usuario:

    github/workflows/cicd.yml: Pipeline automatizado en GitHub Actions encargado de la ejecución de pruebas unitarias, compilación de imágenes Docker y actualización de los servicios en AWS.

    app/: Directorio principal del backend. Contiene main.py (servidor FastAPI que inicializa el motor ONNX Runtime y gestiona la lógica de almacenamiento de trazas en S3) y requirements.txt (dependencias estrictas del contenedor).

    frontend/: Portal interactivo desarrollado en Streamlit (app_tester.py) para consumir los endpoints remotos mediante controles deslizantes numéricos calibrados.

    tests/: Suite de pruebas unitarias (test_model.py) para validar que las dimensiones del modelo ONNX y la respuesta de la API cumplan con los requerimientos técnicos antes de compilar.

    Dockerfile: Manifiesto de construcción de la imagen de Docker basada en capas optimizadas de Python Alpine/Slim.

    model_features.json: Registro estricto del orden secuencial de las 20 características que exige la matriz matemática del modelo.

## 🚀 3. Pipeline de Integración y Despliegue Continuo (CI/CD)
El flujo automatizado en GitHub Actions se dispara ante eventos de push o pull_request en la rama principal y ejecuta de manera secuencial los siguientes pasos técnicos:

    Entorno y Dependencias: Levanta un corredor virtual Linux, configura Python e instala todas las dependencias de testing.

    Pruebas Unitarias: Ejecuta la suite de pruebas sobre el comportamiento del modelo a través de pytest. Si alguna prueba falla, el pipeline se aborta de inmediato protegiendo los entornos en la nube.

    Autenticación en AWS: Utiliza credenciales seguras almacenadas en los GitHub Actions Secrets (AWS_ACCESS_KEY_ID y AWS_SECRET_ACCESS_KEY) para firmar llamadas con la región de destino (us-east-2).

    Compilación y Empaquetado Docker: Se autentica en Amazon ECR (Elastic Container Registry), construye la imagen Docker local de la API y le asigna tags dinámicos basados en el hash del commit.

    Push a Registros Aislados: Envía la imagen compilada hacia los repositorios correspondientes en la nube:

    salud-mental-task-dev para el entorno de desarrollo.

    salud-mental-task-prod para el entorno de producción.

    Despliegue Flotante (AWS ECS): Actualiza de manera forzada la definición de la tarea en Amazon ECS (Elastic Container Service) bajo el tipo de lanzamiento AWS Fargate, levantando los nuevos contenedores en tiempo real sin caídas de servicio (Zero-Downtime Deployment).

## ☁️ 4. Infraestructura e Integración en AWS
La solución en la nube se soporta enteramente sobre servicios serverless e independientes que garantizan alta disponibilidad:

    Amazon S3 (Simple Storage Service)
    Modelos: Almacena de manera desacoplada los artefactos binarios de los modelos en rutas separadas (s3://bucket-salud-mental/modelos/dev/mental_health_model.onnx y su homólogo en prod). Al iniciar, cada contenedor descarga el archivo asignado a su entorno.

    Monitoreo / Drift: Cada inferencia exitosa procesada por la API genera un log estructurado append-only en monitoreo/predicciones_dev.txt o predicciones_prod.txt para auditorías posteriores de datos y detección de data drift.

    Amazon ECS y AWS Fargate
    El clúster ejecuta en paralelo dos servicios lógicos sobre micro-instancias de Fargate aisladas por VPC.

    Seguridad de Red (Security Groups): El tráfico entrante está estrictamente controlado, manteniendo abierto únicamente el puerto 8000 para procesar peticiones HTTP REST de la API.

🖥️ 5. Interfaz Gráfica de Pruebas (Streamlit)
Para realizar pruebas clínicas simuladas e interactuar de forma intuitiva con la infraestructura en AWS, se desarrolló un panel de control interactivo que respeta los rangos de entrenamiento reales del modelo (Edades entre 18-80 y escalas de síntomas de severidad numérica continua de 0 a 10).

Cómo ejecutar la aplicación de pruebas en tu máquina local:
1. Crear y activar el entorno virtual de Python:

            # En macOS o Linux
            python3 -m venv venv
            source venv/bin/activate

            # En Windows (Powershell)
            python -m venv venv
            .\venv\Scripts\Activate.ps1

2. Instalar las dependencias de la interfaz:

            pip install streamlit requests

3. Configurar los Endpoints Activos:
Abra el archivo frontend/app_tester.py y asegúrese de actualizar las variables de dirección IP pública provistas por las tareas activas de AWS Fargate:

            IP_DEV = "TU_NUEVA_IP_DE_DEV"
            IP_PROD = "TU_NUEVA_IP_DE_PROD"

4. Lanzar la aplicación web:

            Bash
            streamlit run frontend/app_tester.py

El portal web se abrirá automáticamente en http://localhost:8501, permitiendo configurar un perfil, seleccionar mediante botones si la inferencia se evalúa en el entorno de Desarrollo o Producción, y visualizar un diagnóstico médico categorizado por colores e interactivo en español.


## ⚙️ 6. Gestión de Costos e Infraestructura (Mantenimiento)
Dado que AWS Fargate factura bajo un esquema continuo por segundo de uso de CPU y Memoria RAM asignada mientras los contenedores se encuentren en estado RUNNING, se definió una política de suspensión manual obligatoria para periodos de inactividad:

Apagado Seguro (Costo $0 USD): Acceder a la consola de AWS ECS -> Clúster -> Pestaña Servicios -> Actualizar Servicio -> Establecer el número de Tareas deseadas (Desired tasks) en 0. Fargate destruirá los contenedores de forma segura y congelará el cobro por cómputo sin perder las configuraciones.

Reactivación: Actualizar el servicio de nuevo fijando las Tareas deseadas en 1. Al iniciar, Fargate asignará una nueva IP pública que deberá ser copiada en el portal de Streamlit para reanudar el procesamiento