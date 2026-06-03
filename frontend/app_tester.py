import streamlit as st
import requests
import json

# Configuración de la página
st.set_page_config(
    page_title="Portal MLOps - Salud Mental",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Portal de Inferencia Multietiqueta - MLOps")
st.write("Configura las características en los rangos numéricos reales del modelo para evaluar los entornos.")

# =====================================================================
# CONFIGURACIÓN DE DIRECCIONES IP (Asegúrate de poner tus IPs actuales)
# =====================================================================
IP_DEV = "18.218.75.80"        
IP_PROD = "18.216.56.92"      

# Diccionario de traducción para la interfaz
TRADUCCION_CONDICIONES = {
    "Depression": "Depresión",
    "Anxiety": "Ansiedad",
    "Bipolar": "Trastorno Bipolar",
    "ADHD": "TDAH (Trastorno por Déficit de Atención e Hiperactividad)",
    "SubstanceRelated": "Trastornos relacionados con Sustancias"
}

st.header("📋 Parámetros del Paciente (Valores del Modelo)")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Demografía y Base")
    Age = st.slider("Edad (Age):", min_value=18, max_value=80, value=25, step=1)
    Sex = st.radio("Sexo (Sex):", options=[0, 1], format_func=lambda x: "0 (Masculino)" if x == 0 else "1 (Femenino)")
    
    st.subheader("Bloque de Síntomas A")
    insomnia = st.slider("Insomnio (insomnia):", min_value=0, max_value=10, value=1, step=1)
    fatigue = st.slider("Fatiga (fatigue):", min_value=0, max_value=10, value=1, step=1)
    panic = st.slider("Pánico (panic):", min_value=0, max_value=10, value=0, step=1)
    panic_sleep = st.slider("Pánico al dormir (panic_sleep):", min_value=0, max_value=10, value=0, step=1)
    social_withdrawal = st.slider("Aislamiento social (social_withdrawal):", min_value=0, max_value=10, value=1, step=1)
    mood_swings = st.slider("Cambios de humor (mood_swings):", min_value=0, max_value=10, value=1, step=1)
    anhedonia = st.slider("Anhedonia (anhedonia):", min_value=0, max_value=10, value=0, step=1)
    rumination = st.slider("Rumiación (rumination):", min_value=0, max_value=10, value=1, step=1)

with col2:
    st.subheader("Bloque de Síntomas B")
    appetite_change = st.slider("Cambio de apetito (appetite_change):", min_value=0, max_value=10, value=0, step=1)
    concentration_issues = st.slider("Concentración (concentration_issues):", min_value=0, max_value=10, value=1, step=1)
    self_harm_ideation = st.slider("Ideación autolesiva (self_harm_ideation):", min_value=0, max_value=10, value=0, step=1)
    irritability = st.slider("Irritabilidad (irritability):", min_value=0, max_value=10, value=1, step=1)
    withdrawal = st.slider("Abstinencia (withdrawal):", min_value=0, max_value=10, value=0, step=1)
    anxiety_general = st.slider("Ansiedad general (anxiety_general):", min_value=0, max_value=10, value=1, step=1)
    psychomotor = st.slider("Psicomotor (psychomotor):", min_value=0, max_value=10, value=0, step=1)
    suicidal_thoughts = st.slider("Pensamientos suicidas (suicidal_thoughts):", min_value=0, max_value=10, value=0, step=1)
    hopelessness = st.slider("Desesperanza (hopelessness):", min_value=0, max_value=10, value=1, step=1)
    panic_physical = st.slider("Pánico físico (panic_physical):", min_value=0, max_value=10, value=0, step=1)

payload = {
    "Age": float(Age), "Sex": float(Sex), "insomnia": float(insomnia), "fatigue": float(fatigue), "panic": float(panic),
    "panic_sleep": float(panic_sleep), "social_withdrawal": float(social_withdrawal), "mood_swings": float(mood_swings),
    "anhedonia": float(anhedonia), "rumination": float(rumination), "appetite_change": float(appetite_change),
    "concentration_issues": float(concentration_issues), "self_harm_ideation": float(self_harm_ideation),
    "irritability": float(irritability), "withdrawal": float(withdrawal), "anxiety_general": float(anxiety_general),
    "psychomotor": float(psychomotor), "suicidal_thoughts": float(suicidal_thoughts), "hopelessness": float(hopelessness),
    "panic_physical": float(panic_physical)
}

st.write("---")
st.header("🚀 Acciones de Envío")

btn_col1, btn_col2 = st.columns(2)
target_url = None
env_name = None

with btn_col1:
    if st.button("🔴 Enviar a Desarrollo (DEV)", use_container_width=True):
        target_url = f"http://{IP_DEV}:8000/predict"
        env_name = "Desarrollo (DEV)"

with btn_col2:
    if st.button("🟢 Enviar a Producción (PROD)", use_container_width=True):
        target_url = f"http://{IP_PROD}:8000/predict"
        env_name = "Producción (PROD)"

if target_url:
    st.info(f"Conectando con el entorno de {env_name}...")
    try:
        response = requests.post(
            target_url,
            data=json.dumps(payload),
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            res_data = response.json()
            predictions = res_data.get("predictions", {})
            entorno_servidor = res_data.get("environment", "Desconocido")
            
            # --- DISEÑO MEJORADO DE RESULTADOS ---
            st.success(f"✅ ¡Inferencia completada con éxito en el entorno: **{entorno_servidor.upper()}**!")
            
            st.subheader("🎯 Diagnóstico Estimado por el Modelo:")
            
            # Crear dos columnas para clasificar visualmente los resultados
            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.markdown("### 🚨 Condiciones Detectadas")
                detectados = 0
                for eng, esp in TRADUCCION_CONDICIONES.items():
                    if predictions.get(eng, 0) == 1:
                        st.error(f"⚠️ **{esp}**")
                        detectados += 1
                if detectados == 0:
                    st.info("Ninguna condición fue detectada para este perfil.")

            with col_res2:
                st.markdown("### ✅ Condiciones No Detectadas")
                for eng, esp in TRADUCCION_CONDICIONES.items():
                    if predictions.get(eng, 0) == 0:
                        st.success(f"• {esp}")
            
            # Pestaña colapsable para auditoría técnica
            with st.expander("🔍 Ver metadatos técnicos de la API"):
                st.json(res_data)
                
        else:
            st.error(f"Error {response.status_code}")
            st.text(response.text)
    except Exception as e:
        st.error(f"Error de conexión: {str(e)}")