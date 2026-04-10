import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
import datetime
import pytz
from PIL import Image

# --- SOLUCIÓN AL ERROR DE IMPORTACIÓN ---
try:
    import mediapipe as mp
    # Importación directa del módulo para evitar AttributeError en Streamlit Cloud
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
    
    # Inicializar el motor de malla facial
    face_mesh_engine = mp_face_mesh.FaceMesh(
        static_image_mode=True, 
        max_num_faces=1, 
        min_detection_confidence=0.5
    )
except Exception as e:
    st.error(f"Error crítico al cargar módulos de IA: {e}")

# --- CONFIGURACIÓN GENERAL ---
st.set_page_config(page_title="IA Facial Vectorial", page_icon="🧬")
st.title("🧬 Biometría por Malla Facial")
st.markdown("Sistema de reconocimiento basado en vectores de 468 puntos clave.")

ZONA_HORARIA = pytz.timezone('America/Santiago')
DB_VECTORES = "base_vectores.csv"
DB_EXCEL = "log_accesos.xlsx"

# --- FUNCIONES DE PROCESAMIENTO ---
def obtener_vector(img_file):
    """Extrae la firma biométrica del rostro."""
    img = Image.open(img_file)
    img_array = np.array(img)
    # Mediapipe requiere formato RGB (OpenCV usa BGR por defecto, pero PIL usa RGB)
    results = face_mesh_engine.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # Creamos un vector de 1404 elementos (468 puntos x 3 coordenadas X,Y,Z)
        return np.array([ [lm.x, lm.y, lm.z] for lm in landmarks ]).flatten()
    return None

def guardar_registro_excel(nombre):
    """Guarda la marca de tiempo en el archivo Excel."""
    ahora = datetime.datetime.now(ZONA_HORARIA)
    nuevo = {'ID': nombre, 'Fecha_Hora': ahora.strftime("%d/%m/%Y %H:%M:%S")}
    
    if os.path.exists(DB_EXCEL):
        df = pd.read_excel(DB_EXCEL)
    else:
        df = pd.DataFrame(columns=['ID', 'Fecha_Hora'])
        
    df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
    df.to_excel(DB_EXCEL, index=False)

# --- INTERFAZ DE USUARIO ---
tab1, tab2 = st.tabs(["📸 Escaneo Biométrico", "📊 Historial de Accesos"])

with tab1:
    foto = st.camera_input("Captura tu rostro para validación")
    
    if foto:
        with st.spinner("Analizando biometría facial..."):
            vector_actual = obtener_vector(foto)
        
        if vector_actual is not None:
            identificado = False
            
            # 1. Intentar reconocer comparando con la base de datos CSV
            if os.path.exists(DB_VECTORES):
                df_v = pd.read_csv(DB_VECTORES)
                
                for index, row in df_v.iterrows():
                    # Extraer el vector guardado (todas las columnas excepto 'nombre')
                    vector_guardado = np.array(row[1:].values).astype(float)
                    
                    # Cálculo de Distancia Euclidiana
                    distancia = np.linalg.norm(vector_actual - vector_guardado)
                    
                    # Umbral de tolerancia: 0.20 - 0.30 suele ser ideal para Mediapipe
                    if distancia < 0.28: 
                        nombre_encontrado = row['nombre']
                        st.success(f"✅ ACCESO AUTORIZADO: Bienvenido {nombre_encontrado}")
                        guardar_registro_excel(nombre_encontrado)
                        st.balloons()
                        identificado = True
                        break
            
            # 2. Si no se reconoce, permitir registro nuevo
            if not identificado:
                st.warning("⚠️ Perfil biométrico no encontrado.")
                nuevo_nombre = st.text_input("Ingresa tu nombre para registrarte:")
                if st.button("Crear Perfil Vectorial"):
                    # Crear fila: nombre + los 1404 valores del vector
                    columnas = ['nombre'] + [f'p{i}' for i in range(len(vector_actual))]
                    datos_fila = [nuevo_nombre] + vector_actual.tolist()
                    nuevo_df = pd.DataFrame([datos_fila], columns=columnas)
                    
                    if not os.path.exists(DB_VECTORES):
                        nuevo_df.to_csv(DB_VECTORES, index=False)
                    else:
                        nuevo_df.to_csv(DB_VECTORES, mode='a', header=False, index=False)
                    
                    st.info(f"Registro exitoso para {nuevo_nombre}. ¡Prueba escanearte de nuevo!")
        else:
            st.error("No se detectó un rostro. Asegúrate de estar frente a la cámara con buena luz.")

with tab2:
    st.subheader("Registros en el Servidor")
    if os.path.exists(DB_EXCEL):
        df_log = pd.read_excel(DB_EXCEL)
        st.dataframe(df_log.sort_values(by="Fecha_Hora", ascending=False), use_container_width=True)
    else:
        st.info("No hay registros de acceso todavía.")
