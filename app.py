import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import datetime
import pytz
from PIL import Image

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="IA Facial Vectorial", page_icon="🧬")
st.title("🧬 Biometría por Malla Facial")

ZONA_HORARIA = pytz.timezone('America/Santiago')
DB_VECTORES = "base_vectores.csv"
DB_EXCEL = "log_accesos.xlsx"

# Inicializar Mediapipe (Malla Facial de 468 puntos)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)

# --- FUNCIONES CORE ---
def obtener_vector(img_file):
    """Convierte un rostro en un vector numérico de puntos clave."""
    img = Image.open(img_file)
    img_array = np.array(img)
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        # Creamos un vector plano con las coordenadas X, Y, Z de cada punto
        return np.array([ [lm.x, lm.y, lm.z] for lm in landmarks ]).flatten()
    return None

def guardar_acceso(nombre):
    ahora = datetime.datetime.now(ZONA_HORARIA)
    nuevo = {'ID': nombre, 'Fecha_Hora': ahora.strftime("%d/%m/%Y %H:%M:%S")}
    df = pd.read_excel(DB_EXCEL) if os.path.exists(DB_EXCEL) else pd.DataFrame(columns=['ID', 'Fecha_Hora'])
    df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
    df.to_excel(DB_EXCEL, index=False)

# --- INTERFAZ ---
tab1, tab2 = st.tabs(["📸 Escaneo Biométrico", "📊 Historial"])

with tab1:
    foto = st.camera_input("Enfoca tu rostro")
    
    if foto:
        vector_actual = obtener_vector(foto)
        
        if vector_actual is not None:
            # Intentar reconocer
            identificado = False
            if os.path.exists(DB_VECTORES):
                df_v = pd.read_csv(DB_VECTORES)
                
                # Comparamos el vector actual con todos los guardados
                for index, row in df_v.iterrows():
                    vector_guardado = np.array(row[1:]) # Saltamos la columna del nombre
                    # Calculamos la distancia Euclidiana (qué tan cerca están los puntos)
                    distancia = np.linalg.norm(vector_actual - vector_guardado)
                    
                    # Umbral de reconocimiento (ajustable)
                    if distancia < 0.25: 
                        nombre = row['nombre']
                        st.success(f"✅ BIOMETRÍA VALIDADA: Hola {nombre}")
                        guardar_acceso(nombre)
                        st.balloons()
                        identificado = True
                        break
            
            if not identificado:
                st.warning("❓ Rostro no registrado en la base de datos vectorial.")
                nuevo_nombre = st.text_input("Ingresa tu nombre para crear tu Perfil Biométrico:")
                if st.button("Registrar Huella Facial"):
                    # Guardar el vector en el CSV
                    nuevo_perfil = pd.DataFrame([ [nuevo_nombre] + vector_actual.tolist() ])
                    if not os.path.exists(DB_VECTORES):
                        cols = ['nombre'] + [f'p{i}' for i in range(len(vector_actual))]
                        nuevo_perfil.columns = cols
                        nuevo_perfil.to_csv(DB_VECTORES, index=False)
                    else:
                        nuevo_perfil.to_csv(DB_VECTORES, mode='a', header=False, index=False)
                    
                    st.info(f"Perfil de {nuevo_nombre} creado con éxito. ¡Vuelve a escanearte!")
        else:
            st.error("No se pudo extraer la malla facial. Asegúrate de tener buena luz.")

with tab2:
    if os.path.exists(DB_EXCEL):
        st.dataframe(pd.read_excel(DB_EXCEL).sort_values(by="Fecha_Hora", ascending=False))
