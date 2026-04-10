import streamlit as st
import pandas as pd
import cv2
import os
import datetime
import pytz
import numpy as np
from PIL import Image

# Configuración
st.set_page_config(page_title="Sistema de Acceso Ultra-Lite", page_icon="👤")
st.title("👤 Control de Acceso (Versión Estable)")

EXCEL_FILE = "registro_ia.xlsx"
ZONA_HORARIA = pytz.timezone('America/Santiago')

# Cargar el detector de rostros pre-entrenado (viene con OpenCV)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def anotar_en_excel(nombre_id, estado):
    ahora = datetime.datetime.now(ZONA_HORARIA)
    nuevo = {'ID': nombre_id, 'Fecha_Hora': ahora.strftime("%d/%m/%Y %H:%M:%S"), 'Estado': estado}
    df = pd.read_excel(EXCEL_FILE) if os.path.exists(EXCEL_FILE) else pd.DataFrame(columns=['ID', 'Fecha_Hora', 'Estado'])
    df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

tab1, tab2 = st.tabs(["📸 Escáner", "📊 Historial"])

with tab1:
    img_file = st.camera_input("Capturar Rostro")
    if img_file:
        # Convertir imagen para OpenCV
        img = Image.open(img_file)
        img_array = np.array(img)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Detectar rostros
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            st.success(f"✅ ¡Rostro detectado!")
            nombre = st.text_input("Ingresa tu Nombre para registrar acceso:")
            if st.button("Confirmar Registro"):
                anotar_en_excel(nombre, "Acceso Autorizado")
                st.balloons()
                st.info(f"Registro completado para: {nombre}")
        else:
            st.error("❌ No se detecta un rostro claro. Reintenta con mejor luz.")

with tab2:
    if os.path.exists(EXCEL_FILE):
        st.subheader("Libro de Asistencia")
        df_log = pd.read_excel(EXCEL_FILE)
        st.dataframe(df_log.sort_values(by="Fecha_Hora", ascending=False), use_container_width=True)
    else:
        st.info("Aún no hay registros.")
