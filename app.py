import streamlit as st
import pandas as pd
import face_recognition
import cv2
import os
import datetime
import pytz
import numpy as np
from PIL import Image

st.set_page_config(page_title="Sistema IA Facial Lite", page_icon="👤")
st.title("👤 Control de Acceso (Versión Lite)")

DB_PATH = "mis_rostros"
EXCEL_FILE = "registro_ia.xlsx"
ZONA_HORARIA = pytz.timezone('America/Santiago')

if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)

def anotar_en_excel(nombre_id, confianza):
    ahora = datetime.datetime.now(ZONA_HORARIA)
    nuevo = {'ID': nombre_id, 'Fecha_Hora': ahora.strftime("%d/%m/%Y %H:%M:%S"), 'Estado': confianza}
    df = pd.read_excel(EXCEL_FILE) if os.path.exists(EXCEL_FILE) else pd.DataFrame(columns=['ID', 'Fecha_Hora', 'Estado'])
    df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

tab1, tab2 = st.tabs(["📸 Escáner", "📊 Historial"])

with tab1:
    img_file = st.camera_input("Capturar")
    if img_file:
        # Cargar imagen capturada
        image = face_recognition.load_image_file(img_file)
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) > 0:
            current_face_encoding = face_encodings[0]
            identificado = False
            
            # Comparar con la base de datos
            for file in os.listdir(DB_PATH):
                if file.endswith((".jpg", ".png")):
                    known_image = face_recognition.load_image_file(f"{DB_PATH}/{file}")
                    known_encodings = face_recognition.face_encodings(known_image)
                    
                    if len(known_encodings) > 0:
                        results = face_recognition.compare_faces([known_encodings[0]], current_face_encoding, tolerance=0.6)
                        if results[0]:
                            nombre = file.split(".")[0]
                            st.success(f"✅ Identificado: {nombre}")
                            anotar_en_excel(nombre, "Reconocido")
                            identificado = True
                            break
            
            if not identificado:
                st.error("❓ No reconocido")
                nuevo_nombre = st.text_input("Registrar como:")
                if st.button("Guardar"):
                    img_pil = Image.open(img_file)
                    img_pil.save(f"{DB_PATH}/{nuevo_nombre}.jpg")
                    anotar_en_excel(nuevo_nombre, "Registrado")
                    st.experimental_rerun()
        else:
            st.warning("No se detectó ningún rostro en la imagen.")

with tab2:
    if os.path.exists(EXCEL_FILE):
        st.dataframe(pd.read_excel(EXCEL_FILE).sort_values(by="Fecha_Hora", ascending=False))
