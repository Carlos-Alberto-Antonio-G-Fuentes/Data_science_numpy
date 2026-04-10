import os

# 2. Creamos el archivo de la aplicación (app.py)
with open('app.py', 'w') as f:
    f.write('''
import streamlit as st
import pandas as pd
import cv2
from deepface import DeepFace
import os
import datetime
import pytz
from PIL import Image
import numpy as np

st.set_page_config(page_title="Sistema IA Facial", page_icon="👤")
st.title("👤 Control de Acceso Local")

DB_PATH = "mis_rostros"
EXCEL_FILE = "registro_ia.xlsx"
ZONA_HORARIA = pytz.timezone('America/Santiago')

if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)

def anotar_en_excel(nombre_id, confianza, ruta):
    ahora = datetime.datetime.now(ZONA_HORARIA)
    nuevo = {'ID': nombre_id, 'Fecha_Hora': ahora.strftime("%d/%m/%Y %H:%M:%S"), 'Confianza': confianza, 'Archivo_Origen': ruta}
    df = pd.read_excel(EXCEL_FILE) if os.path.exists(EXCEL_FILE) else pd.DataFrame(columns=['ID', 'Fecha_Hora', 'Confianza', 'Archivo_Origen'])
    df = pd.concat([df, pd.DataFrame([nuevo])], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)
    return df

tab1, tab2 = st.tabs(["📸 Escáner", "📊 Historial"])

with tab1:
    img_file = st.camera_input("Capturar")
    if img_file:
        img = Image.open(img_file)
        img_array = np.array(img)
        temp_path = "temp_scan.jpg"
        cv2.imwrite(temp_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
        
        # Reconocimiento
        if len([f for f in os.listdir(DB_PATH) if f.endswith('.jpg')]) == 0:
            nombre = st.text_input("Primer registro - Nombre:")
            if st.button("Guardar"):
                os.rename(temp_path, f"{DB_PATH}/{nombre}.jpg")
                anotar_en_excel(nombre, "100%", f"{DB_PATH}/{nombre}.jpg")
                st.success(f"Registrado: {nombre}")
        else:
            try:
                res = DeepFace.find(img_path=temp_path, db_path=DB_PATH, model_name="Facenet", enforce_detection=False)
                if len(res) > 0 and not res[0].empty and res[0].iloc[0]['distance'] < 0.4:
                    nombre = os.path.basename(res[0].iloc[0]['identity']).split('.')[0]
                    st.success(f"Identificado: {nombre}")
                    anotar_en_excel(nombre, f"{(1-res[0].iloc[0]['distance'])*100:.1f}%", res[0].iloc[0]['identity'])
                else:
                    st.error("No reconocido")
                    nuevo = st.text_input("Nombre para registrar:")
                    if st.button("Registrar"):
                        os.rename(temp_path, f"{DB_PATH}/{nuevo}.jpg")
                        anotar_en_excel(nuevo, "Nuevo", f"{DB_PATH}/{nuevo}.jpg")
            except Exception as e: st.error(f"Error: {e}")

with tab2:
    if os.path.exists(EXCEL_FILE):
        st.dataframe(pd.read_excel(EXCEL_FILE))
    ''')

# 3. Lanzamos la aplicación en segundo plano
import urllib
print("Tu IP pública es:", urllib.request.urlopen('https://ipv4.icanhazip.com').read().decode('utf8').strip())
!streamlit run app.py & npx localtunnel --port 8501
