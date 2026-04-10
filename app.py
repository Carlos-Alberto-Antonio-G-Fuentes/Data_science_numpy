import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import datetime
import pytz

# Inicializar Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

st.set_page_config(page_title="IA Facial por Vectores", page_icon="👤")
st.title("👤 Reconocimiento por Malla Facial")

VECTORES_FILE = "vectores_rostros.csv"
ZONA_HORARIA = pytz.timezone('America/Santiago')

def obtener_vector(img_array):
    results = face_mesh.process(cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
    if results.multi_face_landmarks:
        # Extraemos las coordenadas X, Y, Z de los 468 puntos como un vector
        landmarks = results.multi_face_landmarks[0].landmark
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
    return None

def registrar_acceso(nombre):
    ahora = datetime.datetime.now(ZONA_HORARIA)
    st.success(f"✅ Acceso concedido: {nombre}")
    # Aquí podrías añadir la lógica de guardar en Excel como antes

if st.camera_input("Enfoca tu rostro"):
    # (Lógica de procesamiento aquí...)
    st.info("Procesando biometría de puntos clave...")
