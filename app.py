import os
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import io
import logging

app = Flask(__name__)

# Configurar logging
logging.basicConfig(level=logging.INFO)

# Configurar credenciales desde variable de entorno
credentials_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_CONTENT')
if not credentials_json:
    raise ValueError("La variable de entorno GOOGLE_APPLICATION_CREDENTIALS_CONTENT no está configurada")
with open('brainmriapp-credentials.json', 'w') as f:
    f.write(credentials_json)
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'brainmriapp-credentials.json'

# Autenticar con Google Drive
try:
    credentials = service_account.Credentials.from_service_account_file('brainmriapp-credentials.json')
    drive_service = build('drive', 'v3', credentials=credentials)
except Exception as e:
    logging.error(f"Error al autenticar con Google Drive: {e}")
    raise

# Función para descargar archivos desde Drive
def download_file(file_id, destination):
    try:
        request = drive_service.files().get_media(fileId=file_id)
        with open(destination, 'wb') as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                logging.info(f"Descargando {destination}: {int(status.progress() * 100)}%")
    except Exception as e:
        logging.error(f"Error al descargar archivo {file_id}: {e}")
        raise

# Descargar modelos al iniciar la aplicación
MODEL_JSON_ID = '1eRuupBoFuEB-VfhewOiS_SVEaTA2AgV8'  # Reemplaza con tu ID real si es diferente
WEIGHTS_ID = '1fv7XRHe9WsbZEEunX7V_WOpa4WgP6Wjt'  # Reemplaza con el ID real del archivo de pesos
download_file(MODEL_JSON_ID, 'resnet-50-MRI.json')
download_file(WEIGHTS_ID, 'weights.hdf5')

# Cargar el modelo
try:
    with open('resnet-50-MRI.json', 'r') as json_file:
        model_json = json_file.read()
    model = tf.keras.models.model_from_json(model_json)
    model.load_weights('weights.hdf5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    logging.info("Modelo cargado exitosamente")
except Exception as e:
    logging.error(f"Error al cargar el modelo: {e}")
    raise

# Ruta principal para la interfaz
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se proporcionó archivo'}), 400
    file = request.files['file']
    try:
        # Leer la imagen en color (RGB)
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        # Redimensionar a 256x256
        img = cv2.resize(img, (256, 256))
        # Normalizar los valores de los píxeles (0 a 1)
        img = img / 255.0
        # Añadir dimensión del batch
        img = np.expand_dims(img, axis=0)
        # Registrar la forma para depuración
        logging.info(f"Forma de la imagen procesada: {img.shape}")
        
        # Hacer la predicción con el modelo
        prediction = model.predict(img)
        logging.info(f"Predicción cruda: {prediction}")
        # Suponiendo que 0 = no tumor, 1 = tumor
        has_tumor = np.argmax(prediction[0])
        confidence = float(prediction[0][has_tumor]) * 100
        result = 'Tumor detectado' if has_tumor else 'No se detectó tumor'
        logging.info(f"Resultado: {result}, Confianza: {confidence}")
        return jsonify({'result': result, 'confidence': confidence})
    except Exception as e:
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': 'Error al procesar la imagen'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
