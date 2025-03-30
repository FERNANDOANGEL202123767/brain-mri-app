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
import base64
from PIL import Image
from io import BytesIO

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

# Función para descargar archivos desde Drive solo si no existen
def download_file_if_not_exists(file_id, destination):
    if not os.path.exists(destination):
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
    else:
        logging.info(f"El archivo {destination} ya existe, no se descargará nuevamente.")

# Descargar modelos al iniciar la aplicación solo si no existen
MODEL_JSON_ID = '1eRuupBoFuEB-VfhewOiS_SVEaTA2AgV8'
WEIGHTS_ID = '1fv7XRHe9WsbZEEunX7V_WOpa4WgP6Wjt'
download_file_if_not_exists(MODEL_JSON_ID, 'resnet-50-MRI.json')
download_file_if_not_exists(WEIGHTS_ID, 'weights.hdf5')

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

# Función para generar Grad-CAM
def get_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out"):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 1]  # Clase "tumor" (índice 1)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)  # Normalizar entre 0 y 1
    return cv2.resize(heatmap, (256, 256))

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
        img_resized = cv2.resize(img, (256, 256))
        img_input = img_resized / 255.0
        img_input = np.expand_dims(img_input, axis=0)
        logging.info(f"Forma de la imagen procesada: {img_input.shape}")

        # Hacer la predicción
        prediction = model.predict(img_input)
        logging.info(f"Predicción cruda: {prediction}")
        has_tumor = np.argmax(prediction[0])
        confidence = float(prediction[0][has_tumor]) * 100
        result = 'Tumor detectado' if has_tumor else 'No se detectó tumor'
        logging.info(f"Resultado: {result}, Confianza: {confidence}")

        # Convertir imagen original a base64
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_original_base64 = base64.b64encode(buffered.getvalue()).decode()
        img_original_data = f"data:image/png;base64,{img_original_base64}"

        # Si hay tumor, generar Grad-CAM y dibujar círculo
        img_tumor_data = None
        if has_tumor:
            # Generar heatmap con Grad-CAM
            heatmap = get_gradcam_heatmap(img_input, model)
            # Encontrar el centro del área más activa
            y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            # Dibujar círculo rojo en la imagen original
            img_tumor = img_rgb.copy()
            center = (x, y)
            radius = 30  # Ajusta el tamaño del círculo según necesites
            cv2.circle(img_tumor, center, radius, (255, 0, 0), 2)  # Círculo rojo con grosor 2
            # Convertir a base64
            img_tumor_pil = Image.fromarray(img_tumor)
            buffered = BytesIO()
            img_tumor_pil.save(buffered, format="PNG")
            img_tumor_base64 = base64.b64encode(buffered.getvalue()).decode()
            img_tumor_data = f"data:image/png;base64,{img_tumor_base64}"

        return jsonify({
            'result': result,
            'confidence': confidence,
            'img_original': img_original_data,
            'img_tumor': img_tumor_data
        })
    except Exception as e:
        logging.error(f"Error en la predicción: {e}")
        return jsonify({'error': 'Error al procesar la imagen'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
