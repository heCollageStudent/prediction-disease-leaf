from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import cv2
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from skimage.feature import graycomatrix, graycoprops
from tensorflow.keras.applications import MobileNetV2

app = Flask(__name__)

# Load model dan tools
model = load_model('model_v2_normalization.h5')
scaler_warna = joblib.load('scaler_warna2.pkl')
scaler_tekstur = joblib.load('scaler_tekstur2.pkl')
scaler_bentuk = joblib.load('scaler_bentuk2.pkl')
label_encoder = joblib.load('label_encoder2.pkl')

cnn_base = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))

# Fungsi ekstraksi fitur manual
def extract_fitur_manual(img):
    # Warna
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray > 0
    L = img_lab[..., 0].astype(np.float32)[mask]
    a = img_lab[..., 1].astype(np.float32)[mask]
    b = img_lab[..., 2].astype(np.float32)[mask]
    fitur_warna = np.array([np.mean(L), np.std(L), np.mean(a), np.std(a), np.mean(b), np.std(b)])

    # Tekstur (GLCM)
    gray_rs = cv2.resize(gray, (128,128))
    glcm = graycomatrix(gray_rs, distances=[1,2,3,4], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)
    fitur_tekstur = np.array([graycoprops(glcm, 'contrast').mean(),
                              graycoprops(glcm, 'dissimilarity').mean(),
                              graycoprops(glcm, 'homogeneity').mean(),
                              graycoprops(glcm, 'energy').mean(),
                              graycoprops(glcm, 'correlation').mean()])

    # Bentuk
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = w / h if h != 0 else 0
    circularity = 4 * np.pi * area / (perimeter**2) if perimeter != 0 else 0
    rectangularity = area / (w * h) if w*h != 0 else 0
    diameter = np.sqrt((4 * area) / np.pi)
    fitur_bentuk = np.array([area, perimeter, w, h, aspect_ratio, circularity, rectangularity, diameter])

    return fitur_warna, fitur_tekstur, fitur_bentuk

# Fungsi prediksi
def predict_hybrid(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Gambar tidak ditemukan.")

    cnn_img = cv2.resize(img, (224, 224))
    cnn_img = img_to_array(cnn_img)
    cnn_img = preprocess_input(cnn_img)
    cnn_img = np.expand_dims(cnn_img, axis=0)
    fitur_cnn = cnn_base.predict(cnn_img)[0]

    warna, tekstur, bentuk = extract_fitur_manual(img)

    # Gabungkan semua fitur manual
    warna_cols = ['L_mean', 'L_std', 'a_mean', 'a_std', 'b_mean', 'b_std']
    tekstur_cols = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
    bentuk_cols = ['area', 'perimeter', 'width', 'height', 'aspect_ratio', 'circularity', 'rectangularity', 'diameter']

    df_warna = pd.DataFrame([warna], columns=warna_cols)
    df_tekstur = pd.DataFrame([tekstur], columns=tekstur_cols)
    df_bentuk = pd.DataFrame([bentuk], columns=bentuk_cols)

    warna_scaled = scaler_warna.transform(df_warna)
    tekstur_scaled = scaler_tekstur.transform(df_tekstur)
    bentuk_scaled = scaler_bentuk.transform(df_bentuk)

    fitur_manual = np.concatenate([warna_scaled[0], tekstur_scaled[0], bentuk_scaled[0]])

    # Gabungkan CNN dan manual
    combined_input = [np.expand_dims(fitur_cnn, axis=0), np.expand_dims(fitur_manual, axis=0)]

    # Prediksi
    probs = model.predict(combined_input)[0]
    pred_index = np.argmax(probs)
    pred_label = label_encoder.inverse_transform([pred_index])[0]

    return pred_label, probs, warna, tekstur, bentuk  # Return the extracted features too

# API endpoint untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file'].read()
    try:
        # Decode image
        img = cv2.imdecode(np.frombuffer(file, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"error": "Image could not be read"}), 400

        # Ekstraksi fitur CNN dan manual
        fitur_cnn = cnn_base.predict(cnn_img, verbose=0)[0]
        warna, tekstur, bentuk = extract_fitur_manual(img)
        fitur_manual = np.concatenate([warna, tekstur, bentuk])

        # Gabung input dan prediksi
        combined_input = [np.expand_dims(fitur_cnn, axis=0), np.expand_dims(fitur_manual, axis=0)]
        probs = model.predict(combined_input, verbose=0)[0]

        # Format hasil prediksi
        pred_index = np.argmax(probs)
        pred_label = label_encoder.inverse_transform([pred_index])[0]
        prob_dict = {label_encoder.inverse_transform([i])[0]: round(float(p) * 100, 2) for i, p in enumerate(probs)}

        # Menyusun hasil prediksi dan fitur
        result = {
            "prediction": pred_label,
            "probabilities": prob_dict,
            "features": {
                "L_mean": warna[0],
                "L_std": warna[1],
                "a_mean": warna[2],
                "a_std": warna[3],
                "b_mean": warna[4],
                "b_std": warna[5],
                "contrast": tekstur[0],
                "dissimilarity": tekstur[1],
                "homogeneity": tekstur[2],
                "energy": tekstur[3],
                "correlation": tekstur[4],
                "area": bentuk[0],
                "perimeter": bentuk[1],
                "width": bentuk[2],
                "height": bentuk[3],
                "aspect_ratio": bentuk[4],
                "circularity": bentuk[5],
                "rectangularity": bentuk[6],
                "diameter": bentuk[7],
            }
        }

        # Pastikan hasilnya valid dan bisa di-serialize
        print(f"Prediction Result: {result}")

        # Kembalikan hasil sebagai JSON
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "success", "message": "API is running"})
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
