import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from preprocessing.clean_data import clean_dataframe
from preprocessing.normalize import normalize_dataframe

# إعدادات المسارات
SCALER_PATH = "model/scaler_global.pkl"
MODEL_PATH = "model/lstm_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
TEST_DIR = "scripts/data/test_samples"
WINDOW_SIZE = 50
CONFIDENCE_THRESHOLD = 0.6

# تحميل الموديل والـ encoder
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

def predict_file(filepath):
    df = pd.read_csv(filepath)
    df = clean_dataframe(df)

    if df.empty or df.shape[0] < WINDOW_SIZE:
        return ["❌ Not enough valid data"]

    df = normalize_dataframe(df, scaler)

    predictions = []
    for start in range(0, len(df) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = df.iloc[start:start + WINDOW_SIZE]
        input_tensor = np.expand_dims(window.values, axis=0)
        
        prediction = model.predict(input_tensor, verbose=0)
        confidence = float(np.max(prediction))
        label_index = int(np.argmax(prediction))

        if confidence >= CONFIDENCE_THRESHOLD:
            label = encoder.inverse_transform([label_index])[0]
        else:
            label = "Unknown"

        predictions.append(f"{label} ({confidence:.2f})")

    return predictions

# تنفيذ التنبؤ على جميع الملفات
if __name__ == "__main__":
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".csv")])

    if not files:
        print("📂 No CSV files found in test_samples.")
        exit()

    print(f"🔍 Found {len(files)} files in '{TEST_DIR}':\n")

    for file in files:
        path = os.path.join(TEST_DIR, file)
        predictions = predict_file(path)
        print(f"\n📄 {file} Predictions:")
        for i, result in enumerate(predictions):
            print(f"  └─ Window {i+1}: {result}")
