import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model  # type: ignore
import joblib
import os
from datetime import datetime
import csv
import time
from dotenv import load_dotenv

import firebase_admin
from firebase_admin import credentials, db

# Load environment variables from .env file
load_dotenv()

from preprocessing.clean_data import clean_dataframe
from preprocessing.normalize import normalize_dataframe

# === Paths ===
MODEL_PATH = "model/lstm_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
SCALER_PATH = "model/scaler_global.pkl"
PREDICTION_FILE = "backend/latest_prediction.json"
PREDICTION_LOG = "backend/prediction_log.csv"
WINDOW_SIZE = 50

# === Load ML model ===
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

# === Firebase Init ===
if not firebase_admin._apps:
    firebase_creds = {
        "type": os.environ.get("FIREBASE_TYPE", "service_account"),
        "project_id": os.environ.get("FIREBASE_PROJECT_ID"),
        "private_key_id": os.environ.get("FIREBASE_PRIVATE_KEY_ID"),
        "private_key": os.environ.get("FIREBASE_PRIVATE_KEY", "").replace("\\n", "\n"),
        "client_email": os.environ.get("FIREBASE_CLIENT_EMAIL"),
        "client_id": os.environ.get("FIREBASE_CLIENT_ID"),
        "auth_uri": os.environ.get("FIREBASE_AUTH_URI"),
        "token_uri": os.environ.get("FIREBASE_TOKEN_URI"),
        "auth_provider_x509_cert_url": os.environ.get("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
        "client_x509_cert_url": os.environ.get("FIREBASE_CLIENT_X509_CERT_URL"),
    }
    cred = credentials.Certificate(firebase_creds)
    firebase_admin.initialize_app(cred, {
        "databaseURL": os.environ.get("FIREBASE_DATABASE_URL", "https://leap-smart-band-default-rtdb.firebaseio.com/")
    })

# === Globals ===
clients = set()
data_buffer = []
last_result = None
last_prediction_time = 0
prediction_cooldown = 0.3
movement_threshold = 0.7

# === Movement Threshold Logic ===
def movement_detected(df):
    std_dev = df.std().mean()
    return std_dev > movement_threshold

# === WebSocket Handler ===
async def handle_client(websocket):
    global data_buffer
    clients.add(websocket)
    print("🟢 New WebSocket connection")

    try:
        async for message in websocket:
            try:
                incoming = json.loads(message)
                if not isinstance(incoming, dict) or "data" not in incoming or "gesture_type" not in incoming:
                    continue

                new_data = incoming["data"]
                if not isinstance(new_data, list) or not all(isinstance(row, dict) for row in new_data):
                    continue

                data_buffer += new_data
                if len(data_buffer) > 200:
                    data_buffer = data_buffer[-200:]

            except Exception as e:
                print(f"❌ Error parsing message: {e}")
    finally:
        clients.remove(websocket)
        print("🔴 WebSocket client disconnected")

# === Prediction Logic ===
async def prediction_loop():
    global last_result, data_buffer, last_prediction_time

    os.makedirs(os.path.dirname(PREDICTION_LOG), exist_ok=True)
    if not os.path.exists(PREDICTION_LOG):
        with open(PREDICTION_LOG, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "label", "confidence", "raw_window"])

    while True:
        await asyncio.sleep(0.01)

        if len(data_buffer) < WINDOW_SIZE:
            continue

        window_data = data_buffer[-WINDOW_SIZE:]
        df = pd.DataFrame(window_data)
        df = df.rename(columns={
            "accel_x": "acc_x",
            "accel_y": "acc_y",
            "accel_z": "acc_z"
        })

        if df.empty or df.shape[0] < WINDOW_SIZE:
            continue

        if not movement_detected(df):
            continue

        if time.time() - last_prediction_time < prediction_cooldown:
            continue

        input_tensor = np.expand_dims(df.values, axis=0)
        prediction = model.predict(input_tensor, verbose=0)

        confidence = float(np.max(prediction))
        label_index = int(np.argmax(prediction))
        label = encoder.inverse_transform([label_index])[0] if confidence >= 0.6 else "Unknown movement"

        result = {
            "result": label,
            "confidence": round(confidence, 2)
        }

        if result != last_result:
            last_result = result
            last_prediction_time = time.time()

            try:
                with open(PREDICTION_LOG, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        label,
                        round(confidence, 2),
                        json.dumps(window_data)
                    ])
            except Exception as e:
                print(f"❌ Failed to write to prediction log: {e}")

            for ws in clients.copy():
                try:
                    await ws.send(json.dumps(result))
                except:
                    pass

            with open(PREDICTION_FILE, "w") as f:
                json.dump(result, f)

            print(f"📡 Predicted: {label} ({confidence:.2f})")

            # === Firebase Actions ===
            try:
                if label == "push_pull":
                    light_ref = db.reference("Lights/stat")
                    current = light_ref.get()
                    new_val = str(1 - int(current)) if current is not None else "1"
                    light_ref.set(new_val)
                    db.reference("Lights/01").set(new_val == "1")
                    print(f"💡 Light toggled to {new_val}")

                elif label == "circle_cw":
                    door_ref = db.reference("Door/stat")
                    current = door_ref.get()
                    new_val = str(1 - int(current)) if current is not None else "1"
                    door_ref.set(new_val)
                    db.reference("Door/01").set(new_val == "1")
                    print(f"🚪 Door toggled to {new_val}")

            except Exception as e:
                print(f"❌ Firebase update error: {e}")

        data_buffer.pop(0)

# === Run Server ===
async def main():
    port = int(os.environ.get("WS_PORT", 8765))
    print(f"🚀 WebSocket server running on ws://0.0.0.0:{port}")
    server = await websockets.serve(
        handle_client,
        "0.0.0.0",
        port,
        max_size=2**20,
        ping_interval=30,
        ping_timeout=30
    )
    await asyncio.gather(server.wait_closed(), prediction_loop())

# === Exported function for combined_runner ===
def start_ws_server():
    asyncio.run(main())

# === CLI run ===
if __name__ == "__main__":
    start_ws_server()
