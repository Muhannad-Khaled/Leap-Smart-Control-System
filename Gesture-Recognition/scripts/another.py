import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import joblib
import os

from preprocessing.normalize import normalize_dataframe

MODEL_PATH = "model/lstm_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
SCALER_PATH = "model/scaler_global.pkl"
PREDICTION_FILE = "backend/latest_prediction.json"
WINDOW_SIZE = 50

model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)
scaler = joblib.load(SCALER_PATH)

clients = set()
data_buffer = []
last_prediction = None  # ✅ علشان ما نكررش نفس الحركة كل مرة

async def handle_client(websocket):
    global data_buffer, last_prediction
    clients.add(websocket)
    print("🟢 New WebSocket connection")

    try:
        async for message in websocket:
            try:
                incoming = json.loads(message)
                if "data" not in incoming or not isinstance(incoming["data"], list):
                    await safe_send(websocket, {"error": "Invalid data format"})
                    continue

                data_buffer += incoming["data"]

                # حافظ على آخر 100 نقطة فقط
                if len(data_buffer) > 100:
                    data_buffer = data_buffer[-100:]

                # لو عدد كافي للتنبؤ
                if len(data_buffer) >= WINDOW_SIZE:
                    window = data_buffer[-WINDOW_SIZE:]
                    df = pd.DataFrame(window)

                    df = df.rename(columns={
                        "accel_x": "acc_x",
                        "accel_y": "acc_y",
                        "accel_z": "acc_z"
                    })

                    # ✅ Apply normalization (مهم جدًا)
                    df = normalize_dataframe(df, scaler_path=SCALER_PATH)

                    input_tensor = np.expand_dims(df.values, axis=0)
                    prediction = model.predict(input_tensor, verbose=0)

                    confidence = float(np.max(prediction))
                    label_index = int(np.argmax(prediction))
                    label = encoder.inverse_transform([label_index])[0] if confidence >= 0.6 else "Unknown movement"

                    result = {
                        "result": label,
                        "confidence": round(confidence, 2)
                    }

                    # ✅ ارسال فقط لو النتيجة مختلفة عن السابقة
                    if result != last_prediction:
                        await safe_send(websocket, result)
                        with open(PREDICTION_FILE, "w") as f:
                            json.dump(result, f)
                        print(f"📡 Predicted: {label} ({confidence:.2f})")
                        last_prediction = result

            except Exception as e:
                await safe_send(websocket, {"error": str(e)})
                print(f"❌ Error: {e}")

    finally:
        clients.remove(websocket)
        print("🔴 WebSocket client disconnected")

async def safe_send(ws, message_dict):
    try:
        await ws.send(json.dumps(message_dict))
    except Exception as e:
        print(f"❌ Failed to send to client: {e}")

async def main():
    print("🚀 WebSocket server running on ws://0.0.0.0:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
