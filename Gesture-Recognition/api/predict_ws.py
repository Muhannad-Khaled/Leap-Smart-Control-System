# import asyncio
# import websockets
# import json
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model
# import joblib

# from preprocessing.clean_data import clean_dataframe
# from preprocessing.normalize import normalize_dataframe

# MODEL_PATH = "model/lstm_model.h5"
# ENCODER_PATH = "model/label_encoder.pkl"
# WINDOW_SIZE = 50

# model = load_model(MODEL_PATH)
# encoder = joblib.load(ENCODER_PATH)

# clients = set()
# data_buffer = []

# async def handle_client(websocket):
#     global data_buffer
#     clients.add(websocket)
#     print("🟢 New WebSocket connection")

#     try:
#         async for message in websocket:
#             print("📥 Received message:", message)

#             try:
#                 incoming = json.loads(message)

#                 if not isinstance(incoming, dict) or "data" not in incoming or "gesture_type" not in incoming:
#                     await websocket.send(json.dumps({"error": "Invalid format. Expected gesture_type and data."}))
#                     continue

#                 gesture_type = incoming["gesture_type"]
#                 new_data = incoming["data"]

#                 if not isinstance(new_data, list) or not all(isinstance(row, dict) for row in new_data):
#                     await websocket.send(json.dumps({"error": "Invalid data format. Expected list of dict sensor readings."}))
#                     continue

#                 data_buffer += new_data
#                 print(f"🧠 Buffer size: {len(data_buffer)}")

#                 if gesture_type == "realtime" and len(data_buffer) >= WINDOW_SIZE:
#                     window_data = data_buffer[-WINDOW_SIZE:]
#                     df = pd.DataFrame(window_data)

#                     # ✅ تعديل أسماء الأعمدة لتطابق التدريب
#                     df = df.rename(columns={
#                         "accel_x": "acc_x",
#                         "accel_y": "acc_y",
#                         "accel_z": "acc_z"
#                     })

#                     df = clean_dataframe(df)
#                     if df.empty or df.shape[0] < WINDOW_SIZE:
#                         await websocket.send(json.dumps({"error": "Invalid/insufficient data after cleaning."}))
#                         continue

#                     df = normalize_dataframe(df)
#                     input_tensor = np.expand_dims(df.values, axis=0)

#                     prediction = model.predict(input_tensor)
#                     confidence = float(np.max(prediction))
#                     label_index = int(np.argmax(prediction))
#                     label = encoder.inverse_transform([label_index])[0] if confidence >= 0.6 else "Unknown movement"

#                     await websocket.send(json.dumps({
#                         "result": label,
#                         "confidence": round(confidence, 2)
#                     }))
#                     print(f"📡 Predicted: {label} ({confidence:.2f})")

#                     data_buffer.pop(0)

#             except Exception as e:
#                 await websocket.send(json.dumps({"error": str(e)}))
#                 print(f"❌ Error: {e}")
#     finally:
#         clients.remove(websocket)
#         print("🔴 WebSocket client disconnected")

# async def main():
#     print("🚀 WebSocket server running on ws://0.0.0.0:8765")
#     async with websockets.serve(handle_client, "0.0.0.0", 8765, max_size=2**20):
#         await asyncio.Future()

# if __name__ == "__main__":
#     asyncio.run(main())


import asyncio
import websockets
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib
import os

from preprocessing.clean_data import clean_dataframe
from preprocessing.normalize import normalize_dataframe

MODEL_PATH = "model/lstm_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"
WINDOW_SIZE = 50
PREDICTION_FILE = "latest_prediction.json"

model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

clients = set()
data_buffer = []

def save_prediction_to_file(label, confidence):
    prediction = {
        "result": label,
        "confidence": round(confidence, 2)
    }
    with open(PREDICTION_FILE, "w") as f:
        json.dump(prediction, f)
    print(f"💾 Saved prediction to {PREDICTION_FILE}")

async def handle_client(websocket):
    global data_buffer
    clients.add(websocket)
    print("🟢 New WebSocket connection")

    try:
        async for message in websocket:
            print("📥 Received message:", message)

            try:
                incoming = json.loads(message)

                if not isinstance(incoming, dict) or "data" not in incoming or "gesture_type" not in incoming:
                    await websocket.send(json.dumps({"error": "Invalid format. Expected gesture_type and data."}))
                    continue

                gesture_type = incoming["gesture_type"]
                new_data = incoming["data"]

                if not isinstance(new_data, list) or not all(isinstance(row, dict) for row in new_data):
                    await websocket.send(json.dumps({"error": "Invalid data format. Expected list of dict sensor readings."}))
                    continue

                data_buffer += new_data
                print(f"🧠 Buffer size: {len(data_buffer)}")

                if gesture_type == "realtime" and len(data_buffer) >= WINDOW_SIZE:
                    window_data = data_buffer[-WINDOW_SIZE:]
                    df = pd.DataFrame(window_data)

                    # ✅ Rename columns
                    df = df.rename(columns={
                        "accel_x": "acc_x",
                        "accel_y": "acc_y",
                        "accel_z": "acc_z"
                    })

                    df = clean_dataframe(df)
                    if df.empty or df.shape[0] < WINDOW_SIZE:
                        await websocket.send(json.dumps({"error": "Invalid/insufficient data after cleaning."}))
                        continue

                    df = normalize_dataframe(df)
                    input_tensor = np.expand_dims(df.values, axis=0)

                    prediction = model.predict(input_tensor)
                    confidence = float(np.max(prediction))
                    label_index = int(np.argmax(prediction))
                    label = encoder.inverse_transform([label_index])[0] if confidence >= 0.6 else "Unknown movement"

                    # ✅ Save to file
                    save_prediction_to_file(label, confidence)

                    # ✅ Optional: send back to client
                    await websocket.send(json.dumps({
                        "result": label,
                        "confidence": round(confidence, 2)
                    }))

                    print(f"📡 Predicted: {label} ({confidence:.2f})")

                    data_buffer.pop(0)

            except Exception as e:
                await websocket.send(json.dumps({"error": str(e)}))
                print(f"❌ Error: {e}")
    finally:
        clients.remove(websocket)
        print("🔴 WebSocket client disconnected")

async def main():
    print("🚀 WebSocket server running on ws://0.0.0.0:8765")
    async with websockets.serve(handle_client, "0.0.0.0", 8765, max_size=2**20):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())
