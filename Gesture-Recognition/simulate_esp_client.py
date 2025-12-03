# import websocket
# import json
# import time
# import random

# PREDICT_WS_URL = "ws://localhost:8765"

# def generate_data_point():
#     return {
#         "accel_x": round(random.uniform(-2, 2), 3),
#         "accel_y": round(random.uniform(-2, 2), 3),
#         "accel_z": round(random.uniform(-2, 2), 3),
#         "gyro_x": round(random.uniform(-180, 180), 3),
#         "gyro_y": round(random.uniform(-180, 180), 3),
#         "gyro_z": round(random.uniform(-180, 180), 3),
#     }

# def simulate_realtime_stream():
#     ws = websocket.WebSocket()
#     ws.connect(PREDICT_WS_URL)
#     print("🟢 Connected to prediction server")

#     data_batch = []

#     for i in range(60):  # Send 60 readings (more than WINDOW_SIZE)
#         data_batch.append(generate_data_point())

#         # Send data every few samples to mimic real-time streaming
#         if len(data_batch) >= 3:
#             message = {
#                 "gesture_type": "realtime",
#                 "data": data_batch
#             }
#             ws.send(json.dumps(message))
#             print(f"📤 Sent {len(data_batch)} readings")
#             data_batch = []
#             time.sleep(0.1)

#     ws.close()
#     print("🔴 Disconnected")

# if __name__ == "__main__":
#     simulate_realtime_stream()


import asyncio
import websockets
import json
import random
import time

PREDICTOR_WS_URL = "ws://localhost:8765"

def generate_imu_sample():
    return {
        "gesture_type": "realtime",
        "data": [
            {
                "accel_x": round(random.uniform(-2, 2), 6),
                "accel_y": round(random.uniform(-10, 10), 6),
                "accel_z": round(random.uniform(6, 10), 6),
                "gyro_x": round(random.uniform(-0.05, 0.05), 6),
                "gyro_y": round(random.uniform(-0.05, 0.05), 6),
                "gyro_z": round(random.uniform(-0.05, 0.05), 6)
            }
        ]
    }

async def send_fake_data():
    async with websockets.connect(PREDICTOR_WS_URL) as ws:
        print("✅ Connected to prediction WebSocket")
        while True:
            sample = generate_imu_sample()
            await ws.send(json.dumps(sample))
            print("📤 Sent:", sample)
            await asyncio.sleep(0.05)  # simulate 100Hz => 50ms

if __name__ == "__main__":
    try:
        asyncio.run(send_fake_data())
    except KeyboardInterrupt:
        print("🛑 Simulator stopped by user")
