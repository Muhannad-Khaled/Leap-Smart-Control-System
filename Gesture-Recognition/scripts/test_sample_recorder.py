import websocket
import json
import os
import time

# اقرأ IP من الملف اللي سجله Flask
def get_esp_ip():
    try:
        with open("esp_ip.json", "r") as f:
            return json.load(f)["ip"]
    except:
        print("❌ ESP IP not found. Make sure ESP sent it.")
        exit()

# ⚙️ إعداد المسار
BASE_DIR = "scripts/data/test_samples"
os.makedirs(BASE_DIR, exist_ok=True)

sample_count = len([f for f in os.listdir(BASE_DIR) if f.endswith(".csv")]) + 1
buffer = []
start_time = None
RECORD_DURATION = 2.0  # تسجيل لمدة ثانيتين

ESP_IP = get_esp_ip()

def on_message(ws, message):
    global buffer, start_time, sample_count

    try:
        data = json.loads(message)
        if "data" not in data or not isinstance(data["data"], list):
            return

        for sample in data["data"]:
            row = f"{sample['accel_x']},{sample['accel_y']},{sample['accel_z']}," \
                  f"{sample['gyro_x']},{sample['gyro_y']},{sample['gyro_z']}"
            buffer.append(row)

        if start_time is None:
            start_time = time.time()

        if time.time() - start_time >= RECORD_DURATION:
            filename = f"{BASE_DIR}/sample_{sample_count:03d}.csv"
            with open(filename, 'w') as f:
                f.write("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
                f.write("\n".join(buffer))
            print(f"💾 Saved test sample: {filename}")
            sample_count += 1
            buffer = []
            start_time = None

    except Exception as e:
        print(f"❌ Error: {e}")

def on_error(ws, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔴 WebSocket closed")

def on_open(ws):
    print("🟢 Connected to ESP WebSocket. Move now to capture...")

# ✅ تشغيل التطبيق
if __name__ == "__main__":
    ws = websocket.WebSocketApp(f"ws://{ESP_IP}:81/",
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_error=on_error,
                                 on_close=on_close)
    ws.run_forever()
