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

# ✅ إعدادات المستخدم
GESTURE = input("📌 Enter gesture name (e.g., push_pull): ").strip()
SET = input("📂 Enter set (A / B / C / D / E): ").strip().upper()

if SET not in ["A", "B", "C", "D", "E"]:
    print("❌ Invalid set. Use A / B / C / D / E only.")
    exit()

# ✅ إعداد المسار
BASE_DIR = f"data/{GESTURE}/set_{SET}"
os.makedirs(BASE_DIR, exist_ok=True)

sample_count = len(os.listdir(BASE_DIR)) + 1
buffer = []
start_time = None
RECORD_DURATION = 2.0  # الثواني المطلوبة لكل تسجيل

ESP_IP = get_esp_ip()

# ✅ استلام الرسائل من ESP
def on_message(ws, message):
    global buffer, start_time, sample_count

    try:
        data = json.loads(message)
        row = f"{data['accel_x']},{data['accel_y']},{data['accel_z']}," \
              f"{data['gyro_x']},{data['gyro_y']},{data['gyro_z']}"

        if start_time is None:
            start_time = time.time()

        buffer.append(row)

        if time.time() - start_time >= RECORD_DURATION:
            filename = f"{BASE_DIR}/sample_{sample_count:03d}.csv"
            with open(filename, 'w') as f:
                f.write("acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z\n")
                f.write("\n".join(buffer))
            print(f"💾 Saved: {filename}")
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
    print("🟢 Connected to ESP WebSocket")

# ✅ تشغيل التطبيق
if __name__ == "__main__":
    ws = websocket.WebSocketApp(f"ws://{ESP_IP}:81/",
                                 on_open=on_open,
                                 on_message=on_message,
                                 on_error=on_error,
                                 on_close=on_close)
    ws.run_forever()
