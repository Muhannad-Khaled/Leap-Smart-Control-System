import asyncio
import websockets
import json
import os
import time
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

ESP_IP_FILE = "esp_ip.json"
PREDICT_API_WS_URL = "ws://localhost:8765"
PING_INTERVAL = 20
PING_TIMEOUT = 20
RECONNECT_DELAY = 3  # ثواني بين محاولات إعادة الاتصال

async def forward_data():
    while True:
        if not os.path.exists(ESP_IP_FILE):
            print("❌ ESP IP not registered yet. Make sure the ESP sent it via /register_ip.")
            await asyncio.sleep(RECONNECT_DELAY)
            continue

        with open(ESP_IP_FILE, "r") as f:
            esp_ip_data = json.load(f)
            esp_ip = esp_ip_data.get("ip")

        if not esp_ip:
            print("❌ No IP found in esp_ip.json.")
            await asyncio.sleep(RECONNECT_DELAY)
            continue

        ESP_WS_URL = f"ws://{esp_ip}:81"
        print(f"\n🔌 Connecting to ESP at {ESP_WS_URL}")
        print(f"🔗 Forwarding to prediction server at {PREDICT_API_WS_URL}")

        try:
            async with websockets.connect(ESP_WS_URL, ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT) as esp_ws, \
                       websockets.connect(PREDICT_API_WS_URL, ping_interval=PING_INTERVAL, ping_timeout=PING_TIMEOUT) as predict_ws:
                print("✅ Connected to both WebSockets\n")

                while True:
                    try:
                        msg = await esp_ws.recv()
                        print("📥 From ESP:", msg)

                        await predict_ws.send(msg)
                        print("📤 Sent to Predictor")

                    except (ConnectionClosedError, ConnectionClosedOK) as e:
                        print(f"⚠️  Connection closed during transfer: {e}")
                        break
                    except Exception as e:
                        print(f"❌ Error during transfer: {e}")
                        break

        except Exception as e:
            print(f"❌ Failed to connect or maintain connection: {e}")

        print(f"🔄 Retrying in {RECONNECT_DELAY} seconds...\n")
        await asyncio.sleep(RECONNECT_DELAY)

if __name__ == "__main__":
    try:
        asyncio.run(forward_data())
    except KeyboardInterrupt:
        print("🛑 Stopped by user.")
