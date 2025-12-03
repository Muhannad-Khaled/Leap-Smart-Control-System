# test_ws_connection.py
import websocket

try:
    ws = websocket.create_connection("ws://localhost:8765")
    print("✅ Connected successfully")
    ws.close()
except Exception as e:
    print("❌ Connection failed:", e)
