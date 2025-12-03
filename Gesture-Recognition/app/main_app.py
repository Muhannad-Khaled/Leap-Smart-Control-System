# import streamlit as st
# import websocket
# import threading
# import json
# import time
# from streamlit_autorefresh import st_autorefresh
# import queue

# import streamlit.runtime.scriptrunner.script_run_context as script_context
# from streamlit.runtime.scriptrunner import add_script_run_ctx

# # ✅ Message queue for thread-safe communication
# message_queue = queue.Queue()

# PREDICT_API_WS_URL = "ws://127.0.0.1:8765"

# st.set_page_config(page_title="Live Gesture Prediction", layout="wide")
# st.title("🖐️ Real-Time Gesture Prediction")

# # ✅ Auto-refresh every 1 second
# st_autorefresh(interval=1000, limit=None, key="refresh")

# # ✅ UI placeholders
# status_placeholder = st.empty()
# result_placeholder = st.empty()

# # ✅ Session state variables
# if "prediction" not in st.session_state:
#     st.session_state.prediction = None
# if "error" not in st.session_state:
#     st.session_state.error = None
# if "connected" not in st.session_state:
#     st.session_state.connected = False
# if "started" not in st.session_state:
#     st.session_state.started = False


# # ✅ WebSocket listener (runs in background thread)
# def listen():
#     def on_message(ws, message):
#         message_queue.put(message)

#     def on_error(ws, error):
#         message_queue.put(json.dumps({"error": f"WebSocket error: {error}"}))

#     def on_close(ws, close_status_code, close_msg):
#         message_queue.put(json.dumps({"status": "disconnected"}))

#     def on_open(ws):
#         print("🔥 on_open triggered")
#         message_queue.put(json.dumps({"status": "connected"}))

#         # ✅ Force rerun from inside the thread
#         ctx = script_context.get_script_run_ctx()
#         if ctx is not None:
#             add_script_run_ctx(threading.current_thread(), ctx)
#             st.experimental_rerun()

#     print("🚀 Launching WebSocket connection to", PREDICT_API_WS_URL)
#     ws = websocket.WebSocketApp(
#         PREDICT_API_WS_URL,
#         on_open=on_open,
#         on_message=on_message,
#         on_error=on_error,
#         on_close=on_close
#     )
#     ws.run_forever()


# # ✅ Start Listening
# if st.button("Start Listening") and not st.session_state.started:
#     print("🔄 Starting WebSocket listener thread")
#     threading.Thread(target=listen, daemon=True).start()
#     st.session_state.started = True
#     st.success("🔄 Listening started...")


# # ✅ Process all messages from the queue
# while not message_queue.empty():
#     raw_message = message_queue.get()
#     print("📥 Pulled:", raw_message)
#     try:
#         data = json.loads(raw_message)

#         if "result" in data:
#             st.session_state.prediction = f"✅ Prediction: {data['result']} (Confidence: {data['confidence']})"
#             st.session_state.error = None

#         elif "error" in data:
#             st.session_state.error = f"❌ Error: {data['error']}"
#             st.session_state.prediction = None

#         elif "status" in data:
#             st.session_state.connected = (data["status"] == "connected")
#             print("🔁 Updated connected =", st.session_state.connected)

#     except Exception as e:
#         st.session_state.error = f"❌ Parse error: {e}"


# # ✅ Display connection status
# if st.session_state.connected:
#     status_placeholder.success("🟢 Connected to prediction server")
# else:
#     status_placeholder.warning("🔌 Not connected")

# # ✅ Display prediction or error
# if st.session_state.prediction:
#     result_placeholder.success(st.session_state.prediction)
# elif st.session_state.error:
#     result_placeholder.error(st.session_state.error)
# else:
#     result_placeholder.info("⏳ Waiting for predictions...")


import streamlit as st
import requests
import time
from streamlit_autorefresh import st_autorefresh

# إعداد الصفحة
st.set_page_config(page_title="Real-Time Gesture Prediction", layout="centered")
st.title("🖐️ Real-Time Gesture Prediction")

# ⏱️ تحديث تلقائي كل ثانية
st_autorefresh(interval=1000, limit=None, key="auto_refresh")

# الحالة الحالية للعرض
status_placeholder = st.empty()
result_placeholder = st.empty()

# ⚙️ إعداد Session State
if "prediction" not in st.session_state:
    st.session_state.prediction = None
if "error" not in st.session_state:
    st.session_state.error = None

# 🔁 استدعاء REST API
API_URL = "http://127.0.0.1:5000/get_latest_prediction"

try:
    response = requests.get(API_URL, timeout=1)

    if response.status_code == 200:
        data = response.json()
        if data["status"] == "ok":
            label = data["data"]["label"]
            confidence = data["data"]["confidence"]
            st.session_state.prediction = f"✅ Prediction: {label} (Confidence: {confidence})"
            st.session_state.error = None
        else:
            st.session_state.error = f"⚠️ {data.get('message', 'Unknown response')}"
            st.session_state.prediction = None
    else:
        st.session_state.error = f"❌ Server error: {response.status_code}"
        st.session_state.prediction = None

except Exception as e:
    st.session_state.error = f"🔌 Connection error: {str(e)}"
    st.session_state.prediction = None

# ✅ عرض النتائج
if st.session_state.prediction:
    status_placeholder.success("🟢 Connected to prediction API")
    result_placeholder.success(st.session_state.prediction)
elif st.session_state.error:
    status_placeholder.error("🔴 Disconnected from API")
    result_placeholder.error(st.session_state.error)
else:
    status_placeholder.info("🕒 Waiting for prediction...")
    result_placeholder.info("⏳ No prediction yet.")
