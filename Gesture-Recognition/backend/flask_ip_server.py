from flask import Flask, request, jsonify
import json

app = Flask(__name__)

@app.route("/register_ip", methods=["POST"])
def register_ip():
    data = request.get_json()
    esp_ip = data.get("ip", "")
    if esp_ip:
        with open("esp_ip.json", "w") as f:
            json.dump({"ip": esp_ip}, f)
        print(f"✅ ESP IP Registered: {esp_ip}")
        return jsonify({"status": "ok"})
    return jsonify({"error": "Missing IP"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
