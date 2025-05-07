from flask import Flask, request, jsonify
import requests
import os
import time

app = Flask(__name__)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"
MODEL_VERSION_ID = "ID_DEL_MODELLO"  # Sostituisci con l'ID del modello Replicate

@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    image_url = data.get("image_url")
    style = data.get("style")
    room_type = data.get("room_type")

    prompt = f"A {style} {room_type} with modern decor and clean design"
    headers = {
        "Authorization": f"Token {REPLICATE_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "version": MODEL_VERSION_ID,
        "input": {
            "image": image_url,
            "prompt": prompt
        }
    }

    response = requests.post(REPLICATE_API_URL, headers=headers, json=payload)
    if response.status_code != 201:
        return jsonify({"error": "Errore nella richiesta a Replicate"}), 500

    prediction = response.json()
    prediction_id = prediction["id"]

    # Polling per ottenere il risultato
    for _ in range(20):
        time.sleep(1)
        poll_response = requests.get(f"{REPLICATE_API_URL}/{prediction_id}", headers=headers)
        poll_data = poll_response.json()
        if poll_data["status"] == "succeeded":
            return jsonify({"output": poll_data["output"]})
        elif poll_data["status"] == "failed":
            return jsonify({"error": "Generazione fallita"}), 500

    return jsonify({"error": "Timeout nella generazione dell'immagine"}), 500

if __name__ == "__main__":
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
