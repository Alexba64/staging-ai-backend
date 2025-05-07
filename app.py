import os
from flask import Flask, jsonify, request
import replicate

app = Flask(__name__)

# Inizializzazione del client per l'API di replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Definire una route per la home
@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di staging AI!")

# Endpoint per elaborare l'immagine
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Prendi i dati inviati dal frontend
        data = request.json
        image_url = data['image_url']
        style = data['style']
        room_type = data['room_type']
        
        # Eseguiamo il modello di replicate
        model = replicate_client.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("v2.1")

        # Parametri da passare al modello
        prompt = f"A beautiful {room_type} with {style} style"
        output = version.predict(prompt=prompt, image=image_url)

        # Restituisci l'output al frontend
        return jsonify(output=output)

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

