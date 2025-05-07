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

# Route per processare l'immagine
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()  # Otteniamo i dati inviati dal frontend
        image_url = data.get('image_url')
        style = data.get('style')
        room_type = data.get('room_type')

        # Eseguiamo il modello di replicate qui
        model = replicate_client.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("2.1")

        # Parametri da passare al modello
        output = version.predict(prompt=f"A {room_type} styled in {style} with furniture", image=image_url)

        return jsonify({"output": output})  # Invia l'immagine generata come risultato

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # L'app di Flask su Render deve ascoltare sulla porta specificata nell'ambiente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
