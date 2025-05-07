import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)

# Imposta il token API di Replicate (deve essere presente come variabile d'ambiente)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

# Verifica che il token sia presente
if not REPLICATE_API_TOKEN:
    raise ValueError("La variabile REPLICATE_API_TOKEN non è stata impostata.")

# Route principale
@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend AI!")

# Route per processare l'immagine
@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()

        image_url = data.get("image_url")
        style = data.get("style")
        room_type = data.get("room_type")

        if not image_url or not style or not room_type:
            return jsonify(error="image_url, style e room_type sono richiesti."), 400

        prompt = f"A {room_type} styled in {style}"

        # Esegui il modello
        output = replicate.run(
            "stability-ai/stable-diffusion-img2img",
            input={
                "image": image_url,
                "prompt": prompt,
                "strength": 0.75,
                "num_outputs": 1
            }
        )

        return jsonify({"output": output})

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
