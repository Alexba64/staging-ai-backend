from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
import os

app = Flask(__name__)
CORS(app)

# Carica API Key da variabili ambiente
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di Staging AI!")

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        prompt = data.get("prompt")

        if not image_url or not prompt:
            return jsonify(error="URL immagine o prompt mancanti"), 400

        # Esegui modello Replicate direttamente
        output = replicate.run(
            "stability-ai/stable-diffusion-img2img:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={
                "image": image_url,
                "prompt": prompt,
                "strength": 0.7,
                "num_outputs": 1,
                "guidance_scale": 7.5
            }
        )

        # Output è una lista di URL
        return jsonify(output=output[0])
    
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
