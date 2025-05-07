import os
from flask import Flask, request, jsonify
import replicate
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Abilita CORS per consentire richieste dal frontend

# Inizializza il client Replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di staging AI!")

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_url = data.get("image_url")
        style = data.get("style", "Modern")
        room_type = data.get("room_type", "living room")

        prompt = f"A {style} {room_type} interior design"

        # Eseguiamo il modello (es. scenex/room-staging)
        model = replicate_client.models.get("scenex/room-staging")
        version = model.versions.get("fc8b3e7f4cd9164c145482da2b3debf7689ff2c16e2bb1df89d5ee6837f35f48")
        
        output = version.predict(image=image_url, prompt=prompt)

        return jsonify(output=output)

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
