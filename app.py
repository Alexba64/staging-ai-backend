from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
import os

app = Flask(__name__)
CORS(app)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di staging AI!")

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        image_url = data['image_url']
        style = data['style']
        room_type = data['room_type']

        # Modello stabile e attivo su Replicate
        model = replicate_client.models.get("stability-ai/sdxl")
        version = model.versions.get("c6fbd0fd885d8d5c77478f9f9c71d600b6fae50d7ee6f02b1860c66a1301705b")

        prompt = f"interior of a {room_type} in {style} style"

        output = version.predict(prompt=prompt)

        return jsonify(output=output[0])
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
