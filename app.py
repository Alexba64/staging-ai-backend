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
        image_url = data.get("image_url")
        style = data.get("style")
        room_type = data.get("room_type")

        model = replicate_client.models.get("fofr/room-staging")
        version = model.versions.get("cb1e1fd...")  # usa la versione corretta

        output = version.predict(
            image=image_url,
            style=style,
            room_type=room_type
        )

        return jsonify(output=output)
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
