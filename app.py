import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import replicate

# Configurazione logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
os.environ["REPLICATE_API_TOKEN"] = REPLICATE_API_TOKEN  # richiesto da replicate.run

@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di staging AI!")

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json()
        logger.debug(f"Dati ricevuti: {data}")

        image_url = data.get("image_url")
        style = data.get("style")
        room_type = data.get("room_type")

        if not image_url or not style or not room_type:
            logger.error("Dati mancanti.")
            return jsonify({"error": "image_url, style e room_type sono obbligatori"}), 400

        prompt = f"A {room_type} styled in {style} with furniture"

        logger.debug("Invio richiesta a Replicate...")
        output = replicate.run(
            "stability-ai/stable-diffusion:ac732df83cea7fff18b8472768c88ad041fa750ff7682a21affe81863cbe77e4",
            input={"prompt": prompt}
        )

        logger.debug(f"Output ricevuto: {output}")
        return jsonify({"output": output})

    except Exception as e:
        logger.error(f"Errore durante l'elaborazione: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    logger.info("Avvio del backend Flask...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
