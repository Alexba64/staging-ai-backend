from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
import os
import logging

# Configurazione del logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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

        # Log dei parametri ricevuti per il debug
        logger.debug(f"Ricevuti i parametri: image_url={image_url}, style={style}, room_type={room_type}")

        if not image_url or not style or not room_type:
            logger.error("Dati mancanti, assicurati di inviare image_url, style, room_type.")
            return jsonify({"error": "Dati incompleti."}), 400

        # Carica il modello
        model = replicate_client.models.get("fofr/room-staging")
        logger.debug("Modello caricato correttamente.")

        # Prendi la versione corretta del modello (aggiorna questo ID con la versione giusta)
        try:
            version = model.versions.get("cb1e1fd...")  # Usa la versione corretta qui
            logger.debug(f"Versione del modello trovata: {version}")
        except Exception as e:
            logger.error(f"Errore nel caricare la versione del modello: {str(e)}")
            return jsonify({"error": "Impossibile trovare la versione del modello."}), 404

        # Previsione
        output = version.predict(
            image=image_url,
            style=style,
            room_type=room_type
        )
        logger.debug(f"Output generato: {output}")

        return jsonify(output=output)

    except Exception as e:
        logger.error(f"Errore durante l'elaborazione dell'immagine: {str(e)}")
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

