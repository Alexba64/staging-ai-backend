import os
import logging
from flask import Flask, jsonify, request
from flask_cors import CORS
import replicate
from replicate.exceptions import ReplicateError

# Configurazione per logging
logging.basicConfig(level=logging.DEBUG)  # Imposta il livello di logging a DEBUG
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Abilita CORS per tutte le rotte
CORS(app)

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
        # Log dei dati ricevuti
        logger.debug("Ricevuta richiesta per /process_image")
        
        data = request.get_json()  # Otteniamo i dati inviati dal frontend
        logger.debug(f"Dati ricevuti: {data}")

        image_url = data.get('image_url')
        style = data.get('style')
        room_type = data.get('room_type')

        if not image_url or not style or not room_type:
            logger.error("Dati incompleti: assicurati di inviare image_url, style, room_type.")
            return jsonify({"error": "Dati incompleti."}), 400

        # Eseguiamo il modello di replicate
        logger.debug("Caricamento del modello...")
        
        model = replicate_client.models.get("stability-ai/stable-diffusion")
        logger.debug("Modello caricato, ora eseguo la previsione...")

        # Verifica e usa la versione corretta
        try:
            version = model.versions.get("2.1")  # Sostituisci con la versione 2.1, come discusso
            output = version.predict(prompt=f"A {room_type} styled in {style} with furniture", image=image_url)
            logger.debug(f"Elaborazione completata con successo. Output: {output}")
        except ReplicateError as e:
            logger.error(f"Errore nel caricare il modello o la versione: {e}")
            return jsonify({"error": "Modello non trovato o errore nella versione."}), 404

        return jsonify({"output": output})

    except Exception as e:
        # Log dell'errore
        logger.error(f"Errore durante l'elaborazione dell'immagine: {str(e)}")
        return jsonify({"error": f"Errore: {str(e)}"}), 500

if __name__ == "__main__":
    # Log per tracciare l'avvio dell'app
    logger.info("Avvio dell'app Flask...")

    # L'app di Flask su Render deve ascoltare sulla porta specificata nell'ambiente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
