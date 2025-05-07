import os
from flask import Flask, jsonify, request
import replicate
from flask_cors import CORS

# Inizializzazione dell'app Flask e configurazione CORS
app = Flask(__name__)

# Abilita CORS per tutte le origini
CORS(app)

# Inizializzazione del client per l'API di replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Definire una route per la home
@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di staging AI!")

# Esempio di endpoint per ottenere una previsione da un modello di replicate
@app.route('/get_model_output')
def get_model_output():
    try:
        # Eseguiamo il modello di replicate qui
        model = replicate_client.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("xx.xx.x")
        
        # Parametri da passare al modello
        output = version.predict(prompt="A beautiful living room with modern furniture")

        return jsonify(output)
    
    except Exception as e:
        return jsonify(error=str(e)), 500

# Endpoint per l'upload dell'immagine
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        # Ottieni il file caricato dal frontend
        file = request.files['file']
        
        # Controlla se il file è valido
        if not file:
            return jsonify({"error": "Nessun file fornito"}), 400
        
        # Esegui una previsione con il modello, usando il file caricato
        model = replicate_client.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("xx.xx.x")
        
        # Aggiungi la logica per processare l'immagine, ad esempio utilizzando l'immagine come input per il modello
        # Qui potrebbe esserci un'ulteriore logica per elaborare l'immagine prima di inviarla a Replicate
        output = version.predict(image=file)

        return jsonify(output)
    
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # L'app di Flask su Render deve ascoltare sulla porta specificata nell'ambiente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
