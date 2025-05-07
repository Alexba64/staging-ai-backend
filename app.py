import os
from flask import Flask, request, jsonify
import replicate

app = Flask(__name__)

# Inizializzazione del client per l'API di replicate
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Definire una route per la home
@app.route('/')
def home():
    return jsonify(message="Benvenuto nel backend di staging AI!")

# Endpoint per caricare un'immagine e ottenere una previsione
@app.route('/upload_image', methods=['POST'])
def upload_image():
    try:
        # Controllo che ci sia un file nell'input
        if 'file' not in request.files:
            return jsonify({"error": "Nessun file fornito"}), 400
        
        # Otteniamo il file caricato dal frontend
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "Nome del file non valido"}), 400
        
        # Salviamo temporaneamente il file caricato
        file_path = os.path.join("uploads", file.filename)
        file.save(file_path)
        
        # Log per debug: stampa il nome del file
        print(f"File caricato: {file.filename}")

        # Carichiamo il modello di Replicate
        model = replicate_client.models.get("stability-ai/stable-diffusion")
        version = model.versions.get("xx.xx.x")  # Assicurati di sostituire con la versione giusta
        
        # Eseguiamo la previsione passando l'immagine
        output = version.predict(image=file_path)
        
        # Ritorniamo il risultato come risposta JSON
        return jsonify(output)
    
    except Exception as e:
        print(f"Errore durante l'elaborazione: {str(e)}")  # Aggiungi log di errore
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    # L'app di Flask su Render deve ascoltare sulla porta specificata nell'ambiente
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
