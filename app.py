import os
from flask import Flask, jsonify
import replicate

app = Flask(__name__)

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

if __name__ == "__main__":
    # L'app di Flask su Render deve ascoltare sulla porta specificata nell'ambiente
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
