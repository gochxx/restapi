from flask import Flask, request, jsonify
from flask_cors import CORS
from forecast import ForecastModel
import logging
from typing import Any, Dict, Tuple

# Logging-Konfiguration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


app = Flask(__name__)


CORS(app) # alle Anfraen erlauben!
#CORS(app, resources={r"/predict": {"origins": "http://webserver:80"}}) # für ausführen im Container!
#CORS(app, resources={r"/predict": {"origins": "http://127.0.0.1:5500"}}) # Nur für lokale Ausführung
#CORS(app, resources={r"/predict": {"origins": "http://192.168.178.26:5500"}}) # Nur für lokale Ausführung
#CORS(app, resources={r"/predict": {"origins": "*"}})
#CORS(app, resources={r"/predict": {"origins": "*"}})







# Lade das Modell
model = ForecastModel()
model.train()


@app.route("/predict", methods=["POST"])
def predict() -> Tuple[Dict[str, Any], int]:
    """
    Predict endpoint for the Flask API.

    Expects a JSON payload with a 'features' key containing a list of lists.
    Returns a JSON response with the model predictions or an error message.

    Returns:
        Tuple[Dict[str, Any], int]: A JSON response and HTTP status code.
    """
    data = request.get_json()

    # Eingabevalidierung
    if not data or "features" not in data:
        logging.error("Invalid input: Missing 'features' key")
        return jsonify({"error": "Invalid input, 'features' key required"}), 400

    try:
        # Validierung der Features
        features = data["features"]
        if not isinstance(features, list) or not all(isinstance(f, list) and len(f) == 1 for f in features):
            logging.error(f"Invalid input format: {features}")
            return jsonify({"error": "Features must be a list of lists, each containing a single value."}), 400

        # Vorhersage
        prediction = model.predict(features)

        # Validierung des Outputs
        if not isinstance(prediction, list) or not all(isinstance(p, float) for p in prediction):
            logging.error(f"Unexpected model output: {prediction}")
            return jsonify({"error": "Unexpected model output. Please contact support."}), 500

        return jsonify({"prediction": prediction}), 200

    except ValueError as e:
        logging.error(f"Value error: {str(e)}")
        return jsonify({"error": f"Value error: {str(e)}"}), 400

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please contact support."}), 500


if __name__ == "__main__":
    #app.run(debug=True)
    app.run(host="0.0.0.0", port=5000, debug=False)