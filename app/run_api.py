import requests
import json


"""
Dieses Script dient dazu, die API manuell zu testen. 
Um es auszuführen muss die Flask App in "app.py" gestartet werden. Dieses Script hier muss dann in einem separaten Terminal laufen.
"""

# API-Endpunkt (Basis-URL der Flask-App)
BASE_URL = "http://127.0.0.1:5000"

# Beispiel-Daten für die Anfrage
data = {
    "features": [[5], [10], [15]]  # Eingabedaten für die Vorhersage
}

# POST-Anfrage senden
try:
    response = requests.post(f"{BASE_URL}/predict", json=data)
    response.raise_for_status()  # Überprüft, ob ein HTTP-Fehler aufgetreten ist

    # Ausgabe der Antwort
    print("Antwort vom Server:")
    print(json.dumps(response.json(), indent=4))  # Schön formatierte Ausgabe der JSON-Antwort
except requests.exceptions.RequestException as e:
    print(f"Fehler bei der Anfrage: {e}")