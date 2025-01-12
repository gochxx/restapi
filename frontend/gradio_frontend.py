import gradio as gr
import requests

# Funktion, um die Flask-API aufzurufen
def get_predictions(input_values):
    try:
        # Eingabe verarbeiten: Komma-separierte Werte in ein Array umwandeln
        values = [float(val.strip()) for val in input_values.split(",") if val.strip()]
        data = {
            "features": [[v] for v in values]  # Datenformat für die API
        }

        # API-Aufruf
        response = requests.post("http://127.0.0.1:5000/predict", json=data)
        response.raise_for_status()  # Fehler prüfen

        # API-Antwort verarbeiten
        predictions = response.json().get("prediction", [])
        return "\n".join([str(p) for p in predictions])
    except Exception as e:
        return f"Fehler: {str(e)}"

# Gradio-Interface erstellen
with gr.Blocks() as demo:
    gr.Markdown("# Prediction App mit Gradio")
    gr.Markdown("Geben Sie Werte ein, getrennt durch Kommas, z. B.: `5,10,15`")
    
    # Eingabefeld
    input_text = gr.Textbox(label="Eingabewerte (komma-separiert)")
    
    # Ausgabe
    output_text = gr.Textbox(label="Vorhersagen")

    # Button und Logik
    predict_button = gr.Button("Predict")
    predict_button.click(get_predictions, inputs=input_text, outputs=output_text)

# App starten
if __name__ == "__main__":
    demo.launch()
