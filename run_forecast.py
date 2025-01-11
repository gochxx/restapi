from forecast import ForecastModel
import numpy as np

"""
Dieses Script dient dazu, den Forecast im Scripts "forecast.py" manuell zu testen. 
"""

# Instanziiere das Modell
model = ForecastModel()

# Beispiel-Daten vorbereiten
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Trainiere das Modell
print("Training startet...")
model.train(X, y)
print("Training abgeschlossen.")

# Vorhersage testen
print("Vorhersagen werden durchgeführt...")
features = [[6], [7], [8]]
predictions = model.predict(features)

print(f"Features: {features}")
print(f"Vorhersagen: {predictions}")