import pytest
from forecast import ForecastModel
from flask.testing import FlaskClient
from flask import Flask


@pytest.fixture
def trained_model() -> ForecastModel:
    """
    Fixture, das ein trainiertes ForecastModel bereitstellt.

    Returns:
        ForecastModel: Ein trainiertes Instanzobjekt des ForecastModel.
    """
    model = ForecastModel()
    model.train()
    return model


def test_model_training(trained_model: ForecastModel) -> None:
    """
    Testet, ob das Modell nach dem Training korrekt als trainiert markiert ist.

    Args:
        trained_model (ForecastModel): Ein trainiertes Modell.
    """
    assert trained_model.is_trained is True, "Model should be trained after calling train()"


def test_model_prediction_shape(trained_model: ForecastModel) -> None:
    """
    Testet, ob die Vorhersagen die gleiche Länge wie die Eingabe haben.

    Args:
        trained_model (ForecastModel): Ein trainiertes Modell.
    """
    features = [[5], [10], [15]]
    predictions = trained_model.predict(features)
    assert len(predictions) == len(features), "Predictions should match the number of inputs"


def test_model_prediction_values(trained_model: ForecastModel) -> None:
    """
    Testet, ob die Vorhersagen eine Liste von Floats sind.

    Args:
        trained_model (ForecastModel): Ein trainiertes Modell.
    """
    features = [[5], [10]]
    predictions = trained_model.predict(features)
    assert isinstance(predictions, list), "Predictions should be a list"
    assert all(isinstance(x, float) for x in predictions), "All predictions should be floats"


def test_untrained_model_prediction() -> None:
    """
    Testet, ob ein Fehler auftritt, wenn ein untrainiertes Modell Vorhersagen ausführen soll.
    """
    model = ForecastModel()
    with pytest.raises(ValueError, match="Model is not trained yet."):
        model.predict([[5]])


@pytest.fixture
def client() -> FlaskClient:
    """
    Fixture, das einen Test-Client für die Flask-API bereitstellt.

    Returns:
        FlaskClient: Ein Test-Client der Flask-Anwendung.
    """
    from app import app
    with app.test_client() as client:
        yield client


def test_api_valid_input(client: FlaskClient) -> None:
    """
    Testet die API mit gültigen Eingabedaten.

    Args:
        client (FlaskClient): Ein Test-Client der Flask-Anwendung.
    """
    response = client.post("/predict", json={"features": [[5], [10]]})
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)


def test_api_invalid_input(client: FlaskClient) -> None:
    """
    Testet die API mit ungültigen Eingabedaten.

    Args:
        client (FlaskClient): Ein Test-Client der Flask-Anwendung.
    """
    response = client.post("/predict", json={"wrong_key": [[5], [10]]})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_api_invalid_input_values(client: FlaskClient) -> None:
    """
    Testet die API mit ungültigen Eingabedaten, hier nicht Key sondern Werte der JSON.

    Args:
        client (FlaskClient): Ein Test-Client der Flask-Anwendung.
    """
    response = client.post("/predict", json={"features": [[5], [10,20]]})
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data

def test_api_unexpected_error(client: FlaskClient) -> None:
    """
    Testet die API, wenn das Modell nicht verfügbar ist.

    Args:
        client (FlaskClient): Ein Test-Client der Flask-Anwendung.
    """
    from app import app
    app.view_functions["predict"].__globals__["model"] = None  # Modell temporär entfernen
    response = client.post("/predict", json={"features": [[5], [10]]})
    assert response.status_code == 500
    data = response.get_json()
    assert "error" in data