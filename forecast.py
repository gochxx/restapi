import logging
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from typing import Optional, List
import numpy as np

# Logger-Konfiguration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # Ausgabe in die Konsole
        logging.FileHandler("forecast.log")  # Ausgabe in eine Datei
    ]
)

logger = logging.getLogger(__name__)  # Modul-spezifischer Logger


class ForecastModel:
    """
    A class to represent a simple forecasting model using Linear Regression.

    Attributes:
        model (LinearRegression): The underlying scikit-learn Linear Regression model.
        is_trained (bool): Indicates whether the model has been trained.
    """

    def __init__(self) -> None:
        """
        Initializes the ForecastModel with a LinearRegression model.
        """
        self.model: LinearRegression = LinearRegression()
        self.is_trained: bool = False
        logger.info("ForecastModel initialized.")

    def train(self, X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> None:

        """
        Trains the linear regression model with provided or default data.

        Args:
            X (Optional[np.ndarray]): The input features for training. Defaults to dummy data.
            y (Optional[np.ndarray]): The target values for training. Defaults to dummy data.

        if X is None and y is None:
            ValueError: If the shapes of X and y do not match.
        """

        if self.is_trained: # Check if the model is already trained
            logger.info("Model is already trained. Skipping re-training.")
            return

        if X is None or y is None: # Use default dummy data if no training data is provided
            logger.warning("No training data provided. Using default dummy data.")
            X = np.array([[i] for i in range(10)])  # Features
            y = np.array([2 * i + 1 for i in range(10)])  # Labels
        else: # Validate the input data
            if X.shape[0] != y.shape[0]: # Check if the number of samples in X and y match
                logger.error("The number of samples in X and y must match.")
                raise ValueError("The number of samples in X and y must match.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        logger.info("Starting model training...")
        self.model.fit(X_train, y_train) # Train the model 
        self.is_trained = True
        logger.info("Model training completed.")

        # Evaluate the model
        predictions = self.model.predict(X_test) # Make predictions
        mse = mean_squared_error(y_test, predictions) # Calculate Mean Squared Error
        logger.info(f"Model evaluation completed. Mean Squared Error: {mse:.4f}")

    def predict(self, features: List[List[float]]) -> List[float]:
        """
        Predicts target values for the given features.

        Args:
            features (List[List[float]]): A list of feature vectors for prediction.

        Returns:
            List[float]: A list of predicted values.

        Raises:
            ValueError: If the model is not trained or if the input format is invalid.
        """
        if not self.is_trained:
            logger.error("Prediction attempted on an untrained model.")
            raise ValueError("Model is not trained yet. Please train the model before making predictions.")

        # Validate and prepare input features
        features_array = np.array(features)
        if features_array.ndim != 2 or features_array.shape[1] != 1:
            logger.error(f"Invalid input format for prediction: {features}")
            raise ValueError("Input features must be a list of lists, each containing a single value.")

        # Make predictions
        logger.info(f"Making predictions for input: {features}")
        predictions = self.model.predict(features_array)

        # Validate predictions
        if not isinstance(predictions, np.ndarray) or not np.issubdtype(predictions.dtype, np.number):
            logger.error(f"Unexpected model output: {predictions}")
            raise ValueError(f"Unexpected model output: {predictions}")

        logger.info(f"Predictions completed successfully: {predictions.tolist()}")
        return predictions.tolist()