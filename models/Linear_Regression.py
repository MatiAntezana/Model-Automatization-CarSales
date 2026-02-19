import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class LinearRegressionModel:
    """Linear Regression wrapper with preprocessing and evaluation metrics."""

    def __init__(self) -> None:
        """Initialize the linear regression estimator and metric placeholders."""
        self.func_model = LinearRegression()
        self.model = None
        self.mae = None
        self.mse = None
        self.rmse = None
        self.r2 = None

    def evaluate_model(self, x_test, y_test):
        """Run predictions on test data and update stored regression metrics."""
        y_pred = self.model.predict(x_test)
        self.mae = mean_absolute_error(y_test, y_pred)
        self.mse = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        self.rmse = np.sqrt(self.mse)

    def train(self, features, target):
        """Fit preprocessing and regression pipeline on training data."""
        categorical_features = features.select_dtypes(include=["object"]).columns
        numerical_features = features.select_dtypes(exclude=["object"]).columns

        numeric_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
        )
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numerical_features),
                ("cat", categorical_transformer, categorical_features),
            ]
        )

        self.model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", self.func_model)])
        self.model.fit(features, target)


def create_model(model_config):
    """Factory function used by the pipeline to instantiate the linear regression model."""
    _ = model_config
    return LinearRegressionModel()
