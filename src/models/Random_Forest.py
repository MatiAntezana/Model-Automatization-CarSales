from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor

class Random_Forest:
    def __init__(self, n_estimators, random_state) -> None:
        self.model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.MAE = None
        self.MSE = None
        self.RMSE = None
        self.r2 = None
        # self.accuracy = None

    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        self.MAE = mean_absolute_error(y_test, y_pred)
        self.MSE = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        self.RMSE = np.sqrt(self.MSE)

    def train(self, x, y):
        self.model.fit(x,y)

def create_model(model_params):
    return Random_Forest(n_estimators=model_params.n_estimators, random_state=model_params.random_state)
