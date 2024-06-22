from src.utils.trasform_ruth_module import modularization_ruth
from sklearn.model_selection import GridSearchCV    
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

class Initial_Model:
    def __init__(self, model_params, file_model) -> None:
        self.prev_model = GridSearchCV(estimator=file_model.create_model(), 
                                 param_grid=model_params.params, 
                                 scoring="neg_root_mean_squared_error", 
                                 verbose=0, 
                                 cv=model_params.cv, 
                                 n_jobs=-1) # Cant de procesadores a utilizar
        self.model = None
        self.MAE = None
        self.MSE = None
        self.RMSE = None
        self.r2 = None
        self.best_param = None
        self.best_score = None

    def evaluate_model(self, x_test, y_test):
        y_pred = self.model.predict(x_test)
        self.MAE = mean_absolute_error(y_test, y_pred)
        self.MSE = mean_squared_error(y_test, y_pred)
        self.r2 = r2_score(y_test, y_pred)
        self.RMSE = np.sqrt(self.MSE)

    def train(self, x, y):
        self.model  = self.prev_model.fit(x,y)
        self.best_param = self.model.best_params_

def load_model(model_params):
    file_model = modularization_ruth(f"src/models/{model_params.model}")
    class_model = Initial_Model(model_params, file_model)
    return class_model

