from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

class Random_Forest:
    def __init__(self, n_estimators, random_state) -> None:
        self.func_model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        self.model = None
        self.MAE = None
        self.MSE = None
        self.RMSE = None
        self.r2 = None
        # self.accuracy = None

    def evaluate_model(self, X_test, Y_test):
        y_pred = self.model.predict(X_test)
        self.MAE = mean_absolute_error(Y_test, y_pred)
        self.MSE = mean_squared_error(Y_test, y_pred)
        self.r2 = r2_score(Y_test, y_pred)
        self.RMSE = np.sqrt(self.MSE)
        # self.accuracy = accuracy_score(Y_test, y_pred)
        # Ver porque acepta etiquetas de 0 y 1 nomas

    def train(self, X, Y):
        categorical_features = X.select_dtypes(include=['object']).columns
        numerical_features = X.select_dtypes(exclude=['object']).columns

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
        ])
        self.model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', self.func_model)])
        
        self.model.fit(X, Y)

def create_model(model_params):
    return Random_Forest(n_estimators=model_params.n_estimators, random_state=model_params.random_state)
