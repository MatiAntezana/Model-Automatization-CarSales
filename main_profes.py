from src.features.features_extract import *
from src.utils.trasform_ruth_module import modularization_ruth
import joblib
import os
from src.features.features_extract import transform_inverse_scalar
import pandas as pd

def extract_features_test(path_dataset, features_used):
    features = pd.read_csv(path_dataset)
    tipo_vendedor_mode = features['Tipo de vendedor'].mode()[0]
    features['Tipo de vendedor'].fillna(tipo_vendedor_mode, inplace=True)

    tipo_transmisión_mode = features['Transmisión'].mode()[0]
    features['Transmisión'].fillna(tipo_transmisión_mode, inplace=True)

    # Obligatorios
    column_id = features["id"]
    features = extract_kilometres(features)

    features["Años de uso"] = features["Año"].apply(lambda x: max(2024 - x, 0))
    features.drop(["Año"], axis=1, inplace=True)
    
    features['Años de uso'].fillna(0, inplace=True)
    features['Kilómetros'].fillna(0, inplace=True)

    features["Marca Original"] = features["Marca"]

    # Opcionales
    features = transform_categ_features(features, features_used)
    features = standar_features(features, features_used)

    features = apply_transform(features, features_used)

    features = transform_labelencoder(features, features_used)
    features = transform_transm_proba(features, features_used)

    features = apply_method_hot_encoder(features, features_used)

    # Obligatorio
    features = features.drop("Marca Original", axis=1)

    return features, column_id

def load_best_model(path_model):
    route_model = os.path.join(path_model, "model.pkl")
    model = joblib.load(route_model)
    return model

def apply_normalization(set_test, features_used):
    if features_used.apply_normalization_kilometros == True:
        set_test["Kilómetros"] = transform_inverse_scalar(set_test["Kilómetros"])
        return set_test
    else: return set_test

def test_modelo(path_csv):
    features_used = modularization_ruth("best_model/features.py")
    features, column_id = extract_features_test(path_csv, features_used)
    features = apply_normalization(features, features_used)
    best_model = load_best_model("best_model")

    y_pred = best_model.predict(features)
    
    y_pred = pd.Series(y_pred, name='Predicted_Price_USD')

    results = pd.concat([column_id,y_pred],axis=1)

    file_path = "predictions_Antezana_Giacometti.csv"
    results.to_csv(file_path, index=False)


# Ingresar la ruta al csv

test_modelo("dataset/pf_suvs_test_ids_i302_1s2024.csv")