import pandas as pd
import importlib
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

def transform_dolar_to_pesos(features, features_used):
    for index, value in features["Moneda"].items():
        if value == "U$S":
            features.at[index, "Precio"] = np.float64(features_used.value_dolar * features.at[index, "Precio"])

def trasform_years_used(features):
    features["Años de uso"]=features["Año"].apply(lambda x:2024-x)
    features.drop(["Año"],axis=1,inplace=True)
    features[features["Años de uso"] >= 0]

def extract_kilometres(features):
    kilometres = features["Kilómetros"].str.split(" ",expand=True)
    features["Kilómetros"] = pd.to_numeric(kilometres[0],errors="coerce")

def transform_categ_features(features, features_used):
    features_one_hot = features_used.features_one_hot
    if len(features_one_hot) != 0: 
        return pd.get_dummies(features,columns=features_one_hot,drop_first=True)
    else:
        return features

def delete_columns(features, features_used):
    columns_delete = features_used.features_delete
    if len(columns_delete) != 0:
        features.drop(columns_delete,axis=1,inplace=True)

def transform_pesos_to_dolar(features, features_used):
    for index, value in features["Moneda"].items():
        if value == "$":
            features.at[index, "Precio"] = np.float64(features.at[index, "Precio"] / features_used.value_dolar)

def delete_noise_doors(features):
     features[features["Puertas"] <= 6]

def standar_features(features, features_used):
    features_stadar = features_used.features_standar

    if len(features_stadar) != 0:
        scaler = StandardScaler()
        features[features_stadar] = scaler.fit_transform(features[features_stadar])


def prob_model_for_marca(features):
    conteo_versiones_por_marca = features.groupby("Marca")["Modelo"].count().reset_index()
    conteo_versiones_por_marca.columns = ["Marca", "conteo_versiones"]

    df = features.merge(conteo_versiones_por_marca, on="Marca")

    df["probabilidad"] = 1 / df["conteo_versiones"]

    df = df.drop(columns=["conteo_versiones"])

    return df

def transform_price(features, features_used):
    if features_used.price_type == "Dolar":
        transform_pesos_to_dolar(features, features_used)
    else:
        transform_dolar_to_pesos(features, features_used)

def extract_features(path_dataset, features_used):
    features = pd.read_csv(path_dataset)

    transform_price(features, features_used)

    trasform_years_used(features)

    extract_kilometres(features)

    features = transform_categ_features(features, features_used)

    # features = prob_model_for_marca(features)

    delete_noise_doors(features)

    delete_columns(features, features_used)

    print(features)
    return features