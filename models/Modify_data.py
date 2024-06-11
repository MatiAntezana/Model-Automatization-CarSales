import numpy as np

def transform_dolar_to_pesos(features_used, features):
    for index, value in features["Moneda"].items():
        if value == "U$S":
            features.at[index, "Precio"] = np.float64(features_used.value_dolar * features.at[index, "Precio"])
            features.at[index, "Moneda"] = "$"

    return features

def data_modify(features_used, features):
    features = transform_dolar_to_pesos(features_used, features)
