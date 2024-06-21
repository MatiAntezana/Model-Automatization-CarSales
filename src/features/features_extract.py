import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def transform_dolar_to_pesos(features, features_used):
    for index, value in features["Moneda"].items():
        if value == "U$S":
            features.at[index, "Precio"] = np.float64(features_used.value_dolar * features.at[index, "Precio"])
    return features

def transform_pesos_to_dolar(features, features_used):
    for index, value in features["Moneda"].items():
        if value == "$":
            features.at[index, "Precio"] = np.float64(features.at[index, "Precio"] / features_used.value_dolar)
    return features

def transform_price(features, features_used):
    if features_used.price_type == "Dolar":
        features = transform_pesos_to_dolar(features, features_used)
    else:
        features = transform_dolar_to_pesos(features, features_used)
    return features

def trasform_years_used(features, features_used):
    if features_used.transform_years == True:
        features["Años de uso"] = features["Año"].apply(lambda x: 2024 - x)
        features.drop(["Año"], axis=1, inplace=True)
        features = features[features["Años de uso"] >= 0]
        return features
    else: return features

def extract_kilometres(features):
    kilometres = features["Kilómetros"].str.split(" ", expand=True)
    features["Kilómetros"] = pd.to_numeric(kilometres[0], errors="coerce")
    return features

def transform_categ_features(features, features_used):
    features_one_hot = features_used.features_one_hot
    if len(features_one_hot) != 0: 
        features = pd.get_dummies(features, columns=features_one_hot, drop_first=True)
    return features

def delete_columns(features, features_used):
    columns_delete = features_used.features_delete
    if len(columns_delete) != 0:
        columns_features = set(features.columns)
        list_col_delete = list(set(columns_delete).intersection(columns_features))
        features.drop(list_col_delete, axis=1, inplace=True)
    return features

def delete_noise_doors(features):
    features = features[features["Puertas"] <= 6]
    return features

def standar_features(features, features_used):
    features_stadar = features_used.features_standar
    if len(features_stadar) != 0:
        scaler = StandardScaler()
        features[features_stadar] = scaler.fit_transform(features[features_stadar])
    return features

def transform_version(features):
    frecuencia_modelos = features.groupby(["Marca", "Modelo"]).size().reset_index(name="Frecuencia")
    total_marca = features["Marca"].value_counts().reset_index()
    total_marca.columns = ["Marca", "Total"]
    frecuencia_modelos = frecuencia_modelos.merge(total_marca, on="Marca")
    frecuencia_modelos["Probabilidad"] = frecuencia_modelos["Frecuencia"] / frecuencia_modelos["Total"]
    features.drop(columns=["Marca", "Modelo"], inplace=True)

def scalar_color(features, dic_probas):
    for index, value in features["Color"].items():
        features.loc[index, "Color"] = dic_probas[value]
    return features

def apply_transform(features, features_used):
    if features_used.relation_marca_version == True:
        transform_version(features)
    if features_used.transform_color == True:
        transform_features_color(features)
    
    return features

def transform_features_color(features):
    list_color = {"gris": ["gris", "gray"], "blanco": ["blanco", "blanca"], "negro": ["negro", "negra", "black"],
                  "plateado": ["plateado", "plata"], "azul": ["azul", "blue"], "rojo": ["rojo", "red"],
                  "marrón": ["marrón", "café"], "dorado": ["dorado"], "verde": ["verde"], "celeste": ["celeste"],
                  "naranja": ["naranja", "orange"], "amarillo": ["amarillo"], "violeta": ["violeta"],
                  "bordó": ["bordó"]}
    dic_count = {key: 0 for key in list_color}
    dic_count["otro"] = 0

    for index, value in features["Color"].items():
        if isinstance(value, str):
            word = value.lower()
            cond = False
            for key, sub_list_color in list_color.items():
                for color in sub_list_color:
                    if color in word:
                        features.loc[index, "Color"] = key
                        dic_count[key] += 1
                        cond = True
                        break
                if cond:
                    break
            if not cond:
                features.loc[index, "Color"] = "otro"
                dic_count["otro"] += 1
        else:
            features.loc[index, "Color"] = "otro"
            dic_count["otro"] += 1

    size_features = features.shape[0]
    dic_probas = {key: (value / size_features) for key, value in dic_count.items()}
    features = scalar_color(features, dic_probas)
    features.drop(columns=["Color"], inplace=True)

def scalar_kilometres(features):
    features["Kilómetros_invertidos"] = -features["Kilómetros"]
    scaler = MinMaxScaler()
    features["Kilómetros_escalados"] = scaler.fit_transform(features[["Kilómetros_invertidos"]])
    features.drop(columns=["Kilómetros_invertidos", "Kilómetros"], inplace=True)
    return features

def prob_model_for_marca(features):
    conteo_versiones_por_marca = features.groupby("Marca")["Modelo"].count().reset_index()
    conteo_versiones_por_marca.columns = ["Marca", "conteo_versiones"]
    df = features.merge(conteo_versiones_por_marca, on="Marca")
    df["probabilidad"] = 1 / df["conteo_versiones"]
    df.drop(columns=["conteo_versiones"], inplace=True)
    return df

def delete_duplicate(features):
    return features.drop_duplicates().reset_index(drop=True)

def delete_noise_years(features):
    return features[features["Puertas"] <= 2024]

def correct_marca(features):
    if "DS7" in features["Marca"].values:
        features.loc[features["Marca"] == "DS7", "Marca"] = "DS"

    if "DS AUTOMOBILES" in features["Marca"].values:
        features.loc[features["Marca"] == "DS AUTOMOBILES", "Marca"] = "DS"

    if "DS7" in features["Modelo"].values:
        features.loc[features["Modelo"] == "DS7", "Modelo"] = "7"

    if "Jetur" in features["Marca"].values:
        features.loc[features["Marca"] == "Jetur", "Marca"] = "Jetour"

    if "hiunday" in features["Marca"].values:
        features.loc[features["Marca"] == "hiunday", "Marca"] = "Hyundai"

    return features

def correct_dataset(features):
    features = delete_noise_doors(features)
    features = delete_noise_years(features)
    features = correct_marca(features)
    features = delete_duplicate(features)
    return features

def extract_features(path_dataset, features_used):
    # Obligatorios
    features = pd.read_csv(path_dataset)
    features = correct_dataset(features)

    features = transform_price(features, features_used)
    features = extract_kilometres(features)

    # Opcionales
    features = trasform_years_used(features, features_used)
    features = transform_categ_features(features, features_used)
    features = standar_features(features, features_used)

    features = apply_transform(features, features_used)

    # features = scalar_kilometres(features)

    # Obligatorio
    features = delete_columns(features, features_used)

    print(features)
    print(features.columns)
    return features
