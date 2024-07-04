import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

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

def trasform_years_used(features):
    features["Años de uso"] = features["Año"].apply(lambda x: 2024 - x)
    features.drop(["Año"], axis=1, inplace=True)
    features = features[features["Años de uso"] >= 0]
    return features

def convert_km_to_numeric(km_str):
    if km_str == "0 km":
        return 0
    if pd.isna(km_str):
        return np.nan
    km_str = str(km_str)
    km_value = int("".join(filter(str.isdigit, km_str)))
    return km_value

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

def transform_version(features, features_used):
    if features_used.relation_marca_version == True:
        conteo_versiones_por_marca = features.groupby("Marca")["Modelo"].count().reset_index()
        conteo_versiones_por_marca.columns = ["Marca", "conteo_versiones"]
        features = features.merge(conteo_versiones_por_marca, on="Marca")
        features["Prob Marca Modelo"] = 1 / features["conteo_versiones"]
        features.drop(columns=["conteo_versiones"], inplace=True)
        return features
    else: return features

def scalar_color(features, dic_probas):
    for index, value in features["Color"].items():
        features.loc[index, "Color"] = dic_probas[value]
    return features

def transform_features_color(features, features_used):
    if features_used.transform_color == True:

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
                if cond == False:
                    features.loc[index, "Color"] = "otro"
                    dic_count["otro"] += 1
            else:
                features.loc[index, "Color"] = "otro"
                dic_count["otro"] += 1

        size_features = features.shape[0]
        dic_probas = {key: (value / size_features) for key, value in dic_count.items()}
        features = scalar_color(features, dic_probas)
        return features
    else: return features

def apply_transform(features, features_used):
    features = transform_version(features, features_used)
    features = transform_features_color(features, features_used)
    
    return features

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
    features["probabilidad"] = 1 / features["conteo_versiones"]
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

    if 436707 in features["Año"].values: # Estaba en el título
        features.loc[features["Año"] == 436707, "Año"] = 2012

    return features

def correct_dataset(features):
    features = delete_noise_doors(features)
    features = delete_noise_years(features)
    features = correct_marca(features)
    features = delete_duplicate(features)
    return features

def label_encoder(features, list_features):
    for feature in list_features:
        le = LabelEncoder()
        features[feature] = le.fit_transform(features[feature])
    return features

def transform_labelencoder(features, features_used):
    features_label_encoder = features_used.features_label_encoder
    if len(features_label_encoder) != 0:
        features = label_encoder(features, features_label_encoder)
        return features
    else: return features

def calcular_probabilidades(column):
    frec = column.value_counts(normalize=True)
    
    probs = column.map(frec)
    
    most_comun = frec.idxmax()
    
    probs.fillna(most_comun, inplace=True)
    
    return probs

def transform_transm_proba(features, features_used):
    if features_used.transform_proba == True:
        features["Prob Transmisión"] = calcular_probabilidades(features["Transmisión"])
        features["Prob Año de uso"] = calcular_probabilidades(features["Años de uso"])
        features["Prob Tipo de combustible"] = calcular_probabilidades(features["Tipo de combustible"])
        features["Prob Color"] = calcular_probabilidades(features["Color"])
        features["Prob Tipo de vendedor"] = calcular_probabilidades(features["Tipo de vendedor"])
        features["Prob Puertas"] = calcular_probabilidades(features["Puertas"])

        features.drop(["Transmisión", "Años de uso", "Tipo de combustible", "Color", "Tipo de vendedor", "Puertas"], axis=1, inplace=True)
    return features

def transform_inverse_scalar(column):
    min_value = column.min()
    return 1 - (column - min_value) / (column.max() - min_value)

def transform_marca_for_price(features):
    list_max_to_min_price = ["DS", "Abarth", "Jetour", "Haval", "Volkswagen", "Citroën", "Chevrolet", 
                         "Nissan", "Peugeot", "Fiat", "Jeep", "Renault", "Toyota", "Ford", "BAIC", 
                         "Chery", "Geely", "Kia", "JAC", "Hyundai", "Honda", "Lifan", "Dodge", "Lexus", 
                         "Isuzu", "Volvo", "Ssangyong", "Suzuki", "Audi", "BMW", "Daihatsu", "Mitsubishi", 
                         "Land Rover", "Mercedes-Benz", "Subaru", "Porsche", "Sandero", "Alfa Romeo", 
                         "Jaguar", "MINI"]
    size_list = len(list_max_to_min_price) 
    order_list = {marca: (size_list - i) / size_list
                  for i, marca in enumerate(list_max_to_min_price)}

    def map_feature_to_valor(marca):
        return order_list.get(marca, 0)

    features["Marca"] = features["Marca"].map(map_feature_to_valor)

    return features

def transform_features_color_2(features):
    list_color = {"gris": ["gris", "gray"], "blanco": ["blanco", "blanca"], "negro": ["negro", "negra", "black"],
                  "plateado": ["plateado", "plata"], "azul": ["azul", "blue"], "rojo": ["rojo", "red"],
                  "marrón": ["marrón", "café"], "dorado": ["dorado"], "verde": ["verde"], "celeste": ["celeste"],
                  "naranja": ["naranja", "orange"], "amarillo": ["amarillo"], "violeta": ["violeta"],
                  "bordó": ["bordó"]}

    for index, value in features["Color"].items():
        if isinstance(value, str):
            word = value.lower()
            cond = False
            for key, sub_list_color in list_color.items():
                for color in sub_list_color:
                    if color in word:
                        features.loc[index, "Color"] = key
                        cond = True
                        break
                if cond:
                    break
            if cond == False:
                features.loc[index, "Color"] = "otro"
        else:
            features.loc[index, "Color"] = "otro"
                        
    return features

def transform_color_for_price(features):
    list_max_to_min_price = ["bordó", "azul", "blanco", "plateado", "gris", "rojo", 
                             "dorado", "marrón", "naranja", "negro", "amarillo", 
                             "celeste", "violeta", "verde", "otro"]  

    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features["Color"] = features["Color"].map(map_feature_to_valor)

    return features

def hapus_outliers(data, x, factor): 
    # Eliminamos outliers usando el rango IQR
    Q1 = data[x].quantile(0.25) 
    Q3 = data[x].quantile(0.75) 
    IQR = Q3 - Q1 
    data = data[~((data[x] < (Q1 - factor * IQR)) | (data[x] > (Q3 + factor * IQR)))] 
    return data

def marca_price_for_mercado(features):
    marcas = [
    "Porsche", "Jaguar", "Land Rover", "Mercedes-Benz", "BMW", "Audi", "Lexus", 
    "Volvo", "MINI", "Jeep", "DS", "Alfa Romeo", "Subaru", "Mitsubishi", "Honda", 
    "Toyota", "Hyundai", "Kia", "Nissan", "Volkswagen", "Ford", "Chevrolet", 
    "Renault", "Peugeot", "Citroën", "Fiat", "Suzuki", "Chery", "Geely", "JAC", 
    "BAIC", "Haval", "Jetour", "Lifan", "Dodge", "Isuzu", "Daihatsu", "Ssangyong"
    ]
    size_list = len(marcas) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(marcas)}

    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)

    features["Marca"] = features["Marca"].map(map_feature_to_valor)
    return features


def tipo_combustible_price_for_mercado(features):
    tipos_combustible = [
    "Eléctrico", "Híbrido/Diesel", "Híbrido", "Híbrido/Nafta", 
    "Nafta", "Diésel", "GNC", "Nafta/GNC"
    ]
    size_list = len(tipos_combustible) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(tipos_combustible)}

    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)

    features["Tipo de combustible"] = features["Tipo de combustible"].map(map_feature_to_valor)
    return features

def apply_method_hot_encoder(features, features_used):
    if features_used.apply_method_hot_encoder == True:
        features = marca_price_for_mercado(features)
        features = tipo_combustible_price_for_mercado(features)
        
        new_features = pd.concat([
                features["Marca"],
                features["Tipo de combustible"],
                features["Transmisión"],
                features["Tipo de vendedor"],
                features["Años de uso"],
                features["Kilómetros"],
                features["Marca Original"]
                ], axis=1)
        
        cat_features = ["Transmisión", "Tipo de vendedor"]

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan)
        encoder.fit(new_features[cat_features])
        new_features[cat_features] = pd.DataFrame(encoder.transform(new_features[cat_features]), columns=cat_features)
        
        imputer = SimpleImputer(strategy="most_frequent")
        new_features[cat_features] = imputer.fit_transform(new_features[cat_features])

        return new_features
    else: return features

def extract_features(path_dataset, features_used):
    # Obligatorios
    features = pd.read_csv(path_dataset)
    features = correct_dataset(features)

    features = transform_price(features, features_used)
    features = extract_kilometres(features)

    features = hapus_outliers(features, "Precio", 3)

    features = hapus_outliers(features, "Kilómetros", 5)

    features = trasform_years_used(features)

    features["Marca Original"] = features["Marca"]
    column_precio = features["Precio"]

    # Opcionales

    features = transform_categ_features(features, features_used)
    features = standar_features(features, features_used)

    features = apply_transform(features, features_used)

    features = transform_labelencoder(features, features_used)
    features = transform_transm_proba(features, features_used)

    features = apply_method_hot_encoder(features, features_used)

    # Obligatorio
    features = pd.concat([features, column_precio], axis=1)
    features = delete_columns(features, features_used)

    features = features.dropna(subset=["Kilómetros"])

    return features
