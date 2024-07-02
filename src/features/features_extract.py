import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

def transform_version(features, features_used):
    if features_used.relation_marca_version == True:
        conteo_versiones_por_marca = features.groupby("Marca")["Modelo"].count().reset_index()
        conteo_versiones_por_marca.columns = ["Marca", "conteo_versiones"]
        features = features.merge(conteo_versiones_por_marca, on="Marca")
        features["Prob Marca Modelo"] = 1 / features["conteo_versiones"]
        features.drop(columns=["conteo_versiones"], inplace=True)
        return features
    else: return features

# def transform_versioncasi(features, features_used):
#     if features_used.relation_marca_version == True:
#        # Calcula la frecuencia de cada modelo y la probabilidad de cada marca
#         frecuencia_modelos = features.groupby(["Marca", "Modelo"]).size().reset_index(name="Frecuencia")
#         total_marca = features["Marca"].value_counts().reset_index()
#         total_marca.columns = ["Marca", "Total"]
#         total_marca["Probabilidad_Marca"] = total_marca["Total"] / len(features)  # Probabilidad de cada marca

#         # Fusiona la frecuencia de modelos con las probabilidades de marca
#         frecuencia_modelos = frecuencia_modelos.merge(total_marca, on="Marca", how="left")
#         frecuencia_modelos["Probabilidad_Modelo_dado_Marca"] = frecuencia_modelos["Frecuencia"] / frecuencia_modelos["Total"]

#         # Fusiona las probabilidades calculadas de modelo dado marca con el conjunto de características original
#         features = features.merge(frecuencia_modelos[["Marca", "Modelo", "Probabilidad_Modelo_dado_Marca"]], on=["Marca", "Modelo"], how="left")

#         # Elimina las columnas Marca y Modelo después de calcular las probabilidades
#         features.drop(columns=["Marca", "Modelo"], inplace=True)
#         return features
#     else: return features

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
        # features["Transmisión"] = label_encoder(features, ["Transmisión"])
        features['Prob Transmisión'] = calcular_probabilidades(features['Transmisión'])
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
    # Ordenamos de mayor a menor las marcas con precios más caros promedio
    list_max_to_min_price = ['DS', 'Abarth', 'Jetour', 'Haval', 'Volkswagen', 'Citroën', 'Chevrolet', 
                         'Nissan', 'Peugeot', 'Fiat', 'Jeep', 'Renault', 'Toyota', 'Ford', 'BAIC', 
                         'Chery', 'Geely', 'Kia', 'JAC', 'Hyundai', 'Honda', 'Lifan', 'Dodge', 'Lexus', 
                         'Isuzu', 'Volvo', 'Ssangyong', 'Suzuki', 'Audi', 'BMW', 'Daihatsu', 'Mitsubishi', 
                         'Land Rover', 'Mercedes-Benz', 'Subaru', 'Porsche', 'Sandero', 'Alfa Romeo', 
                         'Jaguar', 'MINI']
    size_list = len(list_max_to_min_price) 
    order_list = {marca: (size_list - i) / size_list
                  for i, marca in enumerate(list_max_to_min_price)}

    def map_feature_to_valor(marca):
        return order_list.get(marca, 0)

    features['Marca'] = features['Marca'].map(map_feature_to_valor)

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
    list_max_to_min_price = ['bordó', 'azul', 'blanco', 'plateado', 'gris', 'rojo', 
                             'dorado', 'marrón', 'naranja', 'negro', 'amarillo', 
                             'celeste', 'violeta', 'verde', 'otro']  

    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Color'] = features['Color'].map(map_feature_to_valor)

    return features

def transform_combustible_for_price(features):
    list_max_to_min_price = ['Eléctrico', 'GNC', 'Híbrido/Nafta', 'Nafta', 'Diésel', 'Nafta/GNC', 'Híbrido', 'Híbrido/Diesel']

    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Tipo de combustible'] = features['Tipo de combustible'].map(map_feature_to_valor)

    return features

def transform_transmision_for_price(features):
    list_max_to_min_price = ['Semiautomática', 'Automática secuencial', 'Automática', 'Manual']
    
    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Transmisión'] = features['Transmisión'].map(map_feature_to_valor)

    return features

def transform_tipo_vendedor_for_price(features):
    list_max_to_min_price = ['concesionaria', 'tienda', 'particular']
    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Tipo de vendedor'] = features['Tipo de vendedor'].map(map_feature_to_valor)

    return features

def transform_puertas_for_price(features):
    list_max_to_min_price = [6.0, 4.0, 5.0, 2.0, 3.0]
    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Puertas'] = features['Puertas'].map(map_feature_to_valor)

    return features

def transform_years_to_price(features):
    list_max_to_min_price = [0.0, 1.0, 2.0, 3.0, 5.0, 4.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 17.0, 16.0, 13.0, 20.0, 15.0, 14.0, 21.0, 19.0, 24.0, 18.0, 23.0, 31.0, 28.0, 30.0, 29.0, 27.0, 22.0, 26.0, 25.0, 32.0, 37.0, 33.0]
    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}
    
    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Años de uso'] = features['Años de uso'].map(map_feature_to_valor)

    return features

def transform_marca_to_price(features):
    list_max_to_min_price = ['DS', 'Abarth', 'Jetour', 'Haval', 'Volkswagen', 'Citroën', 'Chevrolet', 'Nissan', 'Peugeot', 'Fiat', 'Jeep', 'Renault', 'Toyota', 'Ford', 'BAIC', 'Chery', 'Geely', 'Kia', 'JAC', 'Hyundai', 'Honda', 'Lifan', 'Dodge', 'Lexus', 'Isuzu', 'Volvo', 'Ssangyong', 'Suzuki', 'Audi', 'BMW', 'Daihatsu', 'Mitsubishi', 'Land Rover', 'Mercedes-Benz', 'Subaru', 'Porsche', 'Sandero', 'Alfa Romeo', 'Jaguar', 'MINI']

    size_list = len(list_max_to_min_price) 
    order_list = {cat: (size_list - i) / size_list
                  for i, cat in enumerate(list_max_to_min_price)}

    def map_feature_to_valor(cat):
        return order_list.get(cat, 0)
    
    features['Marca'] = features['Marca'].map(map_feature_to_valor)

    return features

def apply_probs_price(features, features_used):
    # Transforma kilometros, Marca_Modelo, combustible, 
    # transmisión, tipo de vendedor, puertas, años de uso
    if features_used.apply_prob_price == True:
        features['Kilómetros'] = transform_inverse_scalar(features["Kilómetros"])
        # features = transform_marca_for_price(features)
        # features = transform_features_color_2(features) # Antes de transformar en número
        # features = transform_color_for_price(features)
        features = transform_marca_to_price(features)
        features = transform_years_to_price(features)
        features = transform_combustible_for_price(features)
        features = transform_transmision_for_price(features)
        features = transform_tipo_vendedor_for_price(features)
        features = transform_puertas_for_price(features)
        return features
    else: return features

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

    features = transform_labelencoder(features, features_used)
    features = transform_transm_proba(features, features_used)

    features = apply_probs_price(features, features_used)
    # features = scalar_kilometres(features)

    # Obligatorio
    features = delete_columns(features, features_used)

    print(features)
    print(features.columns)
    return features
