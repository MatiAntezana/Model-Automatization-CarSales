import logging

from src.utils.control_result_paths import new_path_results, metadata_save
from src.dataloaders.funcs_metadata import load_metadata
from src.features.features_extract import extract_features
from src.dataloaders.div_data import get_dataloader
from src.models.control_model import load_model

def run_experiment(model_params, data_used, features_used):

    # Te crea la carpeta
    name_path = new_path_results(model_params, data_used, features_used)
    logging.info("Se creo correctamente la carpeta nueva y el archivo con los parametros")

    # Extrae los features
    features = extract_features(data_used.path_dataset, features_used.features_use)
    logging.info("Se extrajo correctamente los features")

    # Divide los sets
    set_train, final_index = get_dataloader(features, data_used.param_sets, features_used, "Set_Train", 0)
    set_valid, final_index = get_dataloader(features, data_used.param_sets, features_used,"Set_Valid", final_index)
    set_test,_ = get_dataloader(features, data_used.param_sets, features_used, "Set_Test", final_index)
    logging.info("Se dividio correctamente los sets y aplicó a todos la función de trasformación de datos")

    # model = load_model(model_params)
    # logging.info("Se cargo el archivo del modelo correctamente")





# # Cargar los datos desde el archivo CSV
#     data = pd.read_csv(data)

# # Dividir los datos en características (X) y variable objetivo (y)
#     X = data.drop(['Precio', 'Moneda'], axis=1)  # Características
#     y = data['Precio']  # Precio

# # Preprocesar los datos (por ejemplo, codificar variables categóricas)
#     X = pd.get_dummies(X)

# # Dividir los datos en conjuntos de entrenamiento y prueba
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Inicializar y entrenar el modelo de regresión lineal
#     model = LinearRegression()
#     model.fit(X_train, y_train)

# # Predecir los precios en el conjunto de prueba
#     y_pred = model.predict(X_test)

# # Evaluar el rendimiento del modelo
#     mse = mean_squared_error(y_test, y_pred)
#     print('Error cuadrático medio:', mse)
#     joblib.dump(model, 'models/model.pkl')
