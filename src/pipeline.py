import logging

from src.utils.control_result_paths import new_path_results, metadata_save
from src.features.features_extract import extract_features
from src.dataloaders.div_data import get_sets
from src.train_and_test.train import train_model
from src.train_and_test.test import test_model

from src.config_models.initial_model import load_model
from src.config_models.save_model import model_save_in_folder, model_save_result, model_save_best_params_of_params
from src.dataloaders.funcs_metadata import apply_normalization

def run_experiment(model_params, data_used, features_used):

    # Te crea la carpeta
    route_folder = new_path_results(model_params, data_used, features_used)
    logging.info("Se creo correctamente la carpeta nueva y el archivo con los parametros")

    # Extrae los features
    features = extract_features(data_used.path_dataset, features_used)
    logging.info("Se extrajo correctamente los features")

    # Divide los sets
    set_train, set_test = get_sets(features, data_used)
    logging.info("Se dividio correctamente los sets y aplicó a todos la función de trasformación de datos")

    set_train, set_test = apply_normalization(set_train, set_test, features_used)
    logging.info("Se aplicó correctamente la normalización (si se tenia que hacer) de los datos")

    model = load_model(model_params)
    logging.info("Se inicio el modelo correctamente")

    logging.info("Comienza el entrenamiento")
    train_model(model, set_train)
    logging.info("Se realizó correctamente el entrenamiento")

    print("El score llegado es:",model.best_score)
    print("Los mejores param son:",model.best_param)

    logging.info("Comienza el testeo")
    test_model(model, set_test)
    logging.info("Se realizó correctamente el testeo")

    model_save_result(model, route_folder)

    model_save_best_params_of_params(model, route_folder)

    model_save_in_folder(model, route_folder)
    logging.info("Se guardó correctamente el modelo")