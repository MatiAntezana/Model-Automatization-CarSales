import logging

from src.utils.control_result_paths import new_path_results, metadata_save
from src.dataloaders.funcs_metadata import load_metadata
from src.features.features_extract import extract_features
from src.dataloaders.div_data import get_dataloader
from src.train_and_test.train import train_model
from src.train_and_test.test import test_model

from src.config_models.initial_model import load_model
from src.config_models.save_model import model_save_in_folder, model_save_result, model_save_best_params_of_params

def run_experiment(model_params, data_used, features_used):

    # Te crea la carpeta
    route_folder = new_path_results(model_params, data_used, features_used)
    logging.info("Se creo correctamente la carpeta nueva y el archivo con los parametros")

    # Extrae los features
    features = extract_features(data_used.path_dataset, features_used)
    logging.info("Se extrajo correctamente los features")

    # Divide los sets
    set_train, final_index = get_dataloader(features, data_used, features_used, "Set_Train", 0)
    set_valid, final_index = get_dataloader(features, data_used, features_used,"Set_Valid", final_index)
    set_test,_ = get_dataloader(features, data_used, features_used, "Set_Test", final_index)
    logging.info("Se dividio correctamente los sets y aplicó a todos la función de trasformación de datos")

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