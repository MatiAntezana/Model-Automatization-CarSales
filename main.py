import argparse
from src.pipeline import run_experiment
import logging
import os
from src.utils.control_log import define_n_log
from src.utils.trasform_ruth_module import modularization_ruth

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.join("..", "Model-Automatization-CarSales", "logs"), define_n_log())),
    ]
)
logging.info("Se guardo correctamente el log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Ejecutar experimentos de machine learning')
    parser.add_argument('--model', type=str, required=True, help='Ruta al archivo de configuración del modelo')
    parser.add_argument('--data', type=str, required=True, help='Ruta al archivo de configuración de los datos')
    parser.add_argument('--features', type=str, required=True, help='Ruta al archivo de configuración de las características')
    args = parser.parse_args()
    model_used = modularization_ruth(args.model)
    data_used = modularization_ruth(args.data)
    features_used = modularization_ruth(args.features)

    run_experiment(model_used, data_used, features_used)
