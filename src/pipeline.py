import logging

from src.dataloaders.split_data import get_data_split
from src.features.features_extract import extract_features
from src.models.initial_model import load_model
from src.models.save_model import save_model_to_folder
from src.train_and_test.test import test_model
from src.train_and_test.train import train_model
from src.utils.control_result_paths import create_results_folder


def run_experiment(model_config, data_config, feature_config) -> None:
    """Run the complete training and evaluation pipeline for one experiment."""
    results_folder = create_results_folder(model_config, data_config, feature_config)
    logging.info("Results folder and configuration summary created successfully.")

    features = extract_features(data_config.dataset_path, feature_config.selected_features)
    logging.info("Feature extraction completed successfully.")

    train_set, final_index = get_data_split(features, data_config, feature_config, "train", 0)
    validation_set, final_index = get_data_split(features, data_config, feature_config, "valid", final_index)
    test_set, _ = get_data_split(features, data_config, feature_config, "test", final_index)
    logging.info("Dataset splitting and transformations completed successfully.")

    if not validation_set.empty:
        logging.info("Validation split generated with %d rows.", len(validation_set))

    model = load_model(model_config)
    logging.info("Model instance created successfully.")

    logging.info("Starting training stage.")
    train_model(model, train_set)
    logging.info("Training stage completed successfully.")

    logging.info("Starting testing stage.")
    test_model(model, test_set)
    logging.info("Testing stage completed successfully.")

    save_model_to_folder(model, results_folder)
    logging.info("Model artifact saved successfully.")
