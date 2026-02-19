import argparse
import logging
import os

from src.pipeline import run_experiment
from src.utils.control_log import define_log_filename
from src.utils.module_loader import load_module_from_path


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join("logs", define_log_filename()))],
)
logging.info("Log file configured successfully.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment run."""
    parser = argparse.ArgumentParser(description="Run machine learning experiments")
    parser.add_argument("--model", type=str, required=True, help="Path to the model config file")
    parser.add_argument("--data", type=str, required=True, help="Path to the data config file")
    parser.add_argument("--features", type=str, required=True, help="Path to the feature config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model_config = load_module_from_path(args.model)
    data_config = load_module_from_path(args.data)
    feature_config = load_module_from_path(args.features)
    run_experiment(model_config, data_config, feature_config)
