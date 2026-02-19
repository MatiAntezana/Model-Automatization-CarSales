import os

import joblib


def save_model_to_folder(model, results_folder):
    """Persist the trained model pipeline artifact into the results folder."""
    model_path = os.path.join(results_folder, "model.pkl")
    joblib.dump(model.model, model_path)
