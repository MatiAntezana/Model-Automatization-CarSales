import os
import importlib

def load_model(model_params):
    target_dir = os.path.join("..", "Model-Automatization-CarSales", f"models/{model_params.model}")
    spec = importlib.util.spec_from_file_location("models", target_dir)
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    return model