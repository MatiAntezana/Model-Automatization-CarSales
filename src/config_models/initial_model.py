import os
import importlib
from src.utils.trasform_ruth_module import modularization_ruth

def load_model(model_params):
    file_model = modularization_ruth(f"src/models/{model_params.model}")
    model = file_model.create_model(model_params)
    return model