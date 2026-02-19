from src.utils.module_loader import load_module_from_path


def load_model(model_config):
    """Create and return the model instance defined by the selected config module."""
    model_module = load_module_from_path(f"models/{model_config.model}")
    return model_module.create_model(model_config)
