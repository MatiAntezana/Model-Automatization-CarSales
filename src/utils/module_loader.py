import importlib.util


def load_module_from_path(module_path):
    """Load a Python module from a file path and return the imported module object."""
    spec = importlib.util.spec_from_file_location("loaded_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from path: {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
