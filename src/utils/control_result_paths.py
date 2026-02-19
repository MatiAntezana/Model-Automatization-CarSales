import os
import re


def create_results_folder(model_config, data_config, feature_config):
    """Create a numbered results directory and store the run configuration summary."""
    target_dir = "results"
    os.makedirs(target_dir, exist_ok=True)

    pattern = re.compile(r"model_(\d+)_folder$")
    existing_numbers = []

    for folder_name in os.listdir(target_dir):
        match = pattern.match(folder_name)
        if match:
            existing_numbers.append(int(match.group(1)))

    new_number = max(existing_numbers, default=0) + 1
    new_folder_name = f"model_{new_number}_folder"
    new_folder_path = os.path.join(target_dir, new_folder_name)
    os.makedirs(new_folder_path)

    summary_lines = [
        "Configuration files used for this run",
        f"Model config: {getattr(model_config, '__file__', str(model_config))}",
        f"Data config: {getattr(data_config, '__file__', str(data_config))}",
        f"Feature config: {getattr(feature_config, '__file__', str(feature_config))}",
    ]

    summary_file = os.path.join(new_folder_path, f"params_used_{new_number}.txt")
    with open(summary_file, "w", encoding="utf-8") as summary_handle:
        summary_handle.write("\n".join(summary_lines) + "\n")

    return new_folder_path


def metadata_save(metadata):
    """Placeholder metadata persistence function kept for compatibility."""
    _ = metadata
    return "metadata_not_implemented"
