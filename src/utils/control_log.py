import os


def define_log_filename():
    """Return the next sequential log filename inside the local logs folder."""
    log_directory = "logs"
    os.makedirs(log_directory, exist_ok=True)

    existing_files = os.listdir(log_directory)
    matching_files = [name for name in existing_files if name.startswith("logger_") and name.endswith(".log")]

    if not matching_files:
        return "logger_1.log"

    file_numbers = [int(name.split("_")[1].split(".")[0]) for name in matching_files]
    return f"logger_{max(file_numbers) + 1}.log"
