import pandas as pd


def read_data(file_path):
    """Read a CSV file and return its values as a NumPy array."""
    dataframe = pd.read_csv(file_path)
    return dataframe.values


def count_car_types(data):
    """Count occurrences of each car type label in the first column of a dataset."""
    counts_by_type = {}
    for row in data:
        label = row[0]
        counts_by_type[label] = counts_by_type.get(label, 0) + 1
    return counts_by_type
