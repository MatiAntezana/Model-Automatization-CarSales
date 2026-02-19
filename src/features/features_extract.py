import pandas as pd

ENGLISH_TO_SOURCE_COLUMN = {
    "Brand": "Marca",
    "Model": "Modelo",
    "Year": "Año",
    "Version": "Versión",
    "Color": "Color",
    "Fuel Type": "Tipo de combustible",
    "Doors": "Puertas",
    "Transmission": "Transmisión",
    "Engine": "Motor",
    "Body Type": "Tipo de carrocería",
    "Mileage": "Kilómetros",
    "Title": "Título",
    "Price": "Precio",
    "Currency": "Moneda",
    "Seller Type": "Tipo de vendedor",
    "Has Rear Camera": "Con cámara de retroceso",
}
SOURCE_TO_ENGLISH_COLUMN = {source: english for english, source in ENGLISH_TO_SOURCE_COLUMN.items()}


def extract_features(dataset_path, selected_features):
    """Load selected dataset columns and return them renamed to English names."""
    dataframe = pd.read_csv(dataset_path)

    source_columns = []
    for feature_name in selected_features:
        source_column = ENGLISH_TO_SOURCE_COLUMN.get(feature_name, feature_name)
        if source_column not in dataframe.columns:
            raise KeyError(f"Feature '{feature_name}' could not be resolved in dataset columns.")
        source_columns.append(source_column)

    features = dataframe[source_columns].rename(columns=SOURCE_TO_ENGLISH_COLUMN)
    return features
