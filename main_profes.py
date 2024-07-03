from src.features.features_extract import *

def extract_features_test(path_dataset, features_used):
    features = pd.read_csv(path_dataset)
    features = correct_dataset(features)

    features = transform_price(features, features_used)

def test_modelo(path_csv):
    features = extract_features_test(path_csv)
    