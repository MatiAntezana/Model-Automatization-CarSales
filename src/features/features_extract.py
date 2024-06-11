import pandas as pd
import importlib

def extract_features(path_dataset, features_used):
    df = pd.read_csv(path_dataset)
    features = df[features_used]
    return features