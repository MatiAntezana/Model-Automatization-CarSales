import logging
from src.features.features_extract import transform_inverse_scalar

def apply_normalization(set_train, set_test, features_used):
    if features_used.apply_normalization_kilometros == True:
        set_train["Kilómetros"] = transform_inverse_scalar(set_train["Kilómetros"])
        set_test["Kilómetros"] = transform_inverse_scalar(set_test["Kilómetros"])
    
    return set_train, set_test