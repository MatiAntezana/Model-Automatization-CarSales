import numpy as np


def convert_usd_to_ars(feature_config, features):
    """Convert USD-denominated prices into ARS using the configured exchange rate."""
    for row_index, currency in features["Currency"].items():
        if currency == "U$S":
            ars_price = np.float64(feature_config.usd_to_ars_rate * features.at[row_index, "Price"])
            features.at[row_index, "Price"] = ars_price
            features.at[row_index, "Currency"] = "$"

    return features


def data_modify(feature_config, features):
    """Apply configured feature-level transformations to a dataframe split."""
    return convert_usd_to_ars(feature_config, features)
