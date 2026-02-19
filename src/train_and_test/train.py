TARGET_COLUMN = "Price"


def train_model(model, train_set):
    """Train a model using the provided training dataframe."""
    features = train_set.drop(TARGET_COLUMN, axis=1)
    target = train_set[TARGET_COLUMN]
    model.train(features, target)
    return 0
