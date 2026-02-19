TARGET_COLUMN = "Price"


def test_model(model, test_set):
    """Evaluate a trained model on the provided test dataframe."""
    features = test_set.drop(TARGET_COLUMN, axis=1)
    target = test_set[TARGET_COLUMN]
    model.evaluate_model(features, target)
