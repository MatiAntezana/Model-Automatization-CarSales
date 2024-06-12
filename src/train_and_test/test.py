def test_model(model, set_test):
    X = set_test.drop('Precio', axis=1)
    Y = set_test['Precio']
    model.evaluate_model(X, Y)