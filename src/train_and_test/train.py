def train_model(model, set_train):
    X = set_train.drop('Precio', axis=1)
    Y = set_train['Precio']
    model.train(X, Y)

    return 0