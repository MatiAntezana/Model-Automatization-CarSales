model = "Neuronal_Network.py"
cv = 10
params = {
    'hidden_layer_sizes': [(50,), (50, 50), (100, 50), (100, 100, 50), (100,100,40,100)],
    "activation": ["relu"],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    "learning_rate": ["adaptive"],
    'max_iter': [3000]
}