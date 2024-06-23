model = "Neuronal_Network.py"
cv = 10
params = {
    "activation": ["relu", "tanh"],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    "learning_rate": ["adaptive"],
    'max_iter': [3000, 5000]
}