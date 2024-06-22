model = "Neuronal_Network.py"
cv = 8
params = {
    "activation": ["relu", "tanh"],
    "solver": ["adam", "lbfgs"],
    "alpha": [0.001, 0.01],
    "learning_rate": ["constant", "adaptive"],
    'max_iter': [500]
}