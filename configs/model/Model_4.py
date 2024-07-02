model = "Extra_Tree_Regressor.py"
cv = 10
params = {
    "n_estimators": [50, 100, 200, 300],
    "max_features": [None, "sqrt", "log2"],
    "max_depth": [5, 6, 7],
    "min_samples_split": [2, 5, 10, 15],
    "min_samples_leaf": [1, 2, 4, 6],
    "bootstrap": [True, False],
    "random_state": [0, 42]
}