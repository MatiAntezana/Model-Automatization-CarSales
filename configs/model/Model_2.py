model = "Random_Forest.py"
cv = 10
params = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [7, 8],
    "min_samples_split": [4, 10, 15],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],
    "oob_score": [True, False],
    "random_state": [42],
}