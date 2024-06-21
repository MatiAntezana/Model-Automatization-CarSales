model = "Extra_Tree_Regressor.py"
cv = 8
params = {
    "n_estimators": [50, 100, 200],
    "max_features": [1,"sqrt", "log2"],
    "max_depth": [5, 6, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}