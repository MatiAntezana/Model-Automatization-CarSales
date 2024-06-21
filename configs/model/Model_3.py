model = "XG_Boost.py"
cv = 8
params = {"max_depth": [3, 4, 5],
    "learning_rate": [0.1, 0.01],
    "n_estimators": [50, 100, 150],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0]
}