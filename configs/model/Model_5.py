model = "KNN.py"
cv = 8
params = {
    "n_neighbors": [3, 5, 7, 9, 13, 14],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "knn__leaf_size": [20, 30, 40, 50]
}