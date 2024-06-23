model = "KNN.py"
cv = 10
params = {
    "n_neighbors": [3, 5, 7, 9, 13, 14],
    "weights": ["uniform", "distance"],
    "metric": ["euclidean", "manhattan"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "leaf_size": [20, 30, 40, 50],
    "p": [1, 2]
}