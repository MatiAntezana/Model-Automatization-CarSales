model = "Random_Forest.py"
cv = 8
params = {"n_estimators": [50, 100, 150],           # Número de árboles en el bosque
    "max_depth": [4, 6],           # Profundidad máxima del árbol
    "min_samples_split": [2, 5, 10],           # Número mínimo de muestras necesarias para dividir un nodo
    "min_samples_leaf": [1, 2, 4],
    }