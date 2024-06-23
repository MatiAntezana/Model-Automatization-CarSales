model = "Extra_Tree_Regressor.py"
cv = 10
params = {
    "n_estimators": [50, 100, 200, 300],  # Número de árboles en el bosque
    "max_features": [None, "sqrt", "log2"],  # Número de características a considerar para la mejor división
    "max_depth": [5, 6, 7],  # Profundidad máxima del árbol
    "min_samples_split": [2, 5, 10, 15],  # Número mínimo de muestras necesarias para dividir un nodo
    "min_samples_leaf": [1, 2, 4, 6],  # Número mínimo de muestras necesarias en un nodo hoja
    "bootstrap": [True, False],  # Si se deben usar muestras de bootstrap al construir árboles
    "random_state": [0, 42]  # Semilla para el generador de números aleatorios
}