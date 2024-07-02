model = "Random_Forest.py"
cv = 10
params = {
    "n_estimators": [50, 100, 150],    
    "max_depth": [4, 6],                
    "min_samples_split": [4, 10, 15],                # Número mínimo de muestras necesarias para dividir un nodo
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2", None],     # Número de características a considerar para el mejor split
    "oob_score": [True, False],                         # Si se usa la muestra fuera de bolsa para estimar la precisión generalizada
    "random_state": [42],                               # Estado inicial para la reproducibilidad
}