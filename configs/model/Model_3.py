model = "XG_Boost.py"
cv = 10
params = {
    "max_depth": [5, 6],                # Profundidad máxima de un árbol
    "learning_rate": [0.1, 0.01, 0.05],          # Tasa de aprendizaje
    "n_estimators": [100, 150, 200],           # Número de árboles
    "subsample": [0.8, 0.9, 1.0],                      # Proporción de muestras usadas para entrenar cada árbol
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],     # Proporción de características usadas para cada árbol
    "gamma": [0, 0.1, 0.2, 0.3, 0.4],                  # Mínima reducción de pérdida para hacer una partición
    "reg_alpha": [0, 0.01, 0.1, 1],                    # Término de regularización L1
    "reg_lambda": [1, 0.1, 0.01],                      # Término de regularización L2
    "scale_pos_weight": [1, 1.5, 2],                   # Control de balanceo de clases
    "min_child_weight": [1, 2, 3, 4]                   # Mínima suma de pesos de instancias para hacer una partición
}