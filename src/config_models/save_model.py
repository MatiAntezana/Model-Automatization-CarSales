import joblib
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

def graficos_reultados(set_tests, model, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    x = set_tests.drop("Precio",axis=1)
    y = set_tests["Precio"]
    y_pred = model.model.predict(x)

    RMSE = np.sqrt(mean_squared_error(y, y_pred))
    MAE = mean_absolute_error(y, y_pred)

    file_name = 'grafico_real_vs_prediccion.pdf'
    file_path = os.path.join(folder_path, file_name)

    legend_size = 30
    label_size = 41
    ticks_size = 23

    plt.figure(figsize=(25, 20))    
    plt.scatter(y, y_pred, s=1.2, color='orange', label='Valor del precio')

    plt.plot([min(y), max(y)], [min(y), max(y)], color='blue', linestyle='--', label=f'Línea de referencia\nRMSE={RMSE:.2f}\nMAE={MAE:.2f}')

    plt.xlabel('Precio real (USD)', fontsize=label_size)
    plt.ylabel('Precio estimado (USD)', fontsize=label_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.legend(fontsize=legend_size)
    plt.savefig(file_path, format="pdf", bbox_inches="tight")
    plt.show()

def model_save_result(model, route_folder, set_tests):
    file_txt_results = os.path.join(route_folder, f"results.txt")
    
    graficos_reultados(set_tests, model, route_folder)

    with open(file_txt_results, "w") as file_txt:
        file_txt.write(f"MAE = {model.MAE}\nMSE = {model.MSE}\nRMSE = {model.RMSE}\nR2 = {model.r2}")

def model_save_best_params_of_params(model, route_folder):
    # Guarda los mejores parametros si el modelo tiene varias combinaciones posibles de parametros
    file_txt_results = os.path.join(route_folder, f"best_param.txt")
    with open(file_txt_results, "w") as file_txt:
        file_txt.write(f"Best Params = {model.best_param}")

def model_save_in_folder(model, route_folder):
    route_model = os.path.join(route_folder, "model.pkl")
    joblib.dump(model.model, route_model)

