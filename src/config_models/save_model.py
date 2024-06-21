import joblib
import os

def model_save_result(model, route_folder):
    file_txt_results = os.path.join(route_folder, f"results.txt")
    
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

