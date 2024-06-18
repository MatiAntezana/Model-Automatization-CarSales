import joblib
import os

def model_save_result(model, route_folder):
    file_txt_results = os.path.join(route_folder, f"results.txt")
    
    with open(file_txt_results, 'w') as file_txt:
        file_txt.write(f"MAE = {model.MAE}\nMSE = {model.MSE}\nRMSE = {model.RMSE}\nR2 = {model.r2}")

def model_save_in_folder(model, route_folder):
    route_model = os.path.join(route_folder, 'model.pkl')
    joblib.dump(model.model, route_model)

