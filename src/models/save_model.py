import joblib
import os

def model_save_in_folder(model, route_folder):
    route_model = os.path.join(route_folder, 'model.pkl')
    joblib.dump(model.model, route_model)

