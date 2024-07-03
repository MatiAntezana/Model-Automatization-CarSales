import logging
from sklearn.model_selection import train_test_split,GridSearchCV
import pandas as pd

def assign_drop_sets(set_1, set_2, type_marca, cant_marca_set_1):
    list_muestras = set_2[set_2['Marca Original'] == type_marca]
    index_test = list_muestras.index[0]
            
    muestra_drop_set_2 = set_2.loc[index_test]
    set_2 = set_2.drop(index=index_test)
            
    key_remplace_set_1 = None
    for key_, value_ in cant_marca_set_1.items():
        if value_ > 1:
            key_remplace_set_1 = key_
            cant_marca_set_1[key_] -= 1
            break

    index_train = set_1[set_1['Marca Original'] == key_remplace_set_1].index[0]
    muestra_drop_set_1 = set_1.loc[index_train]

    set_1 = set_1.drop(index=index_train)

    set_1 = pd.concat([set_1, muestra_drop_set_2], axis=0)
    set_2 = pd.concat([set_2, muestra_drop_set_1], axis=0)

    return set_1, set_2, cant_marca_set_1

def variability_set(set_1, set_2, original_features, cat_unicas, condition):
    type_marca_types = original_features['Marca Original'].unique()
    
    cant_marca_set_1 = set_1['Marca Original'].value_counts().to_dict()

    for type_marca in type_marca_types:

        if type_marca not in set_1['Marca Original'].values:

            if type_marca in cat_unicas:
                if condition == True:
                    set_1, set_2, cant_marca_set_1 = assign_drop_sets(set_1, set_2, type_marca, cant_marca_set_1)
            else:
                set_1, set_2, cant_marca_set_1 = assign_drop_sets(set_1, set_2, type_marca, cant_marca_set_1)

    return set_1, set_2

def get_sets(features, data_used):
    X = features.drop(["Precio"],axis=1)
    Y = features["Precio"]
    param_sets = data_used.param_sets

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = param_sets["Set_Test"], random_state=data_used.seed_random)

    train_set = pd.concat([X_train, Y_train], axis=1)

    test_set = pd.concat([X_test, Y_test], axis=1)

    cat_unicas = features['Marca Original'].value_counts()[features['Marca Original'].value_counts() == 1].index.tolist()

    train_set, test_set = variability_set(train_set, test_set, features, cat_unicas, True)

    train_set = train_set.drop("Marca Original",axis=1)
    test_set = test_set.drop("Marca Original",axis=1)

    return train_set, test_set