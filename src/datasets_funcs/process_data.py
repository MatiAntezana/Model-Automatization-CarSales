import pandas as pd

def read_data(filename):
    df = pd.read_csv(filename)
    return df.values

def identify_type_cars(data):
    dic = {}
    for x in data:
        if x[0] not in dic:
            dic[x[0]] = 1
        else:
            dic[x[0]] += 1
    return dic

