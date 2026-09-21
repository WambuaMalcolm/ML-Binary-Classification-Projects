import pandas as pd

def load_data(data_path='C:\Users\HP\Desktop\Binary Classification\Diabetes Prediction\data\diabetes.csv'):
    return pd.read_csv(data_path)