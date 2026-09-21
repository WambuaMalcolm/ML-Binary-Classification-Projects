# Custom transformer
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class MedicalFeatureEngineer(BaseEstimator, TransformerMixin):
    def _init_(self):
        self.feature_names_out_ = None
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X, columns=self.feature_names_in_) if not isinstance(X, pd.DataFrame) else X.copy()
        
        # Engineer features
        X['BMI_Obese'] = (X['BMI'] >= 30).astype(int)
        X['BMI_underweight'] = (X['BMI']<18).astype(int)
        X['Glucose_Diabetes'] = (X['Glucose'] >= 126).astype(int)
        X['Age_lower'] = (X['Age']>=18).astype(int)
        X['Age_Senior'] = (X['Age'] >= 50).astype(int)
        X['Low_glucose'] = (X['Glucose']<44).astype(int)
        X['low_chance'] = (X['DiabetesPedigreeFunction']<0.088).astype(int)
        X['High_chance'] = (X['DiabetesPedigreeFunction']>2.42).astype(int)

        # # Ratio Features
        # X['Glucose_BMI_Ratio'] = X['Glucose'] / (X['BMI']+1e-5)
        # X['Glucose_Age_Ratio'] = X['Glucose'] / (X['Age']+1e-5)
        # X['Glucose_DPF_Ratio'] = X['Glucose'] / (X['DiabetesPedigreeFunction']+1e-5)
        
        self.feature_names_out_ = X.columns.tolist()
        return X
    
    def fit_transform(self, X, y=None):
        self.feature_names_in_ = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f'feature_{i}' for i in range(X.shape[1])]
        return self.fit(X, y).transform(X)
    
    def get_feature_names_out(self, input_features=None):
        return self.feature_names_out_