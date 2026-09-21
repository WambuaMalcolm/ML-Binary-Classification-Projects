from src.feature_engineering import MedicalFeatureEngineer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from imblearn.pipeline import Pipeline
def to_dataframe(X, columns):
    return pd.DataFrame(X, columns=columns)
def preprocess_pipeline()
    pipe = Pipeline([
        ('imp', IterativeImputer(max_iter=10, random_state=42)),
        ('to_df', FunctionTransformer(lambda X: to_dataframe(X, X_train.columns), validate=False)),
        ('feature_engineering', MedicalFeatureEngineer()),
        ('sl', MinMaxScaler()),
        ('smp', RandomOverSampler(random_state=42)),
        ('fe', PCA(n_components=2))
    ])
