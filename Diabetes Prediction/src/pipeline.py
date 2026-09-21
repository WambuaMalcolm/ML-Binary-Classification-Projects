from imblearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, classification_report, precision_score, roc_auc_score, balanced_accuracy_score, recall_score, f1_score)
from sklearn.decomposition import PCA


def to_dataframe(X, columns):
    return pd.DataFrame(X, columns=columns)
def Feature_Engineer_pipeline(Imputer, Scaler, Sampler, Model, X_train, y_train, X_test, y_test):

    pipe = Pipeline([
        ('imp', Imputer),
        ('to_df', FunctionTransformer(lambda X: to_dataframe(X, X_train.columns), validate=False)),
        ('feature_engineering', MedicalFeatureEngineer()),
        ('sl', Scaler),
        ('smp', Sampler),
        ('fe', PCA(n_components=2)),
        ('mdl', Model)
    ])


    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(classification_report(y_test, y_pred))
    return pd.DataFrame({
    'Metric': ['Accuracy', 'Balanced Accuracy', 'Precision', 'ROC AUC', 'Recall Score', 'f1 score'],
    'Score': [
        accuracy_score(y_test, y_pred),
        balanced_accuracy_score(y_test, y_pred),
        precision_score(y_test, y_pred),
        roc_auc_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred)
    ]
})    