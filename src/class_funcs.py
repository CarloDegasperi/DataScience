import pandas as pd
import numpy as np
from sklearn import metrics

def get_performance_param(model, X, y, thr = 0.5):

    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= thr).astype(int)

    accuracy_score = metrics.accuracy_score(y, y_pred)
    precision_score = metrics.precision_score(y, y_pred)
    recall_score = metrics.recall_score(y, y_pred)
    f1_score = metrics.f1_score(y, y_pred)
    MCC_score = metrics.matthews_corrcoef(y, y_pred)

    print(f'Accuratezza: {accuracy_score:.5f}')
    print(f'Precisione: {precision_score:.5f}')
    print(f'Recall: {recall_score:.5f}')
    print(f'F1 score: {f1_score:.5f}')
    print(f'MCC: {MCC_score:.5f}')

    return None


