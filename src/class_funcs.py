import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

def display_abs_coefs(model, feature_cols, num_class = 2):
    if (num_class == 2):
        coefs_df = pd.DataFrame(model.coef_.T,columns=['coefficients'],index=feature_cols)

        plt.figure(figsize=(12,9))

        ordered_coefs_df = coefs_df.abs().sort_values('coefficients', ascending=False)
        sns.barplot(x=ordered_coefs_df['coefficients'], y=ordered_coefs_df.index)
        plt.title('Absolute value coefficients of the Logistic Regression model')
        plt.xlabel('Absolute value of coefficients')
        plt.ylabel('Features')
        plt.grid()
    else:
        coefs_df = pd.DataFrame(model.coef_.T,columns=model.classes_,index=feature_cols)
        
        plt.figure(figsize=(12,9))

        ordered_coefs_df = coefs_df.abs().sort_values('good', ascending=False)
        ordered_coefs_df.abs().plot.barh()
        
        sns.set_context('talk')
        plt.title('Absolute value coefficients of the Logistic Regression model')
        plt.xlabel('Absolute value of coefficients')
        plt.ylabel('Features')
        plt.grid()

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


