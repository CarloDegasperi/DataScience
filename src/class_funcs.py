import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

################################################################################################################################################

def get_train_valid_test(dataset_df, frac_train, frac_valid, feature_cols):
    # troviamo i limiti
    unique_days = np.sort(dataset_df['day'].unique())
    split_day_1 = unique_days[int(len(unique_days) * frac_train)]
    split_day_2 = unique_days[int(len(unique_days) * (frac_train + frac_valid))]

    ## dividiamo il dataset nelle tre parti (non togliamo i bordi perché assumiamo portino ad un data leakage trascurabile)
    train_df = dataset_df[dataset_df['day'] <= split_day_1]
    valid_df = dataset_df[(split_day_1 < dataset_df['day']) & (dataset_df['day'] <= split_day_2)]
    test_df  = dataset_df[dataset_df['day'] > split_day_2]

    X_train = train_df[feature_cols]
    X_valid = valid_df[feature_cols]
    X_test = test_df[feature_cols]

    y_train = train_df['target']
    y_valid = valid_df['target']
    y_test = test_df['target']

    return X_train, X_valid, X_test, y_train, y_valid, y_test

################################################################################################################################################

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
    
################################################################################################################################################

def display_confusion_matrix(model, title1, X1, y1, title2 = None, X2 = None, y2 = None, thr = 0.5):
    
    if X2 is None and y2 is None:
        plt.figure(figsize=(12, 5))

        y_proba = model.predict_proba(X1)[:, 1]
        y_pred = (y_proba >= thr).astype(int)
        metrics.ConfusionMatrixDisplay.from_predictions(y1, y_pred, display_labels=model.classes_)
        plt.title(title1)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        y_proba1 = model.predict_proba(X1)[:, 1]
        y_pred1 = (y_proba1 >= thr).astype(int)
        metrics.ConfusionMatrixDisplay.from_predictions(y1, y_pred1, display_labels=model.classes_, ax = axes[0])
        axes[0].set_title(title1)

        y_proba2 = model.predict_proba(X2)[:, 1]
        y_pred2 = (y_proba2 >= thr).astype(int)
        metrics.ConfusionMatrixDisplay.from_predictions(y2, y_pred2, display_labels=model.classes_, ax = axes[1])
        axes[1].set_title(title2)

        plt.tight_layout()
        plt.show()

################################################################################################################################################

def get_performance_param(model, X, y, binary = True, thr = 0.5):

    if binary:
        y_proba = model.predict_proba(X)[:, 1]
        y_pred = (y_proba >= thr).astype(int)

        accuracy_score = metrics.accuracy_score(y, y_pred)
        precision_score = metrics.precision_score(y, y_pred)
        recall_score = metrics.recall_score(y, y_pred)
        f1_score = metrics.f1_score(y, y_pred)
        MCC_score = metrics.matthews_corrcoef(y, y_pred)

        print(f'Accuratezza:    {accuracy_score:.5f}')
        print(f'Precisione:     {precision_score:.5f}')
        print(f'Recall:         {recall_score:.5f}')
        print(f'F1 score:       {f1_score:.5f}')
        print(f'MCC:            {MCC_score:.5f}')
    
    else:
        y_pred = model.predict(X)
        accuracy_score = metrics.accuracy_score(y, y_pred)
        precision_score = metrics.precision_score(y, y_pred, average='weighted')
        recall_score = metrics.recall_score(y, y_pred, average='weighted')
        f1_score = metrics.f1_score(y, y_pred, average='weighted')
        MCC_score = metrics.matthews_corrcoef(y, y_pred)

        print(f'Accuratezza:    {accuracy_score:.5f}')
        print(f'Precisione:     {precision_score:.5f}')
        print(f'Recall:         {recall_score:.5f}')
        print(f'F1 score:       {f1_score:.5f}')
        print(f'MCC:            {MCC_score:.5f}')
        
    return None

################################################################################################################################################

def display_confusion_matrix_multiclass(model, class_order, title1, X1, y1, title2 = None, X2 = None, y2 = None):
    if X2 is None and y2 is None:
        plt.figure(figsize=(12, 5))

        y_pred = model.predict(X1)
        metrics.ConfusionMatrixDisplay.from_predictions(y1, y_pred, labels=class_order, display_labels=class_order)
        plt.title(title1)

    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        y_pred1 = model.predict(X1)
        metrics.ConfusionMatrixDisplay.from_predictions(y1, y_pred1, labels=class_order, display_labels=class_order, ax = axes[0])
        axes[0].set_title(title1)

        y_pred2 = model.predict(X2)
        metrics.ConfusionMatrixDisplay.from_predictions(y2, y_pred2, labels=class_order, display_labels=class_order, ax = axes[1])
        axes[1].set_title(title2)

        plt.tight_layout()
        plt.show()

################################################################################################################################################

def get_AQI_pol(value, intervals):
    if pd.isna(value):
        return np.nan

    for i, (low, high) in enumerate(intervals):
        if high == np.inf:
            if value >= low:
                return 20 * (i + 1)
        if low <= value < high:
            return 20 / (high - low) * (value - low) + 20 * i
               
    return np.nan

################################################################################################################################################

def display_importances(model, feature_cols):
    importance_df = pd.DataFrame({'importance': model.feature_importances_}, index=feature_cols)

    importance_df = importance_df.sort_values(by='importance', ascending=True)

    importance_df.plot.barh(figsize=(9,7))

    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    plt.grid()
    plt.show()

