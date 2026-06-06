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

def display_abs_coefs(model, feature_cols):
    
    coefs_df = pd.DataFrame(model.coef_.T,columns=['coefficients'],index=feature_cols)

    plt.figure(figsize=(12,9))

    ordered_coefs_df = coefs_df.abs().sort_values('coefficients', ascending=False)
    sns.barplot(x=ordered_coefs_df['coefficients'], y=ordered_coefs_df.index)
    plt.title('Absolute value coefficients of the Logistic Regression model')
    plt.xlabel('Absolute value of coefficients')
    plt.ylabel('Features')
    plt.grid()
    plt.show()

################################################################################################################################################

def evaluation(model, X, y):
    
    y_pred = model.predict(X)
    mse = metrics.mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = metrics.mean_absolute_error(y, y_pred)
    r2 = metrics.r2_score(y, y_pred)
    
    print('Mean Squared Error (MSE):         ', mse)
    print('Root Mean Squared Error (RMSE):   ', rmse)
    print('Mean Absolute Error (MAE):        ', mae)
    print('Coefficient of Determination (R2):', r2)

################################################################################################################################################

def recover_hour(sin_hour, cos_hour):
    angle = np.arctan2(sin_hour, cos_hour)
    angle = np.mod(angle, 2 * np.pi)

    hour = angle * 24 / (2 * np.pi)
    return hour

################################################################################################################################################

def plot_predictions(model, X1, y1, X2, y2):

    graph1_df = X1.copy()
    graph1_df['actual'] = y1
    graph1_df['prediction'] = model.predict(X1)
    graph1_df['time'] = (graph1_df['day'] + recover_hour(graph1_df['sin_hour'], graph1_df['cos_hour']) / 24)

    graph2_df = X2.copy()
    graph2_df['actual'] = y2
    graph2_df['prediction'] = model.predict(X2)
    graph2_df['time'] = (graph2_df['day'] + recover_hour(graph2_df['sin_hour'], graph2_df['cos_hour']) / 24)

    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    axes = axes.flatten()

    stations = [col for col in graph1_df.columns if col.startswith('station_')]

    i = 0

    for sta in stations:

        station_data_tr = graph1_df[graph1_df[sta] == 1]
        station_data_te = graph2_df[graph2_df[sta] == 1]
        y_true = station_data_te["actual"]
        y_pred = station_data_te["prediction"]

        if station_data_tr.empty:
            continue

        sta_name = sta.removeprefix('station_')

        rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
        r2 = metrics.r2_score(y_true, y_pred)

        ax = axes[i]
        i += 1

        sns.lineplot(data=station_data_tr, x='time', y='actual', color='blue', label='actual in train', ax=ax)
        sns.lineplot(data=station_data_tr, x='time', y='prediction', color='red', label='prediction in train', ax=ax)
        
        sns.lineplot(data=station_data_te, x='time', y='actual', color='green', label='actual in test', ax=ax)
        sns.lineplot(data=station_data_te, x='time', y='prediction', color='orange', label='prediction in test', ax=ax)
        
        ax.set_title(f'Dati di {sta_name}\nRMSE_test={rmse:.2f}  R²_test={r2:.3f}')
        ax.set_ylabel('concentration [µg/m³]')
        ax.grid(True)
        ax.legend()

    for ax in axes[i:]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

################################################################################################################################################