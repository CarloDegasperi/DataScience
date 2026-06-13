# Modelli di Regressione

## Obiettivo

Prevedere le concentrazioni degli inquinanti atmosferici utilizzando:

- dati meteorologici;
- informazioni temporali;
- misure storiche degli inquinanti.

Gli inquinanti considerati sono:

- PM10
- PM2.5
- O3

---

## Notebook

### 06_regression_preparation.ipynb

Costruzione dei dataset per la regressione.

Sono generati dataset distinti per:

- previsione oraria/giornaliera;
- singolo inquinante.

Le feature includono:

- variabili meteo;
- variabili temporali periodiche;
- inquinanti con differenti lag temporali.

---

### 07_regression.ipynb

Primi modelli di regressione e confronto delle prestazioni.

Possibili attività:

- training;
- validazione;
- confronto tra algoritmi;
- analisi degli errori.

---

### 08_regression_random_forest.ipynb

Applicazione di Random Forest Regressor e XGBoost.

Analisi:
- tuning degli iperparametri;
- Time Series Cross Validation;
- feature importance;
- confronto train/test.

---

## Metriche utilizzate

- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R²

---

## Nota metodologica

Poiché il problema è di natura temporale, la validazione deve rispettare l'ordine cronologico dei dati. Per questo motivo è importante l'utilizzo di:

- train/test split temporale;
- TimeSeriesSplit;
- metriche calcolate su dati futuri non osservati durante il training.