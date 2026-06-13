# Modelli di Classificazione

## Obiettivo

Stimare la classe di qualità dell'aria a partire da informazioni:

- meteorologiche;
- temporali;
- geografiche;
- storiche degli inquinanti.

## Notebook

### 03_classification_preparation.ipynb

Genera i dataset utilizzati per la classificazione.

Principali feature:

- stazione di misura (one-hot encoding);
- elevazione;
- informazioni temporali periodiche;
- variabili meteorologiche;
- valori storici degli inquinanti.

Output:
- dataset classificazione binaria;
- dataset classificazione multiclasse.

---

### 04_logistic_regression.ipynb

Implementazione e valutazione di modelli di regressione logistica.

Analisi:
- training;
- validazione;
- metriche di classificazione;
- interpretabilità dei coefficienti.

---

### 05_random_forest.ipynb

Applicazione di Random Forest alla classificazione.

Analisi:
- ricerca degli iperparametri;
- valutazione tramite cross-validation;
- importanza delle feature;
- confronto con la regressione logistica.

---

## Target

### Classificazione binaria

Separazione tra condizioni favorevoli e sfavorevoli della qualità dell'aria.

### Classificazione multiclasse

Classi:

- good
- fair
- moderate
- poor
- very poor

---

## Metriche utilizzate

- Accuracy
- Precision
- Recall
- F1-score
- Matrice di confusione
