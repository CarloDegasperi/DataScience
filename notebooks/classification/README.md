# Classificazione

Dopo aver preparato e preprocessato il DataFrame nel notebook `EDA.ipynb`, l’obiettivo successivo è quello di predire l’indice di qualità dell’aria (EAQI) in una determinata ora e giorno, utilizzando come input i dati raccolti nelle ore e nei giorni precedenti.

L’indice EAQI può assumere cinque diverse classi qualitative:

- `good`
- `fair`
- `moderate`
- `poor`
- `awful`

corrispondenti rispettivamente ai valori:

- Buono
- Discreto
- Moderato
- Scadente
- Pessimo

Il problema affrontato è quindi un problema di **classificazione multiclasse**.

La classificazione rientra nella branca del *Machine Learning supervisionato* (*Supervised Learning*), nella quale il modello viene addestrato utilizzando dati etichettati. Tra i diversi algoritmi studiati durante il corso, quelli più adatti al problema di classificazione multiclasse considerato sono:

- **Logistic Regression** con funzione **Softmax**
- **Random Forest Classifier**

Nei notebook successivi verranno implementati, addestrati e confrontati entrambi i modelli, analizzandone prestazioni e capacità predittive.