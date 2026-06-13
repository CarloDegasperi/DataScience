# Progetto Data Science – Qualità dell'Aria in Trentino

## Obiettivo del progetto
Questo progetto analizza dati meteorologici, ambientali e di qualità dell'aria della Provincia Autonoma di Trento con l'obiettivo di:

- comprendere l'evoluzione temporale degli inquinanti atmosferici;
- individuare relazioni tra meteo e qualità dell'aria;
- costruire modelli di classificazione della qualità dell'aria;
- costruire modelli di regressione per la previsione delle concentrazioni di inquinanti.

Gli inquinanti principali considerati sono:
- PM10
- PM2.5
- O3 (Ozono)

---

## Struttura del repository

### data/
Contiene i dati utilizzati durante il progetto.

- `raw/`: dati originali provenienti dalle diverse sorgenti.
- `external/`: dati esterni e di supporto.
- `processed/`: dataset puliti e preparati per la modellazione.

### notebooks/

Workflow principale del progetto:

1. `01_dataframe_organization.ipynb`
   - acquisizione e organizzazione dei dati;
   - integrazione delle sorgenti informative.

2. `02_exploratory_data_analysis.ipynb`
   - analisi esplorativa;
   - studio delle distribuzioni;
   - analisi temporali e geografiche.

3. Cartella `classification/`
   - preparazione dei dataset;
   - regressione logistica;
   - Random Forest per classificazione.

4. Cartella `regression/`
   - preparazione dei dataset;
   - modelli di regressione;
   - Random Forest e altri approcci predittivi.

### src/

Contiene le funzioni di supporto utilizzate dai notebook.

- `EDA_funcs.py`
- `class_funcs.py`
- `regr_funcs.py`

### images/

Grafici e immagini utilizzati per l'analisi e la documentazione.

---

## Pipeline del progetto

1. Raccolta dati
2. Pulizia e integrazione
3. Feature engineering
4. Exploratory Data Analysis
5. Classificazione della qualità dell'aria
6. Regressione delle concentrazioni degli inquinanti
7. Valutazione dei modelli

---

## Dipendenze

Installazione:

```bash
pip install -r requirements.txt
```

---

## Autori

- Carlo Degasperi - CarloDegasperi
- Andrea Storchi - ExtravagantReplicator
Progetto sviluppato nell'ambito di un percorso di Data Science.
