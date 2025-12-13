# Neural Network – Training from Scratch

Questo progetto implementa una **rete neurale feed-forward** sviluppata interamente da zero in Python, con l’obiettivo di comprendere a fondo il funzionamento di:

- forward pass
- backward pass
- funzioni di attivazione e loro derivate
- aggiornamento dei pesi tramite gradient descent

Il progetto è pensato a scopo **didattico e sperimentale**.

---

## Requisiti

Prima di avviare il progetto, assicurarsi di avere:

- Python ≥ 3.9
- un ambiente virtuale attivo (consigliato)

Installare le librerie necessarie con:

```bash
pip install -r requirements.txt # (da fare)
```

---

## Configurazione della rete neurale

La configurazione principale della rete è contenuta nel file:

```text
config.py
```

Tutti i parametri modificabili (path, dimensioni dei layer, funzione di attivazione, iperparametri di training) sono definiti lì, in modo da evitare valori hard-coded nel codice.

---

### 📁 Path dei dati

```python
# ======== PATHS DATA ========
PATH_DT = "data/training_data/ML-CUP25-TR.csv"
```

- Il path è **relativo alla root del progetto**.
- Attualmente è previsto un unico training set, ma la struttura è pensata per supportare in futuro più dataset (validation / test).

---

### 🧠 Dimensione dei layer

```python
# ======== UNITS SIZE ========
# Hidden layers (attualmente supportati: 2)
N_HIDDENL1 = 64
N_HIDDENL2 = 32
N_OUTPUTS = 4
```

- La rete utilizza attualmente **due hidden layer**.
- Il numero di neuroni per layer è configurabile.
- È prevista un’evoluzione verso una gestione più flessibile ad esempio tramite lista di layer

---

### ⚡ Funzione di attivazione

```python
# ======= ACTIVATION FUNCTION =======
FUN_ACT = relu
```

- La funzione di attivazione è applicata **a tutti gli hidden layer**.
- `FUN_ACT` è una funzione che implementa **sia l’attivazione che la derivata**.
- Firma attesa:

```python
def activation(x, derivative: bool = False):
    ...
```

- Se `derivative=False`, la funzione restituisce l’attivazione.
- Se `derivative=True`, restituisce la derivata della funzione.

La funzione `relu` è definita nel modulo dedicato alle funzioni di attivazione (es. `activations.py`).

---

### 📉 Learning rate

```python
# ====== LEARNING RATE =======
LEARNING_RATE = 0.000025
```

- Il learning rate è volutamente basso per la **stabilità del training online**

---

### 🔁 Numero di epoche

```python
# ======== N EPOCHS ==========
EPOCHS = 1500
```

- Numero totale di epoche di training.

---

## Avvio del training

Per avviare il training:

1. Spostarsi nella directory root del progetto
2. Eseguire lo script come **modulo Python**

```bash
python3 -m scripts.run_training
```

L’esecuzione come modulo garantisce la corretta risoluzione degli import interni.

---

## Note finali

- Il progetto **non utilizza framework di deep learning** (come PyTorch o TensorFlow).
- Tutte le operazioni (forward, backward, update dei pesi) sono implementate manualmente.
- Il codice è strutturato per essere facilmente estendibile e modificabile a fini sperimentali.

---

## Possibili estensioni future

- Supporto a un numero arbitrario di hidden layer
- Logging delle metriche di training

---

Questo progetto è stato sviluppato come supporto allo studio dei concetti trattati nel corso
*Machine Learning* dell’Università di Pisa, tenuto dal Prof. Alessio Micheli.
