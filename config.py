from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf
from src.activationf.Leaky_relu import leaky_relu
# --------------------  NN config:

# ======== PATHS DATA ========
#PATH_DT = "../data/training_data/ML-CUP25-TR.csv"  #<- CUP
PATH_DT = "../data/training_data/monks-1.train"
MONK= True #<- toggle per il monk

# ======== UNITS SIZE ========
N_HIDDENL1 = 8
N_HIDDENL2 = 4

if MONK==True:
    N_OUTPUTS = 1
else:
    N_OUTPUTS = 4


# ======= ACTIVATION F =======
FUN_ACT = leaky_relu


# ====== LEARNING RATE ======= #<- viene usato come valore centrale per la grid search
LEARNING_RATE = 0.05e-3   #<- Funziona in batch ma da applicare il criterio di stop sennò scavalca il minimo
#LEARNING_RATE = 0.000025

# ========= BATCH ============
BATCH = True


# --------------------- STOP CRITERIA:

# ======== N EPOCHS ==========
EPOCHS = 1000 #usato anche per determinare il numero di epoche per ogni run di grid search, 3% epochs ogni run
# ======== Early Stopping oon/off =============
EARLY_STOPPING = False
# gradient_norm < EPSILON, quindi il gradiente non cresce abbastanza,
# serve per lo stopping criteria come limite inferiore, in percentuale
# ======== EPSILON ==========
#EPSILON = 0.001
EPSILON = 0.00000001

# Dopo quante epoche in cui non cresce il gradiente mi fermo
# ======= PATIENCE ==========
PATIENCE = 10


# --------------------- MOMENTUM:

MOMENTUM = True
ALPHA_MOM = 0.1 #<- viene usato come valore centrale per la grid search
# --------------------- HOLD OUT VALIDATION (SPLIT = percentuale di pattern tenuti da parte per la validation

RUN_HOLD_OUT_VALIDATION= True #toggle per runnare la validation in coda al training
SPLIT = 90

#------------------ Regolarizzazione
LAMBDA=0.01