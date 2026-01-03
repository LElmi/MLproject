from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf
from src.activationf.leaky_relu import leaky_relu
# --------------------  NN config:

# ======== PATHS DATA ========
#PATH_DT = "../data/training_data/ML-CUP25-TR.csv"  #<- CUP
PATH_DT = "data/monk/train_data/monks-1.train"
MONK= True #<- toggle per il monk

# ======== UNITS SIZE ========
UNITS_LIST = [8]
N_OUTPUTS = 1


# ======= ACTIVATION F =======
FUN_ACT = leaky_relu



# ====== LEARNING RATE ======= #<- viene usato come valore centrale per la grid search
LEARNING_RATE = 2e-3   #<- Funziona in batch ma da applicare il criterio di stop sennò scavalca il minimo
#LEARNING_RATE = 0.000025


# PRIMA IMPLEMENTAZIONE: LINEARE # 
USE_DECAY = True
DECAY_FACTOR = 0.99
DECAY_STEP = 10


# ========= BATCH ============
BATCH = True


# --------------------- STOP CRITERIA:

# ======== N EPOCHS ==========
EPOCHS = 3000 #usato anche per determinare il numero di epoche per ogni run di grid search, 3% epochs ogni run
# ======== Early Stopping oon/off =============
EARLY_STOPPING = True
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
ALPHA_MOM = 0.5 #<- viene usato come valore centrale per la grid search
# --------------------- HOLD OUT VALIDATION (SPLIT = percentuale di pattern tenuti da parte per la validation

RUN_HOLD_OUT_VALIDATION= True #toggle per runnare la validation in coda al training
SPLIT = 40

#------------------ Regolarizzazione
LAMBDA=0.01


VERBOSE = True