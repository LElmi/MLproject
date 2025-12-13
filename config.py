from src.activationf.relu import relu
from src.activationf.sigmoid import sigmaf

# ============================
#          NN config
# ============================


# ======== PATHS DATA ========
PATH_DT = "data/training_data/ML-CUP25-TR.csv"


# ======== UNITS SIZE ========
N_HIDDENL1 = 64
N_HIDDENL2 = 32
N_OUTPUTS = 4


# ======= ACTIVATION F =======
FUN_ACT = relu


# ====== LEARNING RATE =======
LEARNING_RATE = 0.000025


# ======== N EPOCHS ==========
EPOCHS = 1500