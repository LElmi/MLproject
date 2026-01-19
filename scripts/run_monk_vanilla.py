from src.training.trainer.trainer import Trainer
from src.activationf.sigmoid import sigmaf
from src.utils import load_monks_data
# Assicurati di importare la funzione one_hot definita sopra
# from src.utils import one_hot_encoding_monk
import numpy as np

import numpy as np

# 1. CARICAMENTO DATI TRAIN
x_i, d = load_monks_data("../data/monk/train_data/monks-1.train")
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

print(f"Shape input dopo encoding: {x_i.shape}")

# 2. CONFIGURAZIONE OTTIMALE PER MONK-1
trainer = Trainer(
    input_size=x_i.shape[1],
    units_list=[4],
    n_outputs=1,
    f_act_hidden=sigmaf,
    f_act_output=sigmaf,
    learning_rate=0.05,  # MONK disattivo
    use_decay=False,
    decay_factor=0.0,
    decay_step=0,
    batch=False,
    epochs=1000,
    early_stopping=False,
    epsilon=0.01,
    patience=20,
    momentum=False,
    alpha_mom=0.0,
    max_gradient_norm=5,
    split=20,
    verbose=True,
    validation=False,
    lambdal2=0
)

print("Inizio Training...")
tr_mee_history,tr_mse_history,vl_mee_history,vl_mse_history=trainer.fit(x_i, d)
print("Fine Training.")

# 3. CARICAMENTO E PREPARAZIONE TEST SET
x_i_test, d_test = load_monks_data("../data/monk/test_data/monks-1.test")
x_i_test = x_i_test.to_numpy().astype(np.float64)
d_test = d_test.to_numpy().astype(np.float64)

# 4. CALCOLO ACCURACY
matched = 0
test_size = x_i_test.shape[0]

for i, x_i_test_pattern in enumerate(x_i_test):
    # Forward pass

    layer_results, layer_nets = trainer.neuraln.forward_network(x_i_test_pattern, sigmaf, sigmaf)
    output = layer_results[-1]
    prediction = 1.0 if output >= 0.5 else 0.0

    if prediction == d_test[i]:
        matched += 1

print(f"\nACCURACY ON TEST SET = {matched / test_size * 100:.2f}%")