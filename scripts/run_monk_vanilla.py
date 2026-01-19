from src.activationf.leaky_relu import leaky_relu
from src.training.trainer.trainer import Trainer
from src.activationf.sigmoid import sigmaf
from src.activationf.relu import relu
from src.activationf.tanh import tanh
from src.utils import load_monks_data
from src.training.validation import Stratified_Split
import matplotlib.pyplot as plt
import numpy as np

# 1. CARICAMENTO DATI TRAIN
x_i, d = load_monks_data("../data/monk/train_data/monks-1.train")
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

print(f"Shape input dopo encoding: {x_i.shape}")

# 2. CONFIGURAZIONE OTTIMALE PER MONK-1
trainer = Trainer(
    input_size=x_i.shape[1],
    units_list=[8],
    n_outputs=1,
    f_act_hidden=leaky_relu,
    f_act_output=sigmaf,
    learning_rate=0.05,
    use_decay=False,
    decay_factor=0.0,
    decay_step=0,
    batch=False,
    epochs=5000,
    early_stopping=False,
    epsilon=0.01,
    patience=20,
    momentum=True,
    alpha_mom=0.2,
    max_gradient_norm=100,
    split=20,
    verbose=False,   # disattivo verbose per non intasare output
    validation=False,
    lambdal2=0.00001
)

# split (anche se non usato esplicitamente)
x_i_tr, d_tr, x_i_vl, d_vl = Stratified_Split.hold_out_validation_stratified(
    x_i, d, 20, random_state=None
)

# 3. CARICAMENTO TEST SET
x_i_test, d_test = load_monks_data("../data/monk/test_data/monks-1.test")
x_i_test = x_i_test.to_numpy().astype(np.float64)
d_test = d_test.to_numpy().astype(np.float64)

# -------------------------------
# TRAINING + TEST ACCURACY PER EPOCA
# -------------------------------
test_accuracies = []
n_epochs = trainer.epochs

print("Inizio Training...")

for epoch in range(n_epochs):

    # ---- training di una singola epoca ----
    trainer._run_epoch(x_i, d, n_patterns=x_i.shape[0])

    # ---- valutazione sul test set ----
    matched = 0
    for i, x_i_test_pattern in enumerate(x_i_test):
        layer_results, _ = trainer.neuraln.forward_network(
            x_i_test_pattern,
            leaky_relu,
            sigmaf
        )
        output = layer_results[-1]
        prediction = 1.0 if output >= 0.5 else 0.0

        if prediction == d_test[i]:
            matched += 1

    accuracy = matched / len(d_test)
    test_accuracies.append(accuracy)

    # stampa ogni 100 epoche
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{n_epochs} - Test Accuracy: {accuracy*100:.2f}%")

print("Fine Training.")

# -------------------------------
# PLOT ACCURACY VS EPOCHE
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs + 1), test_accuracies, color="blue")
plt.xlabel("Epoche")
plt.ylabel("Accuracy Test")
plt.title("Accuracy sul Test Set vs Epoche (MONK-1)")
plt.grid(True)
plt.show()

# -------------------------------
# ACCURACY FINALE
# -------------------------------
print(f"\nACCURACY FINALE ON TEST SET = {test_accuracies[-1] * 100:.2f}%")
