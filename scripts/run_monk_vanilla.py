from src.activationf.leaky_relu import leaky_relu
from src.training.trainer.trainer import Trainer
from src.activationf.sigmoid import sigmaf
from src.activationf.tanh import tanh
from src.utils import load_monks_data
from src.training.validation import Stratified_Split
import matplotlib.pyplot as plt
import numpy as np

x_i, d = load_monks_data("../data/monk/train_data/monks-1.train")
x_i = x_i.to_numpy().astype(np.float64)
d = d.to_numpy().astype(np.float64)

print(f"Shape input dopo encoding: {x_i.shape}")

trainer = Trainer(
    input_size=x_i.shape[1],
    units_list=[4],
    n_outputs=1,
    f_act_hidden=leaky_relu,
    f_act_output=sigmaf,
    learning_rate=0.1,
    use_decay=False,
    decay_factor=95.0,
    decay_step=5,
    batch=False,
    epochs=500,
    early_stopping=False,
    epsilon=0.01,
    patience=20,
    momentum=True,
    alpha_mom=0.5,
    max_gradient_norm=10,
    split=20,
    verbose=False,
    validation=False,
    lambdal2=0.0
)
x_i_tr, d_tr, x_i_vl, d_vl = Stratified_Split.hold_out_validation_stratified(
    x_i, d, 20, random_state=None
)

x_i_test, d_test = load_monks_data("../data/monk/test_data/monks-1.test")
x_i_test = x_i_test.to_numpy().astype(np.float64)
d_test = d_test.to_numpy().astype(np.float64)

train_mees = []
train_mses = []
test_accuracies = []
test_mees = []
test_mses = []
n_epochs = trainer.epochs

print("Inizio Training...")

for epoch in range(n_epochs):

    epoch_metrics = trainer._run_epoch(x_i, d, n_patterns=x_i.shape[0])

    # MEE e MSE dal training (epoch_metrics ritorna grad, mse_tr e mee_tr)
    train_mees.append(epoch_metrics["mee_tr"])
    train_mses.append(epoch_metrics["mse_tr"])

    matched = 0
    all_test_outputs = []

    for i, x_i_test_pattern in enumerate(x_i_test):
        layer_results, _ = trainer.neuraln.forward_network(
            x_i_test_pattern,
            leaky_relu,
            sigmaf
        )
        output = layer_results[-1]
        all_test_outputs.append(output)

        prediction = 1.0 if output >= 0.5 else 0.0
        if prediction == d_test[i]:
            matched += 1

    accuracy = matched / len(d_test)
    test_accuracies.append(accuracy)

    ee_test = np.linalg.norm(
        np.array(all_test_outputs).flatten() - d_test.flatten(), axis=0
    )
    mee_test = ee_test / len(d_test)
    test_mees.append(mee_test)

    mse_test = np.mean(
        (np.array(all_test_outputs).flatten() - d_test.flatten()) ** 2
    )
    test_mses.append(mse_test)

    if (epoch + 1) % 25 == 0:
        print(
            f"Epoch {epoch+1}/{n_epochs} | "
            f"Train MEE: {train_mees[-1]:.4f}, Train MSE: {train_mses[-1]:.4f} | "
            f"Test Acc: {accuracy*100:.2f}%, Test MEE: {mee_test:.4f}, Test MSE: {mse_test:.4f}"
        )

print("Fine Training.")
plt.figure(figsize=(10, 6))

plt.plot(range(1, n_epochs + 1), test_accuracies, label="Accuracy Test", color="blue")

plt.plot(range(1, n_epochs + 1), train_mees, label="MEE Train", color="cyan")
plt.plot(range(1, n_epochs + 1), train_mses, label="MSE Train", color="orange")

plt.plot(range(1, n_epochs + 1), test_mees, label="MEE Test", color="green")
plt.plot(range(1, n_epochs + 1), test_mses, label="MSE Test", color="red")

plt.xlabel("Epoche")
plt.title("Learning Curve (Train + Test Metrics) - MONK‑1")
plt.legend()
plt.grid(True)

plt.show()

print(f"\nVALORI FINALI:")
print(f"Train MEE: {train_mees[-1]:.4f}")
print(f"Train MSE: {train_mses[-1]:.4f}")
print(f"Test Accuracy: {test_accuracies[-1]*100:.2f}%")
print(f"Test MEE: {test_mees[-1]:.4f}")
print(f"Test MSE: {test_mses[-1]:.4f}")
