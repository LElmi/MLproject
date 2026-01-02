def grid_search_train(self, input_matrix, d_matrix, epochs):

    n_patterns = input_matrix.shape[0]
    for i in range (epochs):
        epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

    self.mee_error_history.append(epoch_results["mee"])
    self.mse_error_history.append(epoch_results["mse"])

    return epoch_results["mee"]


def grid_search_train_with_early_stopping(self, input_matrix, d_matrix, epochs):
    n_patterns = input_matrix.shape[0]
    gradient_misbehave = 0
    prev_gradient_norm_epoch = None
    epoch = 0

    # Aggiunge lo STOPPING CRITERIA basato sulla norma del gradiente, che se non scende di molto
    # oltre un certo numero di epoche (= patience) allora ritorna il risultato
    while gradient_misbehave < self.patience and epoch < self.epochs:
        epoch += 1
        epoch_results = self._run_epoch(input_matrix, d_matrix, n_patterns)

        self.mee_error_history.append(epoch_results["mee"])
        self.mse_error_history.append(epoch_results["mse"])
        prev_gradient_norm_epoch, gradient_misbehave = self._check_patience(gradient_misbehave,
                                                                        prev_gradient_norm_epoch,
                                                                            epoch_results["grad_norm"],
                                                                            n_patterns)
    print("Early stopping triggerato dopo ",epoch," epochs")

    return epoch_results["mee"]
