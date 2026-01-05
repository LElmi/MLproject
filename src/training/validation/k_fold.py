import numpy as np

def k_fold(x_i, d, folds, validation_fold):
    x_i_splitted = np.array_split(x_i, folds)
    d_splitted   = np.array_split(d, folds)

    vl_input   = x_i_splitted[validation_fold]
    vl_targets = d_splitted[validation_fold]

    tr_input = np.concatenate(
        [sub for i, sub in enumerate(x_i_splitted) if i != validation_fold]
    )

    tr_targets = np.concatenate(
        [sub for i, sub in enumerate(d_splitted) if i != validation_fold]
    )

    return tr_input, tr_targets, vl_input, vl_targets
