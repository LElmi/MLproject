
def hold_out_validation(x_i, d, split: int):
    n_total = x_i.shape[0]
    n_keep = int(round(n_total - n_total * split / 100.0))

    tr_input = x_i[:n_keep]
    tr_target = d[:n_keep]

    vl_input = x_i[n_keep:]
    vl_targets = d[n_keep:]
    return tr_input,tr_target,vl_input,vl_targets