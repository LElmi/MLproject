import numpy as np

def _normalize_shapes(outputs, targets):
    outputs = np.asarray(outputs)
    targets = np.asarray(targets)
    if outputs.ndim == 1:
        outputs = outputs.reshape(-1, 1)
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    if outputs.shape != targets.shape:
        raise ValueError(f"Shape mismatch: {outputs.shape} vs {targets.shape}")
    return outputs, targets

def mean_euclidean_error(outputs, targets):
    """Calcola MEE su dati (normalizzati o reali)."""
    outputs, targets = _normalize_shapes(outputs, targets)
    errors = np.linalg.norm(outputs - targets, axis=1)
    return np.mean(errors)

def mean_squared_error(outputs, targets):
    """Calcola MSE su dati (normalizzati o reali)."""
    outputs, targets = _normalize_shapes(outputs, targets)
    # Somma degli errori quadratici per pattern (axis=1), poi media sui pattern
    squared_errors = np.sum((outputs - targets) ** 2, axis=1)
    return np.mean(squared_errors)

def mean_euclidean_error_with_denorm(outputs, targets, d_min, d_max):
    """
    Denormalizza outputs e targets usando i range del TARGET, poi calcola MEE.
    """
    outputs, targets = _normalize_shapes(outputs, targets)
    
    # Denormalizzazione vettoriale corretta: Val * Range + Min
    # Usiamo d_max e d_min che sono i vettori di scala del target
    denom = d_max - d_min
    
    outputs_real = outputs * denom + d_min
    targets_real = targets * denom + d_min
    
    errors = np.linalg.norm(outputs_real - targets_real, axis=1)
    return np.mean(errors)

def mean_squared_error_with_denorm(outputs, targets, d_min, d_max):
    """
    Denormalizza outputs e targets usando i range del TARGET, poi calcola MSE.
    """
    outputs, targets = _normalize_shapes(outputs, targets)
    
    denom = d_max - d_min
    
    outputs_real = outputs * denom + d_min
    targets_real = targets * denom + d_min
    
    squared_errors = np.sum((outputs_real - targets_real) ** 2, axis=1)
    return np.mean(squared_errors)