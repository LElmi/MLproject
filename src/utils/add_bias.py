import numpy as np
Array2D = np.ndarray

def add_bias_input_matrix(x: Array2D) -> Array2D:
        """
        Aggiunge la colonna del bias (valori = 1) alla matrice input (patterns, n_units)

        Args:
            x: Matrice di input (patterns, n_units)
        
        Returns:
            Matrice con bias aggiunto (patterns, n_units + 1)
        """
        bias_column = np.ones((x.shape[0], 1)) # Vettore lungo quanto i pattern
        return np.hstack((x, bias_column))