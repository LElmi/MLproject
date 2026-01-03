import numpy as np
from src.training.trainer.forward.forward_pass import  forward_all_layers
from src.activationf.relu import relu

from typing import Callable
# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray

class NN:
    def __init__(self,
                n_inputs: int,
                units_list: list[int],
                n_outputs: int,
                f_act: Callable):
        
        """
        Costruttore della rete neurale 
        Args:
            n_inputs      : Numero di feature di input
            n_hidden1     : Numero di neuroni nel primo hidden layer
            n_hidden      : Numero di neuroni nel secondo hidden layer
            n_outputs     : Numero di neuroni nell'output layer
            learning_rate : eta
        """

        self.n_inputs = n_inputs
        self.units_list = list(units_list)
        self.n_outputs = n_outputs  
        self.f_act = f_act  # Per gli hidden layer
        
        self.weights_matrix_list: list[Array2D] = []
        # Crea la matrice dei pesi 
        self._generate_weights_matrix_list()

        # Crea la matrice di risultati intermedi, all'inizio con valori nulli
        # + 1 per il vettore risultato output
        self.layer_results_list: list[Array1D] = [None] * (len(self.units_list) + 1)


    def update_weights(self, delta_list: list[Array2D], eta):
        """
        Questa funzione accoppia lista delle matrici dei pesi con la lista dei delta che devono avere la stessa size e fa l'aggiornamento
        come nella prima versione, l'eta può essere estratto dalla norma dei gradienti diviso l, come nelle slide del prof. Micheli
        """

        self.weights_matrix_list = [ w + (eta * d) 
                        for (w, d) in zip(self.weights_matrix_list, delta_list) 
        ]

    def forward_network(self, x_pattern: Array1D) -> tuple[Array1D, Array1D, Array1D]:
        """
        Forward pass su tutta la rete per un **singolo pattern!** passato dalla funzione di train
        Ad ora la funzione tiene solo in considerazione della funzione di attivazione per gli hidden layer
        
        Args:
            x_pattern: Vettore di input
        
        Returns:
            x_j1  = Vettore risultato primo hidden layer
            x_j2  = Vettore risultato secondo hidden layer
            x_k   = Vettore risultato output layer
        """
        current_input = x_pattern

        # Enumerate restituisce una coppia con il primo elemento l'indice e il secondo l'oggetto della lista
        for i, weights in enumerate(self.weights_matrix_list):
            
            #print("\n\n\n\self.weights_matrix_list: ", self.weights_matrix_list)

            # weights[0] è il bias, weights[1:] sono i pesi collegati ad altre unità
            net = np.dot(current_input, weights[1:]) + weights[0]
            
            # Verifica se è all'ultimo layer
            is_output_layer = (i == len(self.weights_matrix_list) - 1)

            if is_output_layer:
                # Per l'output lineare in fondo
                output = net 
            else:
                output = self.f_act(net)
            
            self.layer_results_list[i] = output
            current_input = output

            #print("\n\n\n\vl_layer_results list: ", self.layer_results_list)

        return self.layer_results_list


    def _generate_weights_matrix_list(self):
        """
        Lavora con self.units_list: una lista di interi corrispondenti al numero di unità,
        il numero di unità è in base a quanto è lunga questa lista di interi, es:
            - lista vuota, allora la rete ha 0 hidden layer
            - lista con un n: int, allora la rete ha un hidden layer grande n
            - ...
        Ritorna una lista di matrici di pesi
        """
        
        # Es: Input(10) -> Hidden(32) -> Hidden(16) -> Output(1)
        # layer_sizes sarà [10, 32, 16, 1]
        layer_sizes = [self.n_inputs] + self.units_list + [self.n_outputs]

        for i in range(len(layer_sizes) - 1):
            n_in = layer_sizes[i]
            n_out = layer_sizes[i+1] 

            weights = np.random.randn(n_in, n_out) * np.sqrt(2.0 / n_in)
            weights = self._add_bias(weights)

            self.weights_matrix_list.append(weights)

    def _add_bias(self, x: Array2D) -> Array2D:
        """ 
        Aggiunge il bias, in testa alla matrice dei pesi, 
        come vettore di 1 lungo quanti i nodi nel layer di destinazione
        """
        # Vettore di 1
        bias_row = np.ones((1, x.shape[1]))

        return np.concatenate((bias_row, x), axis=0)