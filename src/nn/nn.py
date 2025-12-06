import numpy as np
from src.training.forward.forward_pass import  forward_all_layers

# Tipi utili per chiarezza
Array2D = np.ndarray
Array1D = np.ndarray

"""
Cose da considerare:

    - tipi dei tensori:
    Impostare il tipo dei tensori in cui caricare i pesi del modello.
    float32 → più preciso, più lento, più memoria
    bfloat16 → meno memoria, più veloce, quasi stessa qualità per i LLM

"""

class NN:
    """
    Classe della rete neurale di default feedforward a 2 hidden layers. 
    In futuro proveremo a renderlo parametrizzabile sugli hidden layers.
    Input Layer -> Hidden Layer 1 -> Hidden Layer 2 -> Output Layer

    pro:
        Ogni istanza della classe NN mantiene in memoria i propri valori legati ai vettori risultati
        e alle matrici dei pesi

    Mantiene in memoria:
        - Le matrici dei pesi
        - Vettore risultato x_k
        - Vettore risultato x_j2
        - Vettore risultato x_j1
    """

    def __init__(self,
                 n_inputs: int,
                 n_hidden1: int = 64,
                 n_hidden2: int = 32,
                 n_outputs: int = 4,
                 learning_rate: float = 0.00000025):
        
        """
        Costruttore della rete neurale 
        ad ora l'unico input del costruttore obbligatorio è n_inputs.
        Da aggiungere iperparam:
            momentum, thikonov, margini inizializzazione dei pesi
            cose non legate alla backpropagation.

        Args:
            n_inputs      : Numero di feature di input
            n_hidden1     : Numero di neuroni nel primo hidden layer
            n_hidden      : Numero di neuroni nel secondo hidden layer
            n_outputs     : Numero di neuroni nell'output layer
            learning_rate : eta
        """

        self.n_inputs = n_inputs + 1   # Aggiunge il bias
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_outputs = n_outputs
        self.learning_rate = learning_rate

        # Aggiunge il bias che deve essere aggiunto come un valore (= 1) in più sul vettore x -> x_0 (= 1) + x_1 + .... + x_n
        input_size = n_inputs + 1

        # Inizializza i pesi, prima implementazione le crea staticamente
        self.w_j1i = np.random.randn(n_inputs, n_hidden1) * np.sqrt(2.0 / input_size)
        self.w_j2j1 = np.random.randn(n_hidden1, n_hidden2) * np.sqrt(2.0 / n_hidden1)
        self.w_kj2 = np.random.randn(n_hidden2, n_outputs) * np.sqrt(2.0 / n_hidden2)
 
        self.x_j1  =  0
        self.x_j2  =  0
        self.x_k   =  0

    def update_weights(self,
                       dk: Array1D,
                       dj2: Array1D,
                       dj1: Array1D,
                       x_pattern: Array1D):
        
        """
        Metodo che aggiorna i pesi della rete usando il gradient descent (da modularizzare?)
        
        Args:
            dk: Delta dell'output layer
            dj2: Delta del secondo hidden layer
            dj1: Delta del primo hidden layer
            x_pattern: Input del pattern
        
        Returns:
            Nulla, le modifiche sono fatte in locale
        """
        
        for kunit in range (self.w_kj2.shape[1]):
            for junit in range (self.w_kj2.shape[0]):

                self.w_kj2[junit, kunit] += self.learning_rate * dk[kunit] * self.x_j2[junit]

        for j2unit in range(self.w_j2j1.shape[1]):
            for j1unit in range(self.w_j2j1.shape[0]):
                self.w_j2j1[j1unit, j2unit] += self.learning_rate * dj2[j2unit] * self.x_j1[j1unit]

        for j1unit in range (self.w_j1i.shape[1]):
            for iunit in range (self.w_j1i.shape[0]):

                self.w_j1i[iunit, j1unit] += self.learning_rate * dj1[j1unit] * x_pattern[iunit]     

    
    def forward(self, x_pattern: Array1D) -> tuple[Array1D, Array1D, Array1D]:
        """
        Forward pass su tutta la rete per un **singolo pattern!**

        Args:
            x_pattern: Vettore di input
        
        Returns:
            x_j1  = Vettore risultato primo hidden layer
            x_j2  = Vettore risultato secondo hidden layer
            x_k   = Vettore risultato output layer
        """

        self.x_k, self.x_j2, self.x_j1 = forward_all_layers(x_pattern, self.w_j1i, self.w_j2j1, self.w_kj2)
        return self.x_k, self.x_j2, self.x_j1
