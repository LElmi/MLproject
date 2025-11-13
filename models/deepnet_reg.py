import numpy as np
from models.adam_optimizer import AdamOptimizer
from models.train_model import train
from models.forward_pass import forward
from models.compute_loss import compute_loss

class DeepNetReg:
    def __init__(self, input_dim, h1, h2, h3, output_dim,
                 lr=0.001, lam=1e-4, dropout_rate=0.1):
        self.W1 = np.random.randn(input_dim, h1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, h1))
        self.W2 = np.random.randn(h1, h2) * np.sqrt(2.0 / h1)
        self.b2 = np.zeros((1, h2))
        self.W3 = np.random.randn(h2, h3) * np.sqrt(2.0 / h2)
        self.b3 = np.zeros((1, h3))
        self.W4 = np.random.randn(h3, output_dim) * np.sqrt(2.0 / h3)
        self.b4 = np.zeros((1, output_dim))

        self.lam = lam
        self.dropout_rate = dropout_rate
        self.loss_history = []
        self.optimizer = AdamOptimizer(lr)

    def forward(self, X, training=True):
        return forward(X, self.W1, self.b1, self.W2, self.b2,
                       self.W3, self.b3, self.W4, self.b4,
                       self.dropout_rate, training)

    def compute_loss(self, Y_pred, Y_true):
        return compute_loss(Y_pred, Y_true, self.lam,
                            self.W1, self.W2, self.W3, self.W4)

    def train(self, X, Y, epochs=2000, batch_size=64, verbose=True):
        train(self, X, Y, epochs, batch_size, verbose)
