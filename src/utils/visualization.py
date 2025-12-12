import matplotlib.pyplot as plt
import math
import numpy as np

Array1D = np.ndarray


def plot_errors(self, epoche: Array1D, tempo_di_training: float):
        """
        Plottaggio degli errori MEE e MSE sullo stesso grafico (o separati, a seconda dei valori).
        """

        # --- 1. Formattazione del Tempo (in minuti e secondi) ---
        minuti = math.floor(tempo_di_training / 60)
        secondi = tempo_di_training % 60
        
        # Stringa formattata
        tempo_formattato = f"{minuti}m {secondi:.2f}s"
        
        # --- Configurazione Base del Plot ---
        plt.figure(figsize=(12, 6))
        plt.title(f'Andamento dell\'Errore durante il Training (Tempo Totale: {tempo_formattato})')        
        plt.xlabel('Epoca')
        plt.ylabel('Valore dell\'Errore')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # --- Plot MEE ---
        # MEE (Mean Euclidean Error) è spesso chiamato RME (Root Mean square Error) per pattern
        plt.plot(epoche, self.mee_error_history, label='MEE (Root Mean Error)', color='blue', linewidth=2)
        
        # --- Plot MSE (Optional, se le scale sono diverse) ---
        # Se l'MSE ha valori molto diversi dal MEE, plottarlo sulla stessa scala può essere fuorviante.
        # In questo caso, li plottiamo insieme ma si può commentare se la scala è troppo diversa.
        #plt.plot(epoche, self.mse_error_history, label='MSE (Mean Squared Error)', color='red', linestyle='--', alpha=0.7)
        
        # --- Plot Errore Totale (Quello che hai calcolato) ---
        #plt.plot(epoche, self.total_error_array, label='Total Error (Average RMSE per Output)', color='green', linestyle=':', linewidth=1.5)

        # Aggiungi una legenda per identificare le linee
        plt.legend(loc='upper right')
        
        # Limita l'asse Y se ci sono picchi iniziali molto alti che nascondono i dettagli successivi
        # try:
        #    plt.ylim(0, max(self.mee_error_history[10:]) * 1.1)
        # except ValueError:
        #    pass # Non fare nulla se l'array è vuoto o troppo piccolo

        plt.show()