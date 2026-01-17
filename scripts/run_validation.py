from src.utils.compute_error import *
# from src.utils.visualization import *
import numpy as np
from src.utils.load_model import *

"""def run_validation(weights_filename, architecture_filename, validation_set, validation_d):
    #weights_filename=""
    #architecture_filename=""HL1_input,HL2_HL1,output_HL2,
    w_j1i,w_j2j1, w_kj2,   architecture=load_model(weights_filename,architecture_filename)
    x_k, x_j2, x_j1=forward_all_layers(validation_set,w_j1i, w_j2j1, w_kj2,relu) #x_i: Array1D, w_j1i: Array2D,w_j2j1: Array2D, w_kj2: Array2D, f_act: Callable
    mee=mean_euclidean_error(x_k,validation_d)
    #plot_validation_errors(mee)
    if config.MONK ==True:
        validation_monk(validation_set, validation_d, w_j1i, w_j2j1, w_kj2)
#run_validation("weights_2025-12-22_15-37.txt","architecture.json")
def validation_during_training(validation_set, validation_d, w_j1i, w_j2j1, w_kj2):
    x_k, x_j2, x_j1=forward_all_layers(validation_set,w_j1i, w_j2j1, w_kj2,relu) #x_i: Array1D, w_j1i: Array2D,w_j2j1: Array2D, w_kj2: Array2D, f_act: Callable
    prediction=np.zeros(len(x_k))
    for i in range(x_k.shape[0]):
        if x_k[i] > 0.50:
            prediction[i] = 1.0
        else :
            prediction[i] = 0.0
    mee = mean_euclidean_error(x_k, validation_d)
    mse = mean_squared_error(x_k, validation_d)

    return mee, mse , prediction"""

def accuracy(vl_layer_output, validation_d):
    
    if isinstance(validation_d, np.floating):
        prediction = 0

        correctly_classified = 0
        misclassified = 0
        
        if vl_layer_output > 0.50:
            prediction = 1.0
        else :
            prediction = 0.0

        if prediction == validation_d:
            correctly_classified += 1
        else:
            misclassified += 1


    else: 
        prediction = np.zeros(len(validation_d))
    
        correctly_classified = 0
        misclassified = 0
        for i in range(len(vl_layer_output)):

            if vl_layer_output[i] > 0.50:
                prediction[i] = 1.0
            else :
                prediction[i] = 0.0

            if prediction[i] == validation_d[i]:
                correctly_classified += 1
            else:
                misclassified += 1


    return prediction,correctly_classified,misclassified