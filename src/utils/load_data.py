import pandas as pd

# carica i dati necessari al primo input layer e output layer, prende come argomento il path del file TR

def load_data(path) -> tuple[pd.DataFrame, pd.DataFrame] :

    df_input_layer = pd.read_csv(path, header=None, skiprows=7, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    df_output_layer = pd.read_csv(path, header=None, skiprows=7, usecols=[13, 14, 15, 16])

    #print(df_input_layer, df_output_layer)

    return df_input_layer, df_output_layer

