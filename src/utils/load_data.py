import pandas as pd

# carica i dati necessari al primo input layer e output layer, prende come argomento il path del file TR

def load_data(path) -> tuple[pd.DataFrame, pd.DataFrame] :

    df_input_layer = pd.read_csv(path, header=None, skiprows=7, usecols=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    df_output_layer = pd.read_csv(path, header=None, skiprows=7, usecols=[13, 14, 15, 16])

    #print(df_input_layer, df_output_layer)

    return df_input_layer, df_output_layer


def load_monks_data(path) -> tuple[pd.DataFrame, pd.Series]:
    data = pd.read_csv(path, delimiter=' ', header=None, usecols=range(7))

    X_raw = data.iloc[:, :-1]
    y = data.iloc[:, -1]  # 0/1 target

    X_raw.columns = [f'feature_{i}' for i in range(1, 7)]
    y.name = 'target'

    X = pd.get_dummies(X_raw, columns=X_raw.columns)

    X = X.astype(float)
    y = y.astype(float)

    return X, y



