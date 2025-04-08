import numpy as np
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA

def cargar_datos():
    # Se cargan datos preprocesados
    X_train = np.load("data/X_train.npy")
    y_train = np.load("data/y_train.npy")
    X_test = np.load("data/X_test.npy")
    y_test = np.load("data/y_test.npy")
    
    # Se aplica PCA a todos por igual
    pca = PCA(n_components=100)  # Se reducen a 100 las características
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    
    return X_train, y_train, X_test, y_test

def evaluar_modelo(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    reporte = classification_report(y_test, y_pred, output_dict=True)
    return {
        "Precisión": reporte["accuracy"],
        "F1-Score (Clase 0)": reporte["0"]["f1-score"],
        "F1-Score (Clase 1)": reporte["1"]["f1-score"]
    }