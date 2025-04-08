from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from .utils import cargar_datos, evaluar_modelo

def entrenar_lr():
    X_train, y_train, X_test, y_test = cargar_datos()
    
    # Los hiperparámetros para optimizar
    parametros = {
        'C': [0.1, 1, 10],
        'solver': ['lbfgs', 'saga']
    }
    
    # Se busca los mejores parámetros
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        parametros,
        cv=3,  # Se hace validacion cruzada
        n_jobs=-1  # Usando todos los nucleos de la cpu
    )
    grid.fit(X_train, y_train)
    
    # Mejor modelo
    mejor_modelo = grid.best_estimator_
    metricas = evaluar_modelo(mejor_modelo, X_test, y_test)
    
    return mejor_modelo, metricas