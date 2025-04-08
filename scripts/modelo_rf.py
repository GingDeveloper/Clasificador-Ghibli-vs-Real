from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from .utils import cargar_datos, evaluar_modelo

def entrenar_rf():
    X_train, y_train, X_test, y_test = cargar_datos()
    
    # Los hiperparámetros para optimizar
    parametros = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None]
    }
    
    # Se busca los mejores parámetros
    grid = GridSearchCV(
        RandomForestClassifier(),
        parametros,
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_train, y_train)
    
    # Mejor modelo
    mejor_modelo = grid.best_estimator_
    metricas = evaluar_modelo(mejor_modelo, X_test, y_test)
    
    return mejor_modelo, metricas