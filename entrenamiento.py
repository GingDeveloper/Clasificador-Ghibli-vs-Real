from scripts.modelo_lr import entrenar_lr
from scripts.modelo_svm import entrenar_svm
from scripts.modelo_rf import entrenar_rf

def main():
    resultados = {}
    
    # Entrenamiento de los modelos
    print("=== Entrenando Regresión Logística ===")
    _, metricas_lr = entrenar_lr()
    resultados["Regresión Logística"] = metricas_lr
    
    print("\n=== Entrenando SVM ===")
    _, metricas_svm = entrenar_svm()
    resultados["SVM"] = metricas_svm
    
    print("\n=== Entrenando Random Forest ===")
    _, metricas_rf = entrenar_rf()
    resultados["Random Forest"] = metricas_rf
    
    # Resultados de los modelos
    print("\n=== Comparación Final ===")
    print("Modelo\t\t\tPrecisión\tF1-Score (0)\tF1-Score (1)")
    for nombre, metricas in resultados.items():
        print(f"{nombre.ljust(20)}{metricas['Precisión']:.3f}\t\t{metricas['F1-Score (Clase 0)']:.3f}\t\t{metricas['F1-Score (Clase 1)']:.3f}")

if __name__ == "__main__":
    main()