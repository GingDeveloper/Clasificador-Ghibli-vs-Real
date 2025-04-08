# Importar las bibliotecas
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

RUTA_REAL = "dataset/trainA"          # carpeta de imagenes reales
RUTA_GHIBLI = "dataset/trainB_ghibli" # carpeta de imagenes ghibli
TAMAÑO_IMAGEN = (64, 64)              # Tamaño de redimension a 64x64
TEST_SIZE = 0.2                       # 20% para prueba
SEMILLA = 42                          # Semilla para reproducibilidad

# Funcion para cargar imágenes
def cargar_imagenes(ruta_carpeta, etiqueta):
    X = []
    for archivo in os.listdir(ruta_carpeta):
        try:
            # Lee las imagen en color RGB osea 3 canales
            img = cv2.imread(os.path.join(ruta_carpeta, archivo))
            if img is not None:
                # Redimensiona y normaliza a píxeles [0, 1]
                img = cv2.resize(img, TAMAÑO_IMAGEN)
                img = img / 255.0
                X.append(img)
        except Exception as e:
            print(f"Error al cargar {archivo}: {e}")
    
    y = np.full(len(X), etiqueta)  # Aqui crea las etiquetas
    return np.array(X), y


# Cargar las imágenes reales con la etiqueta 0
X_real, y_real = cargar_imagenes(RUTA_REAL, 0)

# Cargar las imágenes reales con la etiqueta 1
X_ghibli, y_ghibli = cargar_imagenes(RUTA_GHIBLI, 1)

# Combiancion de ambos
X = np.concatenate([X_real, X_ghibli], axis=0)
y = np.concatenate([y_real, y_ghibli], axis=0)


n_muestras = X.shape[0]
X = X.reshape(n_muestras, -1)  # n_muestras, 64*64*3

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=SEMILLA,
    stratify=y  
)

# Salidas en pantalla para verificar resultados
print("\n=== Resumen del preprocesamiento ===")
print(f"- Imágenes reales cargadas: {len(X_real)}")
print(f"- Imágenes Ghibli cargadas: {len(X_ghibli)}")
print(f"- Forma de X_train: {X_train.shape}")
print(f"- Forma de X_test: {X_test.shape}")
print(f"- Ejemplo de etiquetas en y_train: {y_train[:5]}")

# Guardado de datos preprocesados
os.makedirs("data", exist_ok=True)
np.save("data/X_train.npy", X_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_test.npy", y_test)

print("\nPreprocesamiento listo, los datos se guardaron en .npy en la carpeta data")