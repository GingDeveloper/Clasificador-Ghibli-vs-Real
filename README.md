# Clasificador de Imágenes: Ghibli vs Real

*Tarea 1 - Comparación de Modelos de Aprendizaje Supervisado*

---

## Descripción

Este proyecto compara el desempeño de tres modelos de aprendizaje supervisado (**Regresión Logística**, **SVM** y **Random Forest**) para clasificar imágenes reales y estilo Ghibli. El objetivo es determinar qué algoritmo logra mayor precisión al realizar esta tarea.

---

## Dataset

- **Nombre**: Real to Ghibli Image Dataset.

- **Link al repositorio original**: [Dataset](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images).

- **Estructura**:

- `trainA/`: 2500 imágenes reales.

- `trainB_ghibli/`: 2500 imágenes estilo Ghibli.

---

## Requisitos

- **Python**: 3.8 o superior.

---

## Instrucciones de Uso

Como ya estan los datos procesados en la carpeta data solo necesitas:

### 1. Clonar el Repositorio

```bash

git clone https://github.com/tu-usuario/Clasificador-Ghibli-vs-Real.git

cd Clasificador-Ghibli-vs-Real

```

### 2. Descargar el Dataset

1. Descarga el dataset: [Descargar](https://www.kaggle.com/datasets/shubham1921/real-to-ghibli-image-dataset-5k-paired-images).

2. Descomprime el archivo descargado.

3. Coloca la carpeta `dataset/` que contiene las carpetas `trainA` (imágenes reales) y `trainB_ghibli` (imágenes Ghibli) dentro de la carpeta del proyecto.


### 3. Instalación de dependencias

```bash

pip install numpy scikit-learn opencv-python

```

### 4. Ejecuta el script de procesamiento, esto generara una carpeta data con los .npy que se requieren:

```bash

python procesamiento.py

```

### 5. Ejecuta el script de entrenamiento para comparar los tres modelos:

```bash

python entrenamiento.py

```

## Importante Considerar

Al entrenar los modelos, el modleo SVM demorara mucho mas que el resto y dependera del hardware, por lo que se debe esperar unos minutos
