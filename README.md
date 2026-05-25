# Clasificación de figuras mitóticas normales y atípicas

Se procesan imágenes histológicas de figuras mitóticas para extraer características cuantitativas asociadas a la imagen completa, la región celular/nuclear central y la cromatina, con el fin de usar estas variables como entrada para modelos clásicos de Machine Learning como SVM, Random Forest, KNN y regresión logística.

---

## Estructura del repositorio

```text
script/
├── prepare_midog_track2_data.py
├── build_ml_dataset.py
└── data/
    ├── raw/
    ├── removed/
    └── processed/
        ├── color_spaces/
        ├── features/
        ├── logs/
        ├── masks/
        ├── metadata/
        ├── normalized_rgb/
        ├── overlays/
        └── ml_ready/
```

---

## Entradas principales

```text
data/raw/
```

Contiene las imágenes histológicas originales en formato PNG y el archivo CSV original con las etiquetas y la metadata del conjunto de datos.

```text
data/processed/features/segmentation_features.csv
```

Contiene la tabla de características extraídas por imagen. Esta tabla es la entrada principal para construir las matrices de Machine Learning.

---

## Salidas principales

```text
data/processed/ml_ready/
```

Contiene las matrices, vectores, nombres de imágenes, nombres de características y reportes necesarios para entrenar y evaluar modelos de Machine Learning basados en variables tabulares.

---

## Scripts principales

### prepare_midog_track2_data.py

Script encargado de leer las imágenes originales y generar las características cuantitativas iniciales.

#### Entradas

```text
data/raw/
```

El script usa las imágenes PNG y el CSV original de etiquetas.

#### Procesamiento general

El script realiza las siguientes operaciones:

- lectura de imágenes RGB;
- lectura del CSV original de etiquetas;
- normalización de imágenes;
- conversión a espacios de color HSV y LAB;
- generación de máscaras de tejido;
- generación de máscaras de célula o región nuclear central;
- generación de máscaras de cromatina;
- generación de imágenes intermedias de revisión visual;
- extracción de características numéricas por imagen.

#### Salida principal

```text
data/processed/features/segmentation_features.csv
```

---

### build_ml_dataset.py

Script encargado de convertir `segmentation_features.csv` en matrices numéricas listas para Machine Learning.

#### Entrada

```text
data/processed/features/segmentation_features.csv
```

#### Salidas

```text
data/processed/ml_ready/X.npy
data/processed/ml_ready/y.npy
data/processed/ml_ready/valid_image_names.npy

data/processed/ml_ready/X_training.npy
data/processed/ml_ready/y_training.npy
data/processed/ml_ready/training_image_names.npy

data/processed/ml_ready/X_validation.npy
data/processed/ml_ready/y_validation.npy
data/processed/ml_ready/validation_image_names.npy

data/processed/ml_ready/feature_names.csv
data/processed/ml_ready/ml_dataset.csv
data/processed/ml_ready/valid_images.csv
data/processed/ml_ready/training_dataset.csv
data/processed/ml_ready/validation_dataset.csv
data/processed/ml_ready/label_mapping.csv
data/processed/ml_ready/discarded_by_warning.csv
data/processed/ml_ready/discarded_images_detail.csv
```

#### Ejecución

```bash
python build_ml_dataset.py
```

#### Revisión de archivos generados

```bash
ls data/processed/ml_ready/
```

---

## Carpeta `data/processed/features/`

Esta carpeta contiene la tabla principal de características:

```text
data/processed/features/segmentation_features.csv
```

Cada fila de `segmentation_features.csv` corresponde a una imagen procesada. Cada columna corresponde a una etiqueta, una variable de control de calidad o una característica numérica extraída de la imagen.

---

### Identificación, etiqueta y estado de procesamiento

```text
source_csv_image_id
final_label
processing_status
```

- `source_csv_image_id`: identificador o nombre de la imagen según el CSV original.
- `final_label`: etiqueta final de la imagen.
- `processing_status`: estado del procesamiento de la imagen.

Las etiquetas usadas para clasificación son:

```text
NMF → figura mitótica normal
AMF → figura mitótica atípica
```

---

### Áreas y relaciones principales

```text
image_area
tissue_area
cell_area
chromatin_area
cell_area_ratio_over_tissue
cell_area_ratio_over_image
chromatin_area_ratio_over_tissue
chromatin_area_ratio_over_cell
```

Estas variables describen el tamaño de la imagen, el área de tejido detectada, el área de la región celular/nuclear central, el área de cromatina segmentada y las relaciones proporcionales entre estas regiones.

---

### Características morfológicas generales de cromatina

```text
number_of_components
largest_component_area
largest_component_ratio
mean_component_area
std_component_area
component_density_over_cell
granularity_index
chromatin_skeleton_length
chromatin_skeleton_length_ratio_over_cell
total_perimeter
compactness
centroid_x
centroid_y
```

Estas columnas resumen la organización espacial de la cromatina segmentada. Incluyen número de componentes, tamaño del componente principal, tamaño promedio de componentes, dispersión de áreas, densidad de componentes dentro de la célula, granularidad, longitud del esqueleto, perímetro, compacidad y posición del centroide.

---

### Características cromáticas de la imagen completa

#### RGB

```text
image_mean_rgb_r
image_std_rgb_r
image_mean_rgb_g
image_std_rgb_g
image_mean_rgb_b
image_std_rgb_b
```

#### HSV

```text
image_mean_hsv_h
image_std_hsv_h
image_mean_hsv_s
image_std_hsv_s
image_mean_hsv_v
image_std_hsv_v
```

#### LAB

```text
image_mean_lab_l
image_std_lab_l
image_mean_lab_a
image_std_lab_a
image_mean_lab_b
image_std_lab_b
```

Estas variables describen la media y la desviación estándar de los canales RGB, HSV y LAB calculadas sobre la imagen completa.

---

### Características cromáticas de la región celular/nuclear central

#### RGB

```text
cell_mean_rgb_r
cell_std_rgb_r
cell_mean_rgb_g
cell_std_rgb_g
cell_mean_rgb_b
cell_std_rgb_b
```

#### HSV

```text
cell_mean_hsv_h
cell_std_hsv_h
cell_mean_hsv_s
cell_std_hsv_s
cell_mean_hsv_v
cell_std_hsv_v
```

#### LAB

```text
cell_mean_lab_l
cell_std_lab_l
cell_mean_lab_a
cell_std_lab_a
cell_mean_lab_b
cell_std_lab_b
```

Estas variables describen la media y la desviación estándar de los canales RGB, HSV y LAB calculadas únicamente dentro de la región celular/nuclear central.

---

### Características cromáticas de la cromatina

#### RGB

```text
chromatin_mean_rgb_r
chromatin_std_rgb_r
chromatin_mean_rgb_g
chromatin_std_rgb_g
chromatin_mean_rgb_b
chromatin_std_rgb_b
```

#### HSV

```text
chromatin_mean_hsv_h
chromatin_std_hsv_h
chromatin_mean_hsv_s
chromatin_std_hsv_s
chromatin_mean_hsv_v
chromatin_std_hsv_v
```

#### LAB

```text
chromatin_mean_lab_l
chromatin_std_lab_l
chromatin_mean_lab_a
chromatin_std_lab_a
chromatin_mean_lab_b
chromatin_std_lab_b
```

Estas variables describen la media y la desviación estándar de los canales RGB, HSV y LAB calculadas únicamente dentro de la máscara de cromatina.

---

### Características morfológicas de la región celular/nuclear central

```text
cell_number_of_components
cell_largest_component_area
cell_largest_component_ratio
cell_mean_component_area
cell_std_component_area
cell_component_density
cell_granularity_index
cell_skeleton_length
cell_skeleton_length_ratio
cell_total_perimeter
cell_compactness
cell_centroid_x
cell_centroid_y
```

Estas columnas describen la estructura de la región celular/nuclear central. Incluyen número de componentes, área del componente principal, proporción del componente principal, área promedio de componentes, desviación estándar de áreas, densidad de componentes, granularidad, longitud de esqueleto, perímetro total, compacidad y coordenadas del centroide.

---

### Características morfológicas de la cromatina

```text
chromatin_number_of_components
chromatin_largest_component_area
chromatin_largest_component_ratio
chromatin_mean_component_area
chromatin_std_component_area
chromatin_component_density
chromatin_granularity_index
chromatin_skeleton_length_ratio
chromatin_total_perimeter
chromatin_compactness
chromatin_centroid_x
chromatin_centroid_y
```

Estas columnas describen la estructura de la cromatina segmentada. Incluyen número de componentes de cromatina, área del componente principal, proporción del componente principal, área promedio de componentes, desviación estándar de áreas, densidad de componentes, granularidad, relación de longitud de esqueleto, perímetro total, compacidad y coordenadas del centroide.

---

### Warnings de control de calidad

```text
warning_empty_mask
warning_too_large_mask
warning_tissue_almost_empty
warning_too_many_components
warning_selected_empty_but_raw_not_empty
warning_central_fallback_used
```

Estas columnas indican condiciones de procesamiento que pueden afectar la calidad de una imagen o de sus características extraídas. `build_ml_dataset.py` usa estos warnings para filtrar imágenes antes de construir las matrices finales.

---

## Carpeta `data/processed/ml_ready/`

Esta carpeta contiene los archivos finales para construir, entrenar y evaluar modelos de Machine Learning tabular.

---

### Estructura de `X`

`X` es una matriz de tamaño:

```text
n_imágenes_válidas × n_características
```

Ejemplo conceptual:

```python
X = [
  [cell_area, chromatin_area, chromatin_area_ratio_over_cell, granularity_index, ...],
  [cell_area, chromatin_area, chromatin_area_ratio_over_cell, granularity_index, ...],
  [cell_area, chromatin_area, chromatin_area_ratio_over_cell, granularity_index, ...],
  ...
]
```

Cada fila de `X` representa una imagen válida.

Cada columna representa una característica numérica extraída durante el procesamiento.

El orden exacto de las columnas se guarda en:

```text
data/processed/ml_ready/feature_names.csv
```

---

### Estructura de `y`

`y` es el vector de etiquetas:

```python
y = [0, 1, 0, 1, ...]
```

Donde:

```text
0 → NMF
1 → AMF
```

Cada posición de `y` corresponde exactamente a la misma fila de `X`.

Ejemplo:

```text
X[0] corresponde a y[0]
X[1] corresponde a y[1]
X[2] corresponde a y[2]
```

---

### Relación con los nombres de imágenes

Existe además un arreglo paralelo:

```text
valid_image_names.npy
```

Este arreglo permite rastrear qué imagen corresponde a cada fila de `X`.

Ejemplo:

```text
valid_image_names[0] corresponde a X[0] y y[0]
valid_image_names[1] corresponde a X[1] y y[1]
```

El nombre de la imagen se conserva para trazabilidad y depuración.

El nombre de la imagen no se incluye dentro de `X` porque no es una característica biológica ni morfológica y puede introducir sesgos en el entrenamiento del modelo.

---

### Archivos principales para entrenamiento y validación

```text
data/processed/ml_ready/X_training.npy
data/processed/ml_ready/y_training.npy
data/processed/ml_ready/X_validation.npy
data/processed/ml_ready/y_validation.npy
data/processed/ml_ready/training_image_names.npy
data/processed/ml_ready/validation_image_names.npy
data/processed/ml_ready/feature_names.csv
```

---

### X_training.npy

Matriz de características para entrenamiento.

Formato:

```text
n_train × n_features
```

Cada fila representa una imagen del conjunto de entrenamiento.

Cada columna representa una característica numérica.

---

### y_training.npy

Etiquetas correspondientes a `X_training.npy`.

Formato:

```text
n_train
```

Relación de índices:

```text
X_training[i] corresponde a y_training[i]
```

---

### training_image_names.npy

Nombres de las imágenes correspondientes a cada fila de `X_training.npy`.

Relación de índices:

```text
X_training[i] corresponde a y_training[i] y training_image_names[i]
```

Ejemplo:

```text
X_training[0] contiene las características numéricas de training_image_names[0].
y_training[0] contiene la etiqueta correspondiente a training_image_names[0].
```

---

### X_validation.npy

Matriz de características para validación o prueba.

Formato:

```text
n_validation × n_features
```

Cada fila representa una imagen del conjunto de validación.

Cada columna representa una característica numérica.

---

### y_validation.npy

Etiquetas correspondientes a `X_validation.npy`.

Formato:

```text
n_validation
```

Relación de índices:

```text
X_validation[i] corresponde a y_validation[i]
```

---

### validation_image_names.npy

Nombres de las imágenes correspondientes a cada fila de `X_validation.npy`.

Relación de índices:

```text
X_validation[i] corresponde a y_validation[i] y validation_image_names[i]
```

Ejemplo:

```text
X_validation[0] contiene las características numéricas de validation_image_names[0].
y_validation[0] contiene la etiqueta correspondiente a validation_image_names[0].
```

---

### feature_names.csv

Archivo CSV con el nombre exacto de cada característica usada como columna en las matrices de características.

El orden de `feature_names.csv` corresponde al orden de columnas de:

```text
X.npy
X_training.npy
X_validation.npy
```

Ejemplo conceptual:

```text
feature_names[0] corresponde a X[:, 0]
feature_names[1] corresponde a X[:, 1]
feature_names[2] corresponde a X[:, 2]
```

---

### X.npy

Matriz completa de características numéricas de todas las imágenes válidas.

Formato:

```text
n_imágenes_válidas × n_features
```

---

### y.npy

Vector completo de etiquetas asociado a `X.npy`.

Formato:

```text
n_imágenes_válidas
```

Relación de índices:

```text
X[i] corresponde a y[i]
```

---

### valid_image_names.npy

Nombres de las imágenes correspondientes a cada fila de `X.npy`.

Relación de índices:

```text
X[i] corresponde a y[i] y valid_image_names[i]
```

---

### ml_dataset.csv

Tabla revisable con las imágenes válidas, sus etiquetas y las características numéricas usadas para construir `X.npy`.

Contiene:

```text
image_id
numeric_label
final_label
características numéricas seleccionadas
```

---

### valid_images.csv

Tabla con las mismas imágenes válidas usadas para construir las matrices finales.

Contiene las imágenes que cumplen simultáneamente:

```text
processing_status = success
etiqueta válida
características numéricas completas
sin warnings activos
```

---

### training_dataset.csv

Tabla revisable correspondiente al conjunto de entrenamiento.

Contiene las mismas filas usadas para construir:

```text
X_training.npy
y_training.npy
training_image_names.npy
```

---

### validation_dataset.csv

Tabla revisable correspondiente al conjunto de validación.

Contiene las mismas filas usadas para construir:

```text
X_validation.npy
y_validation.npy
validation_image_names.npy
```

---

### label_mapping.csv

Tabla de correspondencia entre etiquetas textuales y etiquetas numéricas.

Formato conceptual:

```text
label_name,numeric_label
NMF,0
AMF,1
```

---

### discarded_by_warning.csv

Reporte resumido de imágenes descartadas por warnings activos.

Contiene:

```text
numero_de_descartadas
motivo_de_descarte
imagenes_descartadas
```

---

### discarded_images_detail.csv

Reporte detallado de imágenes descartadas por warnings activos.

Contiene:

```text
image_id
discard_reason
technical_warning_name
```

---

## Filtrado por warnings

`build_ml_dataset.py` elimina las imágenes que tienen warnings activos en:

```text
data/processed/features/segmentation_features.csv
```

Estas imágenes no entran en:

```text
X.npy
y.npy
X_training.npy
y_training.npy
X_validation.npy
y_validation.npy
```

Si una imagen tiene varios warnings activos, se descarta una sola vez usando el primer warning activo según este orden de prioridad:

```text
warning_empty_mask
warning_too_large_mask
warning_tissue_almost_empty
warning_too_many_components
warning_selected_empty_but_raw_not_empty
warning_central_fallback_used
```

Los reportes del filtrado quedan guardados en:

```text
data/processed/ml_ready/discarded_by_warning.csv
data/processed/ml_ready/discarded_images_detail.csv
```

---

## División entrenamiento-validación

`build_ml_dataset.py` divide las imágenes válidas en:

```text
80% entrenamiento
20% validación
```

La división usa:

```text
random_state = 42
```

Cuando la distribución de clases lo permite, la división mantiene la proporción de clases en entrenamiento y validación.

---

## Carpetas auxiliares de `data/processed/`

### normalized_rgb/

Contiene imágenes RGB normalizadas generadas durante el procesamiento.

### color_spaces/

Contiene canales HSV y LAB calculados durante el procesamiento.

### masks/

Contiene máscaras intermedias y finales de tejido, región celular/nuclear central, cromatina y componentes seleccionados.

### overlays/

Contiene imágenes con máscaras superpuestas sobre las imágenes originales.

### metadata/

Contiene reportes de procesamiento, etiquetas procesadas y tablas auxiliares.

### logs/

Contiene reportes de errores o fallos de procesamiento por imagen.
