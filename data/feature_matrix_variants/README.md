# Variantes de matrices de características

Esta carpeta contiene subconjuntos de matrices para comparar el rendimiento de modelos usando diferentes espacios de color.

Todas las variantes se construyen desde:

```text
data/processed/features/segmentation_features.csv
```

Todas las variantes:

- eliminan imágenes con warnings activos;
- eliminan columnas de identificación, texto, etiquetas y metadata;
- eliminan toda la información de color de la imagen completa (`image_*`);
- conservan características de célula/región nuclear central y cromatina;
- usan la misma división entrenamiento/validación para que la comparación sea justa;
- usan `stratified split by class`;
- guardan arreglos `.npy` listos para entrenar modelos.

## Subcarpetas

### `rgb_cell_chromatin/`

- Imágenes válidas: 9157
- Características: 45
- Entrenamiento: 7325
- Validación: 1832

### `lab_cell_chromatin/`

- Imágenes válidas: 9157
- Características: 45
- Entrenamiento: 7325
- Validación: 1832

### `hsv_cell_chromatin/`

- Imágenes válidas: 9157
- Características: 45
- Entrenamiento: 7325
- Validación: 1832
