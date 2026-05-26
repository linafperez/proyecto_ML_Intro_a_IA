# RGB para célula y cromatina

Incluye características morfológicas de célula/cromatina y estadísticas de color RGB calculadas solo en célula y cromatina. Excluye HSV, LAB y todas las características de color de la imagen completa.

## Fuente

```text
data/processed/features/segmentation_features.csv
```

## Criterio de selección

Este subconjunto elimina toda la información de color de la imagen completa (`image_*`) y conserva únicamente variables asociadas a célula/región nuclear central y cromatina.

También conserva características morfológicas y proporciones de célula/cromatina que no dependen directamente de un espacio de color.

## Espacio de color incluido

```text
RGB
```

## Archivos generados

```text
X.npy
y.npy
valid_image_names.npy
X_training.npy
y_training.npy
training_image_names.npy
X_validation.npy
y_validation.npy
validation_image_names.npy
feature_names.csv
ml_dataset.csv
valid_images.csv
training_dataset.csv
validation_dataset.csv
label_mapping.csv
discarded_by_warning.csv
discarded_images_detail.csv
```

## Relación entre matrices y nombres

```text
X[i] corresponde a y[i] y valid_image_names[i]
X_training[i] corresponde a y_training[i] y training_image_names[i]
X_validation[i] corresponde a y_validation[i] y validation_image_names[i]
```

Los nombres de las imágenes se guardan para trazabilidad, pero no se incluyen dentro de `X`, porque no son características útiles para el modelo.

## Dimensiones

```text
X: 9157 imágenes x 45 características
X_training: 7325 imágenes x 45 características
X_validation: 1832 imágenes x 45 características
```

## Características usadas

- `cell_area`
- `chromatin_area`
- `cell_area_ratio_over_tissue`
- `cell_area_ratio_over_image`
- `chromatin_area_ratio_over_tissue`
- `chromatin_area_ratio_over_cell`
- `chromatin_skeleton_length`
- `chromatin_skeleton_length_ratio_over_cell`
- `cell_mean_rgb_r`
- `cell_std_rgb_r`
- `cell_mean_rgb_g`
- `cell_std_rgb_g`
- `cell_mean_rgb_b`
- `cell_std_rgb_b`
- `chromatin_mean_rgb_r`
- `chromatin_std_rgb_r`
- `chromatin_mean_rgb_g`
- `chromatin_std_rgb_g`
- `chromatin_mean_rgb_b`
- `chromatin_std_rgb_b`
- `cell_number_of_components`
- `cell_largest_component_area`
- `cell_largest_component_ratio`
- `cell_mean_component_area`
- `cell_std_component_area`
- `cell_component_density`
- `cell_granularity_index`
- `cell_skeleton_length`
- `cell_skeleton_length_ratio`
- `cell_total_perimeter`
- `cell_compactness`
- `cell_centroid_x`
- `cell_centroid_y`
- `chromatin_number_of_components`
- `chromatin_largest_component_area`
- `chromatin_largest_component_ratio`
- `chromatin_mean_component_area`
- `chromatin_std_component_area`
- `chromatin_component_density`
- `chromatin_granularity_index`
- `chromatin_skeleton_length_ratio`
- `chromatin_total_perimeter`
- `chromatin_compactness`
- `chromatin_centroid_x`
- `chromatin_centroid_y`
