from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
INPUT_CSV = Path("data/processed/features/segmentation_features.csv")
OUTPUT_ROOT = Path("data/feature_matrix_variants")

RANDOM_STATE = 42
VALIDATION_SIZE = 0.20

WARNING_PRIORITY = [
    "warning_empty_mask",
    "warning_too_large_mask",
    "warning_tissue_almost_empty",
    "warning_too_many_components",
    "warning_selected_empty_but_raw_not_empty",
    "warning_central_fallback_used",
]

WARNING_DESCRIPTIONS = {
    "warning_empty_mask": "sin cromatina segmentada",
    "warning_too_large_mask": "máscara de cromatina demasiado grande",
    "warning_tissue_almost_empty": "tejido casi vacío",
    "warning_too_many_components": "demasiados componentes/células segmentadas",
    "warning_selected_empty_but_raw_not_empty": "selección final vacía aunque la máscara cruda no estaba vacía",
    "warning_central_fallback_used": "selección central realizada por fallback",
}

METADATA_AND_LABEL_COLUMNS = {
    "image_id",
    "source_csv_image_id",
    "image_path",
    "final_label",
    "numeric_label",
    "numeric_label_for_ml",
    "processing_status",
    "filename",
    "coordinateX",
    "coordinateY",
    "Tumor",
    "Scanner",
    "Origin",
    "Species",
}

LEGACY_CHROMATIN_ALIASES = {
    "number_of_components": "chromatin_number_of_components",
    "largest_component_area": "chromatin_largest_component_area",
    "largest_component_ratio": "chromatin_largest_component_ratio",
    "mean_component_area": "chromatin_mean_component_area",
    "std_component_area": "chromatin_std_component_area",
    "component_density_over_cell": "chromatin_component_density",
    "granularity_index": "chromatin_granularity_index",
    "chromatin_skeleton_length_ratio_over_cell": "chromatin_skeleton_length_ratio",
    "total_perimeter": "chromatin_total_perimeter",
    "compactness": "chromatin_compactness",
    "centroid_x": "chromatin_centroid_x",
    "centroid_y": "chromatin_centroid_y",
}

VARIANTS = {
    "rgb_cell_chromatin": {
        "color_space": "rgb",
        "title": "RGB para célula y cromatina",
        "description": (
            "Incluye características morfológicas de célula/cromatina y estadísticas "
            "de color RGB calculadas solo en célula y cromatina. Excluye HSV, LAB y "
            "todas las características de color de la imagen completa."
        ),
    },
    "lab_cell_chromatin": {
        "color_space": "lab",
        "title": "LAB para célula y cromatina",
        "description": (
            "Incluye características morfológicas de célula/cromatina y estadísticas "
            "de color LAB calculadas solo en célula y cromatina. Excluye RGB, HSV y "
            "todas las características de color de la imagen completa."
        ),
    },
    "hsv_cell_chromatin": {
        "color_space": "hsv",
        "title": "HSV para célula y cromatina",
        "description": (
            "Incluye características morfológicas de célula/cromatina y estadísticas "
            "de color HSV calculadas solo en célula y cromatina. Excluye RGB, LAB y "
            "todas las características de color de la imagen completa."
        ),
    },
}


def fail(message):
    raise SystemExit(f"ERROR: {message}")


def is_warning_active(value):
    if pd.isna(value):
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t", "active"}


def get_image_column(dataframe):
    if "image_id" in dataframe.columns:
        return "image_id"
    if "source_csv_image_id" in dataframe.columns:
        return "source_csv_image_id"
    fail("No image identifier column was found. Expected 'image_id' or 'source_csv_image_id'.")


def build_numeric_label_column(dataframe):
    if "numeric_label" in dataframe.columns:
        return pd.to_numeric(dataframe["numeric_label"], errors="coerce"), "numeric_label", {}

    if "final_label" in dataframe.columns:
        normalized = dataframe["final_label"].astype(str).str.strip().str.upper()
        mapping = {
            "NMF": 0,
            "AMF": 1,
            "NORMAL": 0,
            "ATYPICAL": 1,
            "ATYPICAL MITOSIS": 1,
            "NORMAL MITOSIS": 0,
        }
        used_mapping = {key: value for key, value in mapping.items() if key in set(normalized.unique())}
        return normalized.map(mapping), "final_label", used_mapping

    if len(dataframe.columns) < 2:
        fail("No 'numeric_label' or 'final_label' column exists, and the CSV does not have a second column to use as label.")

    fallback_column = dataframe.columns[1]
    numeric = pd.to_numeric(dataframe[fallback_column], errors="coerce")
    if numeric.notna().any():
        return numeric, fallback_column, {}

    normalized = dataframe[fallback_column].astype(str).str.strip().str.upper()
    unique_labels = sorted(normalized[normalized != ""].unique().tolist())
    if len(unique_labels) < 2:
        fail("The fallback label column could not be mapped into at least two numeric classes.")
    mapping = {label: index for index, label in enumerate(unique_labels)}
    return normalized.map(mapping), fallback_column, mapping


def convert_labels_to_numpy(label_series):
    values = label_series.to_numpy(dtype=float)
    if np.allclose(values, np.round(values)):
        return values.astype(int)
    return values


def get_warning_columns(dataframe):
    priority = [column for column in WARNING_PRIORITY if column in dataframe.columns]
    extra = [column for column in dataframe.columns if column.startswith("warning_") and column not in priority]
    return priority + sorted(extra)


def get_first_active_warning(row, warning_columns):
    for column in warning_columns:
        if is_warning_active(row[column]):
            return column
    return np.nan


def build_warning_reports(dataframe, image_column, warning_columns, first_warning_column):
    summary_rows = []
    detail_rows = []
    for column in warning_columns:
        reason = WARNING_DESCRIPTIONS.get(column, column.replace("warning_", "").replace("_", " "))
        discarded_images = dataframe.loc[first_warning_column == column, image_column].astype(str).tolist()
        summary_rows.append({
            "numero_de_descartadas": len(discarded_images),
            "motivo_de_descarte": reason,
            "imagenes_descartadas": "; ".join(discarded_images),
        })
        for image_name in discarded_images:
            detail_rows.append({
                "image_id": image_name,
                "discard_reason": reason,
                "technical_warning_name": column,
            })
    return (
        pd.DataFrame(summary_rows, columns=["numero_de_descartadas", "motivo_de_descarte", "imagenes_descartadas"]),
        pd.DataFrame(detail_rows, columns=["image_id", "discard_reason", "technical_warning_name"]),
    )


def is_numeric_column(dataframe, column):
    converted = pd.to_numeric(dataframe[column], errors="coerce")
    non_empty = dataframe[column].notna() & (dataframe[column].astype(str).str.strip() != "")
    return bool((converted.notna() | ~non_empty).all())


def is_selected_color_column(column, selected_space):
    selected_token = f"_{selected_space}_"
    color_tokens = ["_rgb_", "_hsv_", "_lab_"]
    if selected_token in column:
        return True
    if any(token in column for token in color_tokens):
        return False
    return None


def select_variant_feature_columns(dataframe, selected_space):
    feature_columns = []
    for column in dataframe.columns:
        if column in METADATA_AND_LABEL_COLUMNS or column.startswith("warning_"):
            continue
        if column.startswith("image_"):
            continue
        if column == "tissue_area":
            continue
        if not is_numeric_column(dataframe, column):
            continue

        color_decision = is_selected_color_column(column, selected_space)
        if color_decision is True:
            if column.startswith("cell_") or column.startswith("chromatin_"):
                feature_columns.append(column)
            continue
        if color_decision is False:
            continue

        if column.startswith("cell_") or column.startswith("chromatin_"):
            feature_columns.append(column)
            continue

        if column in LEGACY_CHROMATIN_ALIASES:
            preferred = LEGACY_CHROMATIN_ALIASES[column]
            if preferred not in dataframe.columns:
                feature_columns.append(column)

    feature_columns = list(dict.fromkeys(feature_columns))
    if not feature_columns:
        fail(f"No feature columns were selected for color space '{selected_space}'.")
    return feature_columns


def build_common_valid_mask(dataframe, numeric_labels, first_warning, all_feature_columns):
    if "processing_status" in dataframe.columns:
        success = dataframe["processing_status"].astype(str).str.strip().str.lower() == "success"
    else:
        success = pd.Series(True, index=dataframe.index)
    warning_free = first_warning.isna()
    valid_label = numeric_labels.notna()
    feature_dataframe = dataframe[all_feature_columns].apply(pd.to_numeric, errors="coerce")
    valid_features = feature_dataframe.notna().all(axis=1)
    valid_mask = success & warning_free & valid_label & valid_features
    if valid_mask.sum() == 0:
        fail("No valid images remain after filtering status, warnings, labels and selected numeric features.")
    return valid_mask


def can_use_stratified_split(y_values):
    unique_labels, counts = np.unique(y_values, return_counts=True)
    if len(unique_labels) < 2 or counts.min() < 2:
        return False
    validation_count = int(np.ceil(len(y_values) * VALIDATION_SIZE))
    return validation_count >= len(unique_labels) and (len(y_values) - validation_count) >= len(unique_labels)


def make_train_validation_split(y_values):
    indices = np.arange(len(y_values))
    if len(y_values) < 2:
        fail("At least two valid images are required to create train/validation matrices.")
    if can_use_stratified_split(y_values):
        train_idx, val_idx = train_test_split(
            indices,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            shuffle=True,
            stratify=y_values,
        )
        return train_idx, val_idx, "stratified split by class"
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
        stratify=None,
    )
    return train_idx, val_idx, "random split; stratification was not possible"


def create_label_mapping(used_label_mapping, valid_dataset):
    if used_label_mapping:
        return pd.DataFrame([
            {"label_name": key, "numeric_label": value}
            for key, value in sorted(used_label_mapping.items(), key=lambda item: item[1])
        ])
    if "final_label" in valid_dataset.columns:
        rows = []
        grouped = valid_dataset.groupby("numeric_label")["final_label"].unique()
        for numeric_value, labels in grouped.items():
            rows.append({"label_name": "; ".join(sorted(str(label) for label in labels)), "numeric_label": numeric_value})
        return pd.DataFrame(rows)
    labels = sorted(valid_dataset["numeric_label"].unique())
    return pd.DataFrame({"label_name": labels, "numeric_label": labels})


def save_readme(output_dir, variant_config, feature_columns, X_shape, X_training_shape, X_validation_shape):
    feature_list = "\n".join([f"- `{name}`" for name in feature_columns])
    text = f"""# {variant_config['title']}

{variant_config['description']}

## Fuente

```text
data/processed/features/segmentation_features.csv
```

## Criterio de selección

Este subconjunto elimina toda la información de color de la imagen completa (`image_*`) y conserva únicamente variables asociadas a célula/región nuclear central y cromatina.

También conserva características morfológicas y proporciones de célula/cromatina que no dependen directamente de un espacio de color.

## Espacio de color incluido

```text
{variant_config['color_space'].upper()}
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
X: {X_shape[0]} imágenes x {X_shape[1]} características
X_training: {X_training_shape[0]} imágenes x {X_training_shape[1]} características
X_validation: {X_validation_shape[0]} imágenes x {X_validation_shape[1]} características
```

## Características usadas

{feature_list}
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def save_variant_dataset(dataframe, valid_mask, numeric_labels, image_column, used_label_mapping, feature_columns, variant_name, variant_config, training_indices, validation_indices, warning_summary, warning_detail):
    output_dir = OUTPUT_ROOT / variant_name
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_features = dataframe.loc[valid_mask, feature_columns].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    valid_labels = numeric_labels.loc[valid_mask].reset_index(drop=True)
    valid_image_names = dataframe.loc[valid_mask, image_column].astype(str).to_numpy()

    X = valid_features.to_numpy(dtype=float)
    y = convert_labels_to_numpy(valid_labels)

    X_training = X[training_indices]
    y_training = y[training_indices]
    training_image_names = valid_image_names[training_indices]
    X_validation = X[validation_indices]
    y_validation = y[validation_indices]
    validation_image_names = valid_image_names[validation_indices]

    metadata_parts = [pd.Series(valid_image_names, name="image_id"), pd.Series(y, name="numeric_label")]
    if "final_label" in dataframe.columns:
        metadata_parts.append(dataframe.loc[valid_mask, "final_label"].reset_index(drop=True).rename("final_label"))
    valid_metadata = pd.concat(metadata_parts, axis=1)
    valid_dataset = pd.concat([valid_metadata, valid_features], axis=1)

    training_dataset = valid_dataset.iloc[training_indices].reset_index(drop=True)
    validation_dataset = valid_dataset.iloc[validation_indices].reset_index(drop=True)

    np.save(output_dir / "X.npy", X)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "valid_image_names.npy", valid_image_names)
    np.save(output_dir / "X_training.npy", X_training)
    np.save(output_dir / "y_training.npy", y_training)
    np.save(output_dir / "training_image_names.npy", training_image_names)
    np.save(output_dir / "X_validation.npy", X_validation)
    np.save(output_dir / "y_validation.npy", y_validation)
    np.save(output_dir / "validation_image_names.npy", validation_image_names)

    pd.DataFrame({"feature_name": feature_columns}).to_csv(output_dir / "feature_names.csv", index=False)
    valid_dataset.to_csv(output_dir / "ml_dataset.csv", index=False)
    valid_dataset.to_csv(output_dir / "valid_images.csv", index=False)
    training_dataset.to_csv(output_dir / "training_dataset.csv", index=False)
    validation_dataset.to_csv(output_dir / "validation_dataset.csv", index=False)
    create_label_mapping(used_label_mapping, valid_dataset).to_csv(output_dir / "label_mapping.csv", index=False)
    warning_summary.to_csv(output_dir / "discarded_by_warning.csv", index=False)
    warning_detail.to_csv(output_dir / "discarded_images_detail.csv", index=False)

    save_readme(output_dir, variant_config, feature_columns, X.shape, X_training.shape, X_validation.shape)

    return {
        "variant_name": variant_name,
        "output_dir": str(output_dir),
        "features": len(feature_columns),
        "valid_images": len(valid_dataset),
        "training_images": len(training_dataset),
        "validation_images": len(validation_dataset),
    }


def save_root_readme(summary_rows, split_method):
    lines = [
        "# Variantes de matrices de características",
        "",
        "Esta carpeta contiene subconjuntos de matrices para comparar el rendimiento de modelos usando diferentes espacios de color.",
        "",
        "Todas las variantes se construyen desde:",
        "",
        "```text",
        "data/processed/features/segmentation_features.csv",
        "```",
        "",
        "Todas las variantes:",
        "",
        "- eliminan imágenes con warnings activos;",
        "- eliminan columnas de identificación, texto, etiquetas y metadata;",
        "- eliminan toda la información de color de la imagen completa (`image_*`);",
        "- conservan características de célula/región nuclear central y cromatina;",
        "- usan la misma división entrenamiento/validación para que la comparación sea justa;",
        f"- usan `{split_method}`;",
        "- guardan arreglos `.npy` listos para entrenar modelos.",
        "",
        "## Subcarpetas",
        "",
    ]
    for row in summary_rows:
        lines.extend([
            f"### `{row['variant_name']}/`",
            "",
            f"- Imágenes válidas: {row['valid_images']}",
            f"- Características: {row['features']}",
            f"- Entrenamiento: {row['training_images']}",
            f"- Validación: {row['validation_images']}",
            "",
        ])
    (OUTPUT_ROOT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    if not INPUT_CSV.exists():
        fail(f"Input CSV not found: {INPUT_CSV}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(INPUT_CSV)
    if dataframe.empty:
        fail("The input CSV is empty.")

    image_column = get_image_column(dataframe)
    numeric_labels, label_column, used_label_mapping = build_numeric_label_column(dataframe)

    warning_columns = get_warning_columns(dataframe)
    if warning_columns:
        first_warning = dataframe.apply(lambda row: get_first_active_warning(row, warning_columns), axis=1)
    else:
        first_warning = pd.Series(np.nan, index=dataframe.index)

    warning_summary, warning_detail = build_warning_reports(dataframe, image_column, warning_columns, first_warning)
    warning_summary.to_csv(OUTPUT_ROOT / "discarded_by_warning.csv", index=False)
    warning_detail.to_csv(OUTPUT_ROOT / "discarded_images_detail.csv", index=False)

    variant_feature_columns = {
        variant_name: select_variant_feature_columns(dataframe, config["color_space"])
        for variant_name, config in VARIANTS.items()
    }
    all_feature_columns = sorted(set(column for columns in variant_feature_columns.values() for column in columns))

    valid_mask = build_common_valid_mask(dataframe, numeric_labels, first_warning, all_feature_columns)
    y_all_valid = convert_labels_to_numpy(numeric_labels.loc[valid_mask])
    training_indices, validation_indices, split_method = make_train_validation_split(y_all_valid)

    summary_rows = []
    for variant_name, config in VARIANTS.items():
        summary_rows.append(save_variant_dataset(
            dataframe=dataframe,
            valid_mask=valid_mask,
            numeric_labels=numeric_labels,
            image_column=image_column,
            used_label_mapping=used_label_mapping,
            feature_columns=variant_feature_columns[variant_name],
            variant_name=variant_name,
            variant_config=config,
            training_indices=training_indices,
            validation_indices=validation_indices,
            warning_summary=warning_summary,
            warning_detail=warning_detail,
        ))

    save_root_readme(summary_rows, split_method)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_ROOT / "variant_summary.csv", index=False)

    print("\n================ FEATURE MATRIX VARIANTS ================")
    print(f"Input CSV: {INPUT_CSV}")
    print(f"Output folder: {OUTPUT_ROOT}")
    print(f"Valid images used in every variant: {int(valid_mask.sum())}")
    print(f"Split method: {split_method}")
    print("\nVariants created:")
    for row in summary_rows:
        print(f"  - {row['variant_name']}: {row['valid_images']} images, {row['features']} features, {row['training_images']} train, {row['validation_images']} validation")
    print("=========================================================\n")


if __name__ == "__main__":
    main()
