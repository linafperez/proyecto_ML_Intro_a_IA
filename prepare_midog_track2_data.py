#!/usr/bin/env python3
"""
prepare_midog_track2_data.py

Image preparation and chromatin segmentation pipeline for the MIDOG 2025 Track 2
atypical mitotic figure classification dataset.

This script prepares 2D RGB PNG histology patches. It does not train a machine
learning model.
"""

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import itk
import numpy as np
import pandas as pd
from PIL import Image
from skimage import color, measure, morphology
from skimage.filters import threshold_otsu

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# Parses command-line arguments for the MIDOG image preparation pipeline.
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Prepare and segment MIDOG 2025 Track 2 RGB mitotic figure patches."
    )

    parser.add_argument(
        "--images_dir",
        required=True,
        type=str,
        help="Folder containing the input PNG patches.",
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        type=str,
        help="CSV file containing image metadata and majority labels.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        type=str,
        help="Folder where processed outputs will be saved.",
    )
    parser.add_argument(
        "--target_size",
        default=128,
        type=int,
        help="Target square size used to resize input patches if needed.",
    )
    parser.add_argument(
        "--min_component_area",
        default=5,
        type=int,
        help="Minimum connected-component area kept in chromatin masks.",
    )
    parser.add_argument(
        "--central_radius",
        default=45,
        type=float,
        help="Radius in pixels used to select central mitotic components.",
    )
    parser.add_argument(
        "--save_debug_sample",
        default=None,
        type=int,
        help="If provided, saves overlays only for the first N images.",
    )
    parser.add_argument(
        "--max_images",
        default=None,
        type=int,
        help="If provided, processes only the first N images.",
    )

    return parser.parse_args()


# Creates the output directory tree and returns a dictionary with all paths.
def create_output_directories(output_dir):
    output_root = Path(output_dir)  # Root folder where every output will be stored

    output_dirs = {
        "root": output_root,  # Root output folder
        "metadata": output_root / "metadata",  # Folder for label and processing CSV files
        "normalized_rgb": output_root / "normalized_rgb",  # Folder for normalized RGB patches
        "hsv_h": output_root / "color_spaces" / "hsv" / "H",  # Folder for HSV hue channel
        "hsv_s": output_root / "color_spaces" / "hsv" / "S",  # Folder for HSV saturation channel
        "hsv_v": output_root / "color_spaces" / "hsv" / "V",  # Folder for HSV value channel
        "lab_l": output_root / "color_spaces" / "lab" / "L",  # Folder for LAB luminance channel
        "lab_a": output_root / "color_spaces" / "lab" / "A",  # Folder for LAB A channel
        "lab_b": output_root / "color_spaces" / "lab" / "B",  # Folder for LAB B channel
        "tissue_mask": output_root / "masks" / "tissue",  # Folder for tissue masks
        "chromatin_score": output_root / "masks" / "chromatin_score",  # Folder for chromatin score images
        "chromatin_otsu_raw": output_root / "masks" / "chromatin_otsu_raw",  # Folder for raw OTSU masks
        "chromatin_refined": output_root / "masks" / "chromatin_refined",  # Folder for refined masks
        "selected_components": output_root / "masks" / "selected_components",  # Folder for selected masks
        "tissue_overlay": output_root / "overlays" / "tissue_overlay",  # Folder for tissue QC overlays
        "chromatin_overlay": output_root / "overlays" / "chromatin_overlay",  # Folder for chromatin QC overlays
        "selected_overlay": output_root / "overlays" / "selected_components_overlay",  # Folder for selected QC overlays
        "features": output_root / "features",  # Folder for segmentation feature CSV files
        "logs": output_root / "logs",  # Folder for error logs
        "cell_score": output_root / "masks" / "cell_score",
        "central_cell_mask": output_root / "masks" / "central_cell_mask",
        "cell_overlay": output_root / "overlays" / "cell_overlay",
    }

    for directory_path in output_dirs.values():
        directory_path.mkdir(parents=True, exist_ok=True)

    return output_dirs


# Normalizes a class label into the expected NMF or AMF notation.
def normalize_label_value(label_value):
    if pd.isna(label_value):
        return None

    label_text = str(label_value).strip().upper()  # Uppercase label text used for robust matching

    nmf_values = {"NMF", "NORMAL", "NORMAL_MITOSIS", "NORMAL MITOSIS", "0"}
    amf_values = {"AMF", "ATYPICAL", "ATYPICAL_MITOSIS", "ATYPICAL MITOSIS", "1"}

    if label_text in nmf_values:
        return "NMF"
    if label_text in amf_values:
        return "AMF"

    return None


# Computes the final label using only the majority criterion or expert-majority fallback.
def compute_majority_label(row, column_lookup):
    majority_candidates = [
        "majority",
        "majority_vote",
        "majorityvote",
        "final_label",
        "label",
    ]

    for candidate_name in majority_candidates:
        if candidate_name in column_lookup:
            majority_column = column_lookup[candidate_name]  # Actual CSV column storing majority label
            majority_label = normalize_label_value(row.get(majority_column))  # Majority label normalized to NMF or AMF
            if majority_label in {"NMF", "AMF"}:
                return majority_label, ""

    expert_columns = [
        original_column
        for normalized_column, original_column in column_lookup.items()
        if normalized_column.startswith("expert")
    ]

    expert_labels = [
        normalize_label_value(row.get(expert_column))
        for expert_column in expert_columns
    ]
    expert_labels = [
        expert_label
        for expert_label in expert_labels
        if expert_label in {"NMF", "AMF"}
    ]

    if len(expert_labels) == 0:
        return None, "missing_majority_label"

    label_counts = Counter(expert_labels)  # Count of normalized expert labels
    most_common = label_counts.most_common()  # Expert labels sorted by frequency

    if len(most_common) == 1:
        return most_common[0][0], ""

    if most_common[0][1] == most_common[1][1]:
        return None, "ambiguous_expert_vote"

    return most_common[0][0], ""


# Finds the most likely PNG path for a row using image_id or filename metadata.
def find_image_path(row, images_dir, column_lookup):
    identifier_columns = [
        "image_id",
        "filename",
        "file",
        "image",
        "name",
    ]

    candidate_names = []  # Possible image file names inferred from metadata

    for identifier_column in identifier_columns:
        if identifier_column in column_lookup:
            original_column = column_lookup[identifier_column]  # Actual CSV column for an image identifier
            value = row.get(original_column)  # Row value that may contain a file name
            if not pd.isna(value):
                candidate_names.append(str(value).strip())

    for candidate_name in candidate_names:
        candidate_path = Path(candidate_name)  # Candidate path as written in the CSV

        direct_path = images_dir / candidate_path  # Candidate joined with the image root folder
        if direct_path.exists():
            return direct_path

        if candidate_path.suffix == "":
            png_path = images_dir / f"{candidate_name}.png"  # Candidate with PNG suffix
            if png_path.exists():
                return png_path

        stem_matches = list(images_dir.glob(f"{candidate_path.stem}.*"))  # Any file with the same stem
        if stem_matches:
            return stem_matches[0]

    return images_dir / candidate_names[0] if candidate_names else None

# Converts an image identifier into a normalized file-name key.
def normalize_image_key(value):
    if pd.isna(value):
        return None

    text_value = str(value).strip()  # Raw image identifier from CSV or file system

    if text_value == "":
        return None

    image_name = Path(text_value).name  # File name without parent folders

    if Path(image_name).suffix == "":
        image_name = f"{image_name}.png"  # Adds PNG extension when the CSV stores only the numeric ID

    return image_name.lower()


# Sorts image paths numerically when possible and alphabetically otherwise.
def image_sort_key(image_path):
    image_stem = image_path.stem  # File stem used for numeric sorting

    if image_stem.isdigit():
        return (0, int(image_stem))

    return (1, image_path.name.lower())

# Loads the label CSV by matching only existing image files against the raw CSV image_id column.
def load_labels_table(labels_csv, images_dir, output_dirs):
    labels_path = Path(labels_csv)  # Path to the input metadata CSV file
    image_root = Path(images_dir)  # Folder containing the input PNG patches

    labels_table = pd.read_csv(labels_path)  # Original CSV table with metadata and labels
    column_lookup = {
        column_name.strip().lower(): column_name
        for column_name in labels_table.columns
    }  # Mapping from normalized column names to original column names

    if "image_id" not in column_lookup:
        raise ValueError("The CSV must contain an image_id column to match PNG patches correctly.")

    image_id_column = column_lookup["image_id"]  # Actual CSV image_id column name

    csv_rows_by_image_key = {}  # Dictionary matching normalized image names to CSV rows

    for _, row in labels_table.iterrows():
        image_key = normalize_image_key(row.get(image_id_column))  # Normalized image_id key from CSV

        if image_key is not None and image_key not in csv_rows_by_image_key:
            csv_rows_by_image_key[image_key] = row  # Store first matching row for this image_id

    existing_image_paths = sorted(
        [
            image_path
            for image_path in image_root.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
        ],
        key=image_sort_key,
    )  # Real images currently present in data/raw/images

    existing_image_keys = {
        normalize_image_key(image_path.name)
        for image_path in existing_image_paths
    }  # Image keys that actually exist in the image folder

    valid_rows = []  # Rows with existing image files and valid labels
    skipped_rows = []  # Rows skipped due to missing images, missing labels, or ambiguity

    for image_path in existing_image_paths:
        image_key = normalize_image_key(image_path.name)  # Normalized file name from the actual image file

        if image_key not in csv_rows_by_image_key:
            skipped_rows.append({
                "image_id_for_processing": image_path.name,
                "image_path": str(image_path),
                "skip_reason": "image_file_not_found_in_csv",
            })
            continue

        row = csv_rows_by_image_key[image_key]  # Exact metadata row from the raw CSV
        final_label, skip_reason = compute_majority_label(row, column_lookup)  # Majority-based class label

        output_row = row.to_dict()  # Preserve all original CSV metadata
        output_row["source_csv_image_id"] = str(row.get(image_id_column))  # Original image_id from the CSV
        output_row["image_id_for_processing"] = image_path.name  # Real image file name being processed
        output_row["image_path"] = str(image_path)  # Real image path being processed
        output_row["final_label"] = final_label  # Final majority label
        output_row["numeric_label"] = 0 if final_label == "NMF" else 1 if final_label == "AMF" else np.nan

        if final_label not in {"NMF", "AMF"}:
            output_row["skip_reason"] = skip_reason or "invalid_label"
            skipped_rows.append(output_row)
            continue

        valid_rows.append(output_row)

    for _, row in labels_table.iterrows():
        csv_image_key = normalize_image_key(row.get(image_id_column))  # Normalized image key from CSV

        if csv_image_key is not None and csv_image_key not in existing_image_keys:
            skipped_row = row.to_dict()
            skipped_row["source_csv_image_id"] = str(row.get(image_id_column))
            skipped_row["image_id_for_processing"] = str(row.get(image_id_column))
            skipped_row["image_path"] = ""
            skipped_row["skip_reason"] = "csv_image_not_present_in_images_dir"
            skipped_rows.append(skipped_row)

    valid_table = pd.DataFrame(valid_rows)  # Metadata table containing only existing images with valid labels
    skipped_table = pd.DataFrame(skipped_rows)  # Metadata table containing skipped rows

    valid_table.to_csv(output_dirs["metadata"] / "labels_majority.csv", index=False)
    skipped_table.to_csv(output_dirs["metadata"] / "skipped_rows.csv", index=False)

    return valid_table

# Resizes an RGB NumPy image to the target square size when necessary.
def resize_image_if_needed(rgb_array, target_size):
    current_height = rgb_array.shape[0]  # Current image height in pixels
    current_width = rgb_array.shape[1]  # Current image width in pixels

    if current_height == target_size and current_width == target_size:
        return rgb_array.astype(np.float32)

    pil_image = Image.fromarray((np.clip(rgb_array, 0.0, 1.0) * 255).astype(np.uint8))  # PIL image for robust RGB resizing
    resized_image = pil_image.resize((target_size, target_size), resample=Image.BILINEAR)  # Resized image with bilinear interpolation
    resized_array = np.asarray(resized_image).astype(np.float32) / 255.0  # Resized RGB array in [0, 1]

    return resized_array


# Reads an RGB image from disk and converts it to a NumPy float array.
def read_rgb_image(image_path, target_size):
    image_file = Path(image_path)  # Path to the PNG image patch

    with Image.open(image_file) as input_image:
        rgb_image = input_image.convert("RGB")  # RGB image regardless of the original PNG mode
        rgb_array = np.asarray(rgb_image).astype(np.float32) / 255.0  # RGB array scaled to [0, 1]

    rgb_array = resize_image_if_needed(rgb_array, target_size)  # RGB array resized to the requested size
    rgb_array = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)  # Safe RGB array in [0, 1]

    return rgb_array


# Normalizes an RGB image using global percentile normalization while preserving color relationships.
def normalize_rgb_image(rgb_array):
    clipped_rgb = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)  # Input RGB clipped to valid range

    lower_percentile = np.percentile(clipped_rgb, 1)  # Global 1st percentile across all RGB channels
    upper_percentile = np.percentile(clipped_rgb, 99)  # Global 99th percentile across all RGB channels

    if upper_percentile <= lower_percentile:
        return clipped_rgb

    normalized_rgb = (clipped_rgb - lower_percentile) / (upper_percentile - lower_percentile)  # Percentile-normalized RGB
    normalized_rgb = np.clip(normalized_rgb, 0.0, 1.0).astype(np.float32)  # Final normalized RGB in [0, 1]

    return normalized_rgb


# Converts an RGB image to HSV and LAB color spaces.
def convert_color_spaces(rgb_array):
    safe_rgb = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)  # RGB image prepared for color conversion

    hsv_array = color.rgb2hsv(safe_rgb).astype(np.float32)  # HSV representation with channels H, S, V
    lab_array = color.rgb2lab(safe_rgb).astype(np.float32)  # LAB representation with channels L, A, B

    return hsv_array, lab_array


# Converts a 2D NumPy scalar array to an ITK scalar image.
def numpy_to_itk_scalar_image(scalar_array, pixel_type=np.float32):
    array_for_itk = scalar_array.astype(pixel_type)  # Scalar array cast to the requested ITK-compatible type
    itk_image = itk.image_from_array(array_for_itk)  # ITK scalar image created from the NumPy array

    return itk_image


# Converts an ITK scalar image to a NumPy array.
def itk_scalar_image_to_numpy(itk_image):
    numpy_array = itk.array_from_image(itk_image)  # NumPy array extracted from the ITK image

    return numpy_array


# Scales a scalar array to [0, 255] for grayscale PNG saving.
def scale_to_uint8(scalar_array):
    scalar_array = np.asarray(scalar_array)  # Input scalar image as a NumPy array

    if scalar_array.dtype == np.bool_:
        return (scalar_array.astype(np.uint8) * 255)

    finite_array = np.nan_to_num(scalar_array.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)  # Finite float image

    if finite_array.max() <= 1.0 and finite_array.min() >= 0.0:
        scaled_array = finite_array * 255.0  # Image already normalized to [0, 1]
    else:
        min_value = float(finite_array.min())  # Minimum finite intensity
        max_value = float(finite_array.max())  # Maximum finite intensity
        if max_value <= min_value:
            scaled_array = np.zeros_like(finite_array)
        else:
            scaled_array = (finite_array - min_value) / (max_value - min_value) * 255.0

    return np.clip(scaled_array, 0, 255).astype(np.uint8)


# Saves a 2D scalar image as a grayscale PNG using ITK when possible.
def save_grayscale_image(scalar_array, output_path):
    output_file = Path(output_path)  # Destination path for the grayscale PNG
    output_file.parent.mkdir(parents=True, exist_ok=True)

    uint8_array = scale_to_uint8(scalar_array)  # Grayscale image converted to uint8

    try:
        itk_image = numpy_to_itk_scalar_image(uint8_array, np.uint8)  # ITK image used for writing
        itk.imwrite(itk_image, str(output_file))
    except Exception:
        Image.fromarray(uint8_array, mode="L").save(output_file)


# Saves an RGB image as a PNG file.
def save_rgb_image(rgb_array, output_path):
    output_file = Path(output_path)  # Destination path for the RGB PNG
    output_file.parent.mkdir(parents=True, exist_ok=True)

    uint8_rgb = (np.clip(rgb_array, 0.0, 1.0) * 255).astype(np.uint8)  # RGB array converted to uint8
    Image.fromarray(uint8_rgb, mode="RGB").save(output_file)


# Applies ITK binary morphology with a scikit-image fallback for robustness.
def apply_binary_morphology(binary_mask, operation, radius):
    clean_mask = binary_mask.astype(np.uint8)  # Binary mask encoded as 0 and 1
    
    if radius <= 0:
        return clean_mask > 0  # No morphology is applied when radius is zero
    
    radius_value = int(radius)  # Positive morphology radius in pixels

    try:
        ImageType = itk.Image[itk.UC, 2]  # 2D unsigned-char ITK image type
        KernelType = itk.FlatStructuringElement[2]  # 2D flat structuring element type
        kernel = KernelType.Ball(radius_value)  # Disk-like structuring element for morphology
        itk_mask = itk.image_from_array(clean_mask)  # Input mask as an ITK image

        if operation == "opening":
            result_image = itk.binary_morphological_opening_image_filter(
                itk_mask,
                kernel=kernel,
                foreground_value=1,
            )
        elif operation == "closing":
            result_image = itk.binary_morphological_closing_image_filter(
                itk_mask,
                kernel=kernel,
                foreground_value=1,
            )
        else:
            return clean_mask.astype(bool)

        result_array = itk.array_from_image(result_image).astype(np.uint8)  # Morphology result as NumPy
        return result_array > 0

    except Exception:
        footprint = morphology.disk(radius_value)  # Disk footprint used by the fallback implementation
        if operation == "opening":
            return morphology.binary_opening(clean_mask > 0, footprint)
        if operation == "closing":
            return morphology.binary_closing(clean_mask > 0, footprint)
        return clean_mask > 0


# Fills holes in a binary mask using ITK when possible and scikit-image as fallback.
def fill_binary_holes(binary_mask):
    clean_mask = binary_mask.astype(np.uint8)  # Binary mask encoded as 0 and 1

    try:
        itk_mask = itk.image_from_array(clean_mask)  # Input mask as an ITK image
        result_image = itk.binary_fillhole_image_filter(
            itk_mask,
            foreground_value=1,
        )
        result_array = itk.array_from_image(result_image).astype(np.uint8)  # Hole-filled result as NumPy
        return result_array > 0

    except Exception:
        return morphology.remove_small_holes(clean_mask > 0, area_threshold=64)


# Labels connected components using ITK when possible and scikit-image as fallback.
def label_connected_components(binary_mask, min_component_area=0):
    clean_mask = binary_mask.astype(np.uint8)  # Binary mask encoded as 0 and 1

    try:
        itk_mask = itk.image_from_array(clean_mask)  # Input mask as an ITK image
        component_image = itk.connected_component_image_filter(
            itk_mask,
            fully_connected=False,
        )
        relabeled_image = itk.relabel_component_image_filter(
            component_image,
            minimum_object_size=int(max(0, min_component_area)),
        )
        label_array = itk.array_from_image(relabeled_image).astype(np.int32)  # Component labels sorted by size
        return label_array

    except Exception:
        label_array = measure.label(clean_mask > 0, connectivity=1).astype(np.int32)  # Component labels from fallback
        if min_component_area > 0:
            filtered_mask = morphology.remove_small_objects(label_array > 0, min_size=int(min_component_area))
            label_array = measure.label(filtered_mask, connectivity=1).astype(np.int32)
        return label_array


# Refines a binary mask with opening, closing, hole filling, and small-component removal.
def refine_binary_mask(raw_mask, min_component_area=5, opening_radius=1, closing_radius=1, fill_holes=True):
    refined_mask = raw_mask.astype(bool)  # Input binary mask converted to boolean

    refined_mask = apply_binary_morphology(refined_mask, "opening", opening_radius)  # Opening removes isolated pixels
    refined_mask = apply_binary_morphology(refined_mask, "closing", closing_radius)  # Closing connects nearby fragments

    if fill_holes:
        refined_mask = fill_binary_holes(refined_mask)  # Filled binary mask

    label_array = label_connected_components(refined_mask, min_component_area)  # Connected components after morphology
    refined_mask = label_array > 0  # Final mask after removing small components

    return refined_mask.astype(np.uint8)


# Creates a tissue mask while excluding white background and black padding artifacts.
def create_tissue_mask(rgb_array, hsv_array, lab_array):
    red_channel = rgb_array[:, :, 0]  # Normalized RGB red channel
    green_channel = rgb_array[:, :, 1]  # Normalized RGB green channel
    blue_channel = rgb_array[:, :, 2]  # Normalized RGB blue channel

    value_channel = hsv_array[:, :, 2]  # HSV V channel representing brightness
    saturation_channel = hsv_array[:, :, 1]  # HSV S channel representing color saturation
    lab_l_norm = np.clip(lab_array[:, :, 0] / 100.0, 0.0, 1.0)  # LAB L channel normalized to [0, 1]

    white_background_mask = (
        (value_channel > 0.90)
        & (saturation_channel < 0.15)
        & (lab_l_norm > 0.85)
    )  # Bright, weakly saturated pixels are considered white background

    black_padding_mask = (
        (red_channel < 0.08)
        & (green_channel < 0.08)
        & (blue_channel < 0.08)
        & (value_channel < 0.12)
    )  # Very dark RGB pixels are considered black padding/artifact

    invalid_background_mask = white_background_mask | black_padding_mask  # Pixels that should not be tissue

    raw_tissue_mask = (~invalid_background_mask) & (
        (value_channel < 0.90)
        | (saturation_channel > 0.12)
        | (lab_l_norm < 0.85)
    )  # Tissue-like pixels after excluding white and black artifacts

    tissue_mask = refine_binary_mask(
        raw_tissue_mask,
        min_component_area=25,
        opening_radius=1,
        closing_radius=1,
        fill_holes=False,
    )  # Refined tissue mask without filling internal white spaces

    tissue_mask[invalid_background_mask] = 0  # Force white and black artifacts to remain outside tissue

    return tissue_mask.astype(np.uint8)


# Builds a chromatin score image emphasizing dark purple/blue chromatin-like regions.
def build_chromatin_score(rgb_array, hsv_array, lab_array, cell_mask):
    red_channel = rgb_array[:, :, 0]  # Normalized RGB red channel
    green_channel = rgb_array[:, :, 1]  # Normalized RGB green channel
    blue_channel = rgb_array[:, :, 2]  # Normalized RGB blue channel

    value_channel = hsv_array[:, :, 2]  # HSV V channel; lower values indicate darker pixels
    saturation_channel = hsv_array[:, :, 1]  # HSV S channel; higher values indicate stronger staining
    lab_l_norm = np.clip(lab_array[:, :, 0] / 100.0, 0.0, 1.0)  # LAB L channel normalized to [0, 1]

    inverse_value = 1.0 - value_channel  # High values for dark pixels
    inverse_luminance = 1.0 - lab_l_norm  # High values for low-luminance pixels

    purple_blue_excess = np.maximum.reduce([
        blue_channel - green_channel,
        red_channel - green_channel,
        ((red_channel + blue_channel) / 2.0) - green_channel,
    ])  # High values for purple/blue chromatin-like pixels

    purple_blue_excess = np.clip(purple_blue_excess, 0.0, 1.0)  # Restrict chromatic excess to [0, 1]

    raw_score = (
        0.40 * inverse_value
        + 0.30 * inverse_luminance
        + 0.15 * saturation_channel
        + 0.15 * purple_blue_excess
    )  # Combined chromatin score

    raw_score = raw_score * cell_mask.astype(np.float32)  # Restrict chromatin search to the central cell region

    valid_values = raw_score[cell_mask > 0]  # Score values inside central cell mask only

    if valid_values.size == 0 or np.allclose(valid_values.max(), valid_values.min()):
        return np.zeros_like(raw_score, dtype=np.float32)

    min_score = float(np.percentile(valid_values, 1))  # Robust lower bound
    max_score = float(np.percentile(valid_values, 99))  # Robust upper bound

    if max_score <= min_score:
        normalized_score = np.zeros_like(raw_score, dtype=np.float32)
    else:
        normalized_score = (raw_score - min_score) / (max_score - min_score)

    normalized_score = np.clip(normalized_score, 0.0, 1.0).astype(np.float32)  # Final score in [0, 1]
    normalized_score[cell_mask == 0] = 0.0  # Ensure outside-cell pixels remain zero

    return normalized_score

# Builds a broader cellular/nuclear score image for central mitotic-region segmentation.
def build_cell_score(rgb_array, hsv_array, lab_array, tissue_mask):
    value_channel = hsv_array[:, :, 2]  # HSV V channel; low values indicate darker tissue
    saturation_channel = hsv_array[:, :, 1]  # HSV S channel; high values indicate stained tissue
    lab_l_norm = np.clip(lab_array[:, :, 0] / 100.0, 0.0, 1.0)  # LAB luminance normalized to [0, 1]

    inverse_value = 1.0 - value_channel  # Darker regions receive higher score
    inverse_luminance = 1.0 - lab_l_norm  # Low-luminance regions receive higher score

    raw_score = (
        0.45 * inverse_value
        + 0.30 * inverse_luminance
        + 0.25 * saturation_channel
    )  # Broader score for stained cellular/nuclear material

    raw_score = raw_score * tissue_mask.astype(np.float32)  # Restrict score to tissue

    tissue_values = raw_score[tissue_mask > 0]  # Score values inside tissue
    if tissue_values.size == 0 or np.allclose(tissue_values.max(), tissue_values.min()):
        return np.zeros_like(raw_score, dtype=np.float32)

    min_score = float(np.percentile(tissue_values, 1))  # Robust lower intensity
    max_score = float(np.percentile(tissue_values, 99))  # Robust upper intensity

    if max_score <= min_score:
        normalized_score = np.zeros_like(raw_score, dtype=np.float32)
    else:
        normalized_score = (raw_score - min_score) / (max_score - min_score)

    normalized_score = np.clip(normalized_score, 0.0, 1.0).astype(np.float32)
    normalized_score[tissue_mask == 0] = 0.0

    return normalized_score


# Segments chromatin inside the central cell region using Otsu thresholding.
def segment_chromatin_otsu(chromatin_score, cell_mask):
    masked_score = chromatin_score.astype(np.float32) * cell_mask.astype(np.float32)  # Score restricted to cell region
    cell_values = masked_score[cell_mask > 0]  # Score values inside the central cell mask
    positive_values = cell_values[cell_values > 0]  # Positive score values only

    if positive_values.size == 0 or np.allclose(positive_values.max(), positive_values.min()):
        return np.zeros_like(masked_score, dtype=np.uint8)

    otsu_threshold = threshold_otsu(positive_values)  # Otsu threshold computed only inside the cell region

    raw_mask = (masked_score >= otsu_threshold) & (cell_mask > 0)  # High chromatin-score pixels become foreground

    return raw_mask.astype(np.uint8)

# Segments a broader central cellular/nuclear region using Otsu and central component selection.
def segment_central_cell_by_otsu(cell_score, tissue_mask, image_shape, central_radius):
    masked_score = cell_score.astype(np.float32) * tissue_mask.astype(np.float32)  # Score restricted to tissue
    tissue_values = masked_score[tissue_mask > 0]  # Tissue-only score values

    if tissue_values.size == 0 or np.allclose(tissue_values.max(), tissue_values.min()):
        return np.zeros(image_shape, dtype=np.uint8)

    otsu_threshold = threshold_otsu(tissue_values)  # Otsu threshold for broader cell/nuclear region
    percentile_threshold = np.percentile(tissue_values, 55)  # Avoids selecting very weakly stained tissue
    threshold_value = max(otsu_threshold, percentile_threshold)  # Conservative broad threshold

    raw_cell_mask = (masked_score > threshold_value) & (tissue_mask > 0)  # Broad candidate mask

    refined_cell_mask = refine_binary_mask(
        raw_cell_mask,
        min_component_area=80,
        opening_radius=1,
        closing_radius=2,
        fill_holes=True,
    )  # Refined larger cell/nuclear region

    central_cell_mask, _ = select_relevant_components(
        refined_cell_mask,
        image_shape,
        central_radius,
    )  # Keep central component(s)

    return central_cell_mask.astype(np.uint8)


# Selects connected components likely to correspond to the central mitotic figure.
def select_relevant_components(refined_mask, image_shape, central_radius):
    label_array = label_connected_components(refined_mask, min_component_area=1)  # Component labels for selection
    component_regions = measure.regionprops(label_array)  # Region properties for each component

    if len(component_regions) == 0:
        return np.zeros(image_shape, dtype=np.uint8), True

    image_height = image_shape[0]  # Image height in pixels
    image_width = image_shape[1]  # Image width in pixels
    center_y = (image_height - 1) / 2.0  # Central y coordinate of the patch
    center_x = (image_width - 1) / 2.0  # Central x coordinate of the patch

    y_grid, x_grid = np.ogrid[:image_height, :image_width]  # Coordinate grids used to build a central circle
    central_circle = ((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2) <= (central_radius ** 2)  # Central circular region

    largest_area = max(region.area for region in component_regions)  # Area of the largest component
    selected_labels = []  # Labels selected as relevant mitotic components

    for region in component_regions:
        component_mask = label_array == region.label  # Binary mask for the current component
        centroid_y, centroid_x = region.centroid  # Component centroid in row-column coordinates
        centroid_distance = math.sqrt((centroid_x - center_x) ** 2 + (centroid_y - center_y) ** 2)  # Distance to patch center
        overlaps_center = bool(np.any(component_mask & central_circle))  # Whether component overlaps central circle
        is_large_and_close = (region.area >= 0.20 * largest_area) and (centroid_distance <= 1.5 * central_radius)

        if overlaps_center or centroid_distance <= central_radius or is_large_and_close:
            selected_labels.append(region.label)

    fallback_used = False  # Whether a fallback component selection had to be used

    if len(selected_labels) == 0:
        sorted_regions = sorted(
            component_regions,
            key=lambda region: math.sqrt((region.centroid[1] - center_x) ** 2 + (region.centroid[0] - center_y) ** 2),
        )  # Components sorted by distance to image center
        selected_labels.append(sorted_regions[0].label)
        fallback_used = True

    selected_mask = np.isin(label_array, selected_labels).astype(np.uint8)  # Final selected component mask

    return selected_mask, fallback_used


# Creates a colored overlay of a binary mask on top of an RGB image.
def create_overlay(rgb_array, mask, color_value=(1.0, 0.0, 0.0), alpha=0.45):
    base_rgb = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)  # Base RGB image
    binary_mask = mask.astype(bool)  # Mask used for overlay
    overlay_color = np.array(color_value, dtype=np.float32)  # RGB overlay color

    overlay = base_rgb.copy()  # Output overlay initialized from the original RGB image
    overlay[binary_mask] = (
        (1.0 - alpha) * overlay[binary_mask]
        + alpha * overlay_color
    )  # Alpha-blended overlay pixels

    return np.clip(overlay, 0.0, 1.0)

# Extracts RGB, HSV, and LAB color statistics from a selected image region.
def extract_region_color_features(prefix, region_mask, rgb_array, hsv_array, lab_array):
    region_binary = region_mask.astype(bool)  # Binary region used to select pixels

    feature_values = {}  # Dictionary storing color statistics

    color_spaces = [
        ("rgb", rgb_array, ["r", "g", "b"]),
        ("hsv", hsv_array, ["h", "s", "v"]),
        ("lab", lab_array, ["l", "a", "b"]),
    ]  # Color spaces and channel names used for feature extraction

    for space_name, space_array, channel_names in color_spaces:
        if np.sum(region_binary) > 0:
            region_pixels = space_array[region_binary]  # Pixels inside the selected region
            mean_values = np.mean(region_pixels, axis=0)  # Mean channel values in the region
            std_values = np.std(region_pixels, axis=0)  # Standard deviation of channel values in the region
        else:
            mean_values = np.array([np.nan, np.nan, np.nan])
            std_values = np.array([np.nan, np.nan, np.nan])

        for channel_index, channel_name in enumerate(channel_names):
            feature_values[f"{prefix}_mean_{space_name}_{channel_name}"] = mean_values[channel_index]
            feature_values[f"{prefix}_std_{space_name}_{channel_name}"] = std_values[channel_index]

    return feature_values


# Extracts component-based geometric features from a binary mask.
def extract_component_shape_features(prefix, binary_mask, reference_area=None):
    region_binary = binary_mask.astype(bool)  # Binary region used for shape analysis
    region_area = int(np.sum(region_binary))  # Number of foreground pixels

    label_array = label_connected_components(region_binary, min_component_area=1)  # Connected components
    component_regions = measure.regionprops(label_array)  # Region properties for each component

    number_of_components = int(len(component_regions))  # Number of connected components
    component_areas = np.array(
        [region.area for region in component_regions],
        dtype=np.float32,
    )  # Area of each component

    largest_component_area = int(np.max(component_areas)) if component_areas.size > 0 else 0
    largest_component_ratio = float(largest_component_area / region_area) if region_area > 0 else 0.0
    mean_component_area = float(np.mean(component_areas)) if component_areas.size > 0 else 0.0
    std_component_area = float(np.std(component_areas)) if component_areas.size > 0 else 0.0

    component_density = (
        float(number_of_components / reference_area)
        if reference_area is not None and reference_area > 0
        else 0.0
    )  # Number of components normalized by reference area

    granularity_index = float(number_of_components / region_area) if region_area > 0 else 0.0

    total_perimeter = 0.0
    compactness = 0.0

    if region_area > 0:
        total_perimeter = float(measure.perimeter(region_binary, neighborhood=8))
        compactness = float((4.0 * math.pi * region_area) / (total_perimeter ** 2)) if total_perimeter > 0 else 0.0

    if largest_component_area > 0:
        largest_region = max(component_regions, key=lambda region: region.area)
        centroid_y, centroid_x = largest_region.centroid
    else:
        centroid_x = np.nan
        centroid_y = np.nan

    skeleton = morphology.skeletonize(region_binary)  # Skeletonized binary mask
    skeleton_length = int(np.sum(skeleton))  # Approximate length in pixels

    skeleton_length_ratio = (
        float(skeleton_length / reference_area)
        if reference_area is not None and reference_area > 0
        else 0.0
    )

    return {
        f"{prefix}_area": region_area,
        f"{prefix}_number_of_components": number_of_components,
        f"{prefix}_largest_component_area": largest_component_area,
        f"{prefix}_largest_component_ratio": largest_component_ratio,
        f"{prefix}_mean_component_area": mean_component_area,
        f"{prefix}_std_component_area": std_component_area,
        f"{prefix}_component_density": component_density,
        f"{prefix}_granularity_index": granularity_index,
        f"{prefix}_skeleton_length": skeleton_length,
        f"{prefix}_skeleton_length_ratio": skeleton_length_ratio,
        f"{prefix}_total_perimeter": total_perimeter,
        f"{prefix}_compactness": compactness,
        f"{prefix}_centroid_x": centroid_x,
        f"{prefix}_centroid_y": centroid_y,
    }

# Extracts image-level, cell-level, and chromatin-level features for later modeling.
def extract_segmentation_features(mask, tissue_mask, cell_mask, rgb_array, hsv_array, lab_array, label_info):
    chromatin_binary = mask.astype(bool)  # Final selected chromatin mask
    tissue_binary = tissue_mask.astype(bool)  # Final tissue mask
    cell_binary = cell_mask.astype(bool)  # Central cell or nuclear-region mask
    image_binary = np.ones(rgb_array.shape[:2], dtype=bool)  # Whole image region

    tissue_area = int(np.sum(tissue_binary))  # Number of tissue pixels
    cell_area = int(np.sum(cell_binary))  # Number of central cell pixels
    chromatin_area = int(np.sum(chromatin_binary))  # Number of selected chromatin pixels
    image_area = int(np.sum(image_binary))  # Number of pixels in the full patch

    chromatin_area_ratio_over_tissue = float(chromatin_area / tissue_area) if tissue_area > 0 else 0.0
    chromatin_area_ratio_over_cell = float(chromatin_area / cell_area) if cell_area > 0 else 0.0
    cell_area_ratio_over_tissue = float(cell_area / tissue_area) if tissue_area > 0 else 0.0
    cell_area_ratio_over_image = float(cell_area / image_area) if image_area > 0 else 0.0

    image_color_features = extract_region_color_features(
        "image",
        image_binary,
        rgb_array,
        hsv_array,
        lab_array,
    )  # RGB, HSV, and LAB statistics for the full image

    cell_color_features = extract_region_color_features(
        "cell",
        cell_binary,
        rgb_array,
        hsv_array,
        lab_array,
    )  # RGB, HSV, and LAB statistics for the central cell region

    chromatin_color_features = extract_region_color_features(
        "chromatin",
        chromatin_binary,
        rgb_array,
        hsv_array,
        lab_array,
    )  # RGB, HSV, and LAB statistics for the chromatin region

    cell_shape_features = extract_component_shape_features(
        "cell",
        cell_binary,
        reference_area=tissue_area,
    )  # Shape and component features for the central cell region

    chromatin_shape_features = extract_component_shape_features(
        "chromatin",
        chromatin_binary,
        reference_area=cell_area,
    )  # Shape, skeleton, and granularity features for chromatin

    feature_row = {
        "source_csv_image_id": label_info.get("source_csv_image_id", ""),  # Original image_id from raw CSV

        "final_label": label_info.get("final_label", ""),  # Majority-based class label

        "image_area": image_area,
        "tissue_area": tissue_area,
        "cell_area": cell_area,
        "chromatin_area": chromatin_area,
        "cell_area_ratio_over_tissue": cell_area_ratio_over_tissue,
        "cell_area_ratio_over_image": cell_area_ratio_over_image,
        "chromatin_area_ratio_over_tissue": chromatin_area_ratio_over_tissue,
        "chromatin_area_ratio_over_cell": chromatin_area_ratio_over_cell,

        # Backward-compatible columns used by the processing report and older ML scripts.
        "number_of_components": chromatin_shape_features["chromatin_number_of_components"],
        "largest_component_area": chromatin_shape_features["chromatin_largest_component_area"],
        "largest_component_ratio": chromatin_shape_features["chromatin_largest_component_ratio"],
        "mean_component_area": chromatin_shape_features["chromatin_mean_component_area"],
        "std_component_area": chromatin_shape_features["chromatin_std_component_area"],
        "component_density_over_cell": chromatin_shape_features["chromatin_component_density"],
        "granularity_index": chromatin_shape_features["chromatin_granularity_index"],
        "chromatin_skeleton_length": chromatin_shape_features["chromatin_skeleton_length"],
        "chromatin_skeleton_length_ratio_over_cell": chromatin_shape_features["chromatin_skeleton_length_ratio"],
        "total_perimeter": chromatin_shape_features["chromatin_total_perimeter"],
        "compactness": chromatin_shape_features["chromatin_compactness"],
        "centroid_x": chromatin_shape_features["chromatin_centroid_x"],
        "centroid_y": chromatin_shape_features["chromatin_centroid_y"],

        "processing_status": "success",
    }

    feature_row.update(image_color_features)
    feature_row.update(cell_color_features)
    feature_row.update(chromatin_color_features)
    feature_row.update(cell_shape_features)
    feature_row.update(chromatin_shape_features)

    return feature_row


# Builds warning flags for segmentation quality control.
def build_warning_flags(tissue_mask, raw_mask, selected_mask, number_of_components, central_fallback_used):
    tissue_area = int(np.sum(tissue_mask > 0))  # Number of tissue pixels
    raw_area = int(np.sum(raw_mask > 0))  # Number of raw Otsu foreground pixels
    selected_area = int(np.sum(selected_mask > 0))  # Number of selected chromatin pixels
    image_area = int(tissue_mask.shape[0] * tissue_mask.shape[1])  # Total image area in pixels

    warning_flags = {
        "warning_empty_mask": selected_area == 0,
        "warning_too_large_mask": (selected_area / tissue_area > 0.60) if tissue_area > 0 else False,
        "warning_tissue_almost_empty": (tissue_area / image_area < 0.05) if image_area > 0 else True,
        "warning_too_many_components": number_of_components > 20,
        "warning_selected_empty_but_raw_not_empty": selected_area == 0 and raw_area > 0,
        "warning_central_fallback_used": bool(central_fallback_used),
    }

    return warning_flags


# Processes a single image patch and saves all intermediate outputs.
def process_single_image(label_info, output_dirs, args, image_index):
    image_id = str(label_info.get("image_id_for_processing", ""))  # Original image identifier from metadata
    image_path = Path(str(label_info.get("image_path", "")))  # Resolved image path for processing

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    safe_stem = image_path.stem if image_path.stem else f"image_{image_index:06d}"  # Safe file stem for output names
    output_name = f"{safe_stem}.png"  # Standard PNG output file name

    rgb_array = read_rgb_image(image_path, args.target_size)  # Input RGB patch in [0, 1]
    normalized_rgb = normalize_rgb_image(rgb_array)  # Percentile-normalized RGB patch
    hsv_array, lab_array = convert_color_spaces(normalized_rgb)  # HSV and LAB representations

    lab_l_norm = np.clip(lab_array[:, :, 0] / 100.0, 0.0, 1.0)  # LAB L normalized for saving
    lab_a_vis = scale_channel_for_visualization(lab_array[:, :, 1])  # LAB A normalized only for visualization
    lab_b_vis = scale_channel_for_visualization(lab_array[:, :, 2])  # LAB B normalized only for visualization

    save_rgb_image(normalized_rgb, output_dirs["normalized_rgb"] / output_name)
    save_grayscale_image(hsv_array[:, :, 0], output_dirs["hsv_h"] / output_name)
    save_grayscale_image(hsv_array[:, :, 1], output_dirs["hsv_s"] / output_name)
    save_grayscale_image(hsv_array[:, :, 2], output_dirs["hsv_v"] / output_name)
    save_grayscale_image(lab_l_norm, output_dirs["lab_l"] / output_name)
    save_grayscale_image(lab_a_vis, output_dirs["lab_a"] / output_name)
    save_grayscale_image(lab_b_vis, output_dirs["lab_b"] / output_name)

    tissue_mask = create_tissue_mask(normalized_rgb, hsv_array, lab_array)  # Refined tissue mask
    cell_score = build_cell_score(normalized_rgb, hsv_array, lab_array, tissue_mask)  # Broader central cell/nuclear score
    central_cell_mask = segment_central_cell_by_otsu(cell_score, tissue_mask, tissue_mask.shape, args.central_radius,)  # Central cell/nuclear mask
    chromatin_score = build_chromatin_score(normalized_rgb,hsv_array, lab_array, central_cell_mask,) # Chromatin score inside the central cell region
    raw_chromatin_mask = segment_chromatin_otsu(chromatin_score, central_cell_mask,)  # Raw chromatin mask obtained with Otsu
    refined_mask = refine_binary_mask(
        raw_chromatin_mask,
        min_component_area=args.min_component_area,
        opening_radius=1,
        closing_radius=0,
        fill_holes=False,
    )  # Refined chromatin mask without merging fragments
    selected_mask, central_fallback_used = select_relevant_components(
        refined_mask,
        refined_mask.shape,
        args.central_radius,
    )  # Final selected central chromatin mask

    save_grayscale_image(tissue_mask, output_dirs["tissue_mask"] / output_name)
    save_grayscale_image(chromatin_score, output_dirs["chromatin_score"] / output_name)
    save_grayscale_image(raw_chromatin_mask, output_dirs["chromatin_otsu_raw"] / output_name)    
    save_grayscale_image(refined_mask, output_dirs["chromatin_refined"] / output_name)
    save_grayscale_image(selected_mask, output_dirs["selected_components"] / output_name)
    save_grayscale_image(cell_score, output_dirs["cell_score"] / output_name)
    save_grayscale_image(central_cell_mask, output_dirs["central_cell_mask"] / output_name)

    save_overlays = args.save_debug_sample is None or image_index < args.save_debug_sample  # Whether overlays should be saved
    if save_overlays:
        tissue_overlay = create_overlay(normalized_rgb, tissue_mask, color_value=(0.0, 1.0, 0.0), alpha=0.35)
        chromatin_overlay = create_overlay(normalized_rgb, raw_chromatin_mask, color_value=(1.0, 0.0, 0.0), alpha=0.45)
        selected_overlay = create_overlay(normalized_rgb, selected_mask, color_value=(1.0, 1.0, 0.0), alpha=0.50)
        cell_overlay = create_overlay(normalized_rgb, central_cell_mask, color_value=(0.0, 0.0, 1.0), alpha=0.35)
        
        save_rgb_image(cell_overlay, output_dirs["cell_overlay"] / output_name)
        save_rgb_image(tissue_overlay, output_dirs["tissue_overlay"] / output_name)
        save_rgb_image(chromatin_overlay, output_dirs["chromatin_overlay"] / output_name)
        save_rgb_image(selected_overlay, output_dirs["selected_overlay"] / output_name)

    feature_row = extract_segmentation_features(
        selected_mask,
        tissue_mask,
        central_cell_mask,
        normalized_rgb,
        hsv_array,
        lab_array,
        label_info,
    ) # Feature row for the selected chromatin mask

    warning_flags = build_warning_flags(
        tissue_mask,
        raw_chromatin_mask,
        selected_mask,
        feature_row["number_of_components"],
        central_fallback_used,
    )  # Quality-control warning flags

    report_row = {
        "image_id": image_id,
        "image_path": str(image_path),
        "final_label": label_info.get("final_label", ""),
        "numeric_label": label_info.get("numeric_label", np.nan),
        "status": "success",
        "error_message": "",
        "number_of_components": feature_row["number_of_components"],
        "tissue_area": feature_row["tissue_area"],
        "chromatin_area": feature_row["chromatin_area"],
        **warning_flags,
    }

    feature_row.update(warning_flags)

    return feature_row, report_row, None


# Scales one scalar channel to [0, 1] only for visualization.
def scale_channel_for_visualization(channel_array):
    channel = np.asarray(channel_array, dtype=np.float32)  # Scalar channel to normalize
    min_value = float(np.nanmin(channel))  # Minimum channel value
    max_value = float(np.nanmax(channel))  # Maximum channel value

    if max_value <= min_value:
        return np.zeros_like(channel, dtype=np.float32)

    return np.clip((channel - min_value) / (max_value - min_value), 0.0, 1.0).astype(np.float32)


# Processes the full dataset and writes feature, report, and error CSV files.
def process_dataset(labels_table, output_dirs, args):
    feature_rows = []  # Segmentation feature records
    report_rows = []  # Per-image processing status records
    error_rows = []  # Error records for failed images
    
    records = labels_table.to_dict(orient="records")  # Metadata rows converted to dictionaries
    
    if args.max_images is not None:
        records = records[:args.max_images]  # Only the first N images are processed for quick testing
        
    iterator = tqdm(records, desc="Processing MIDOG patches") if tqdm is not None else records

    for image_index, label_info in enumerate(iterator):
        try:
            feature_row, report_row, error_row = process_single_image(
                label_info,
                output_dirs,
                args,
                image_index,
            )
            if feature_row is not None:
                feature_rows.append(feature_row)
            if report_row is not None:
                report_rows.append(report_row)
            if error_row is not None:
                error_rows.append(error_row)

        except Exception as error:
            image_id = str(label_info.get("image_id_for_processing", ""))  # Image identifier for the failed row
            image_path = str(label_info.get("image_path", ""))  # Image path for the failed row
            error_message = str(error)  # Error message for logging

            report_rows.append({
                "image_id": image_id,
                "image_path": image_path,
                "final_label": label_info.get("final_label", ""),
                "numeric_label": label_info.get("numeric_label", np.nan),
                "status": "failed",
                "error_message": error_message,
                "number_of_components": 0,
                "tissue_area": 0,
                "chromatin_area": 0,
                "warning_empty_mask": True,
                "warning_too_large_mask": False,
                "warning_tissue_almost_empty": True,
                "warning_too_many_components": False,
                "warning_selected_empty_but_raw_not_empty": False,
                "warning_central_fallback_used": False,
            })

            error_rows.append({
                "image_id": image_id,
                "image_path": image_path,
                "error_message": error_message,
            })

    features_table = pd.DataFrame(feature_rows)  # Final feature table
    report_table = pd.DataFrame(report_rows)  # Final processing report table
    errors_table = pd.DataFrame(error_rows)  # Final error log table

    features_table.to_csv(output_dirs["features"] / "segmentation_features.csv", index=False)
    report_table.to_csv(output_dirs["metadata"] / "processing_report.csv", index=False)
    errors_table.to_csv(output_dirs["logs"] / "errors.csv", index=False)

    return features_table, report_table, errors_table


# Runs the full MIDOG preparation pipeline.
def main():
    args = parse_arguments()  # Parsed command-line arguments

    images_dir = Path(args.images_dir)  # Folder containing the input PNG patches
    labels_csv = Path(args.labels_csv)  # CSV file containing metadata and labels
    output_dir = Path(args.output_dir)  # Root folder for outputs

    if not images_dir.exists():
        print(f"Error: images_dir does not exist: {images_dir}", file=sys.stderr)
        sys.exit(1)

    if not labels_csv.exists():
        print(f"Error: labels_csv does not exist: {labels_csv}", file=sys.stderr)
        sys.exit(1)

    output_dirs = create_output_directories(output_dir)  # Dictionary with all output folders
    labels_table = load_labels_table(labels_csv, images_dir, output_dirs)  # Valid metadata table with majority labels

    if labels_table.empty:
        print("Error: no valid rows with NMF or AMF majority labels were found.", file=sys.stderr)
        sys.exit(1)

    process_dataset(labels_table, output_dirs, args)

    print(f"Processing finished. Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
