

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ─────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────

VARIANTS_ROOT = Path("data/feature_matrix_variants")
VARIANTS      = ["rgb_cell_chromatin", "lab_cell_chromatin", "hsv_cell_chromatin"]

KNN_K_VALUES = [1, 3, 5, 7, 9, 11]
DT_MAX_DEPTH = 5
RANDOM_STATE = 42


# ─────────────────────────────────────────────
# Funciones auxiliares
# ─────────────────────────────────────────────

def compute_metrics(name, y_true, y_pred, y_proba=None):
    """
    Balanced Accuracy (BA) = 0.5 * (recall_NMF + recall_AMF)
    Es la métrica principal porque el dataset está desbalanceado.
    """
    return {
        "modelo"            : name,
        "balanced_accuracy" : balanced_accuracy_score(y_true, y_pred),
        "accuracy"          : accuracy_score(y_true, y_pred),
        "precision"         : precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall"            : recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1"                : f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "auc_roc"           : (
            roc_auc_score(y_true, y_proba[:, 1])
            if y_proba is not None and y_proba.shape[1] == 2
            else np.nan
        ),
    }


def plot_confusion_matrix(y_true, y_pred, class_names, title, filepath):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=0.5)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel("Predicción", fontsize=11)
    ax.set_ylabel("Valor real", fontsize=11)
    plt.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"    → Guardada: {filepath}")


def run_variant(variant_name):
    data_dir   = VARIANTS_ROOT / variant_name
    output_dir = data_dir / "sk_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  VARIANTE: {variant_name}")
    print(f"{'='*60}")

    # Carga
    X_train = np.load(data_dir / "X_training.npy")
    y_train = np.load(data_dir / "y_training.npy")
    X_val   = np.load(data_dir / "X_validation.npy")
    y_val   = np.load(data_dir / "y_validation.npy")

    label_mapping = pd.read_csv(data_dir / "label_mapping.csv")
    id2label      = dict(zip(label_mapping["numeric_label"], label_mapping["label_name"]))
    class_names   = [id2label[i] for i in sorted(id2label)]

    print(f"  X_train: {X_train.shape}  X_val: {X_val.shape}  Clases: {class_names}")

    # Distribución original
    classes, counts = np.unique(y_train, return_counts=True)
    print(f"  Distribución train original: { {id2label[c]: n for c, n in zip(classes, counts)} }")

    # Normalización
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # ── Sobremuestreo para KNN ────────────────

    ros = RandomOverSampler(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = ros.fit_resample(X_train_s, y_train)
    classes_bal, counts_bal = np.unique(y_train_bal, return_counts=True)
    print(f"  Distribución train balanceado (KNN): { {id2label[c]: n for c, n in zip(classes_bal, counts_bal)} }")

    # ── KNN sobre datos balanceados ───────────
    print(f"\n  KNN — k ∈ {KNN_K_VALUES}  (con sobremuestreo RandomOverSampler)")
    knn_results = []
    for k in KNN_K_VALUES:
        knn     = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_bal, y_train_bal)
        y_pred  = knn.predict(X_val_s)
        y_proba = knn.predict_proba(X_val_s)
        m = compute_metrics(f"KNN k={k}", y_val, y_pred, y_proba)
        knn_results.append({**m, "k": k, "model_obj": knn,
                             "y_pred": y_pred, "y_proba": y_proba})
        print(f"    k={k:2d}  ba={m['balanced_accuracy']:.4f}  acc={m['accuracy']:.4f}  "
              f"f1={m['f1']:.4f}  auc={m['auc_roc']:.4f}")

    best_knn = max(knn_results, key=lambda r: r["balanced_accuracy"])
    print(f"\n  Mejor KNN: k={best_knn['k']}  (BA={best_knn['balanced_accuracy']:.4f})")
    print(f"\n  --- Reporte KNN (k={best_knn['k']}) ---")
    print(classification_report(y_val, best_knn["y_pred"],
                                 target_names=class_names, zero_division=0))

    plot_confusion_matrix(y_val, best_knn["y_pred"], class_names,
                          title=f"KNN (k={best_knn['k']}) — {variant_name}",
                          filepath=output_dir / "cm_knn.png")


    print(f"\n  Decision Tree (max_depth={DT_MAX_DEPTH}, class_weight='balanced')")
    dt = DecisionTreeClassifier(max_depth=DT_MAX_DEPTH,
                                 class_weight="balanced",
                                 random_state=RANDOM_STATE)
    dt.fit(X_train_s, y_train)
    y_pred_dt  = dt.predict(X_val_s)
    y_proba_dt = dt.predict_proba(X_val_s)
    m_dt = compute_metrics(f"DT max_depth={DT_MAX_DEPTH}", y_val, y_pred_dt, y_proba_dt)
    print(f"    ba={m_dt['balanced_accuracy']:.4f}  acc={m_dt['accuracy']:.4f}  "
          f"f1={m_dt['f1']:.4f}  auc={m_dt['auc_roc']:.4f}")
    print(f"\n  --- Reporte Decision Tree ---")
    print(classification_report(y_val, y_pred_dt, target_names=class_names, zero_division=0))

    plot_confusion_matrix(y_val, y_pred_dt, class_names,
                          title=f"Decision Tree (max_depth={DT_MAX_DEPTH}) — {variant_name}",
                          filepath=output_dir / "cm_dt.png")


    all_metrics = []
    for r in knn_results:
        all_metrics.append({k: r[k] for k in
            ["modelo","balanced_accuracy","accuracy","precision","recall","f1","auc_roc"]})
    all_metrics.append({k: m_dt[k] for k in
        ["modelo","balanced_accuracy","accuracy","precision","recall","f1","auc_roc"]})

    df = pd.DataFrame(all_metrics).round(4)
    df.to_csv(output_dir / "metrics_summary.csv", index=False)

    best_row  = df.loc[df["balanced_accuracy"].idxmax()]
    best_pred = best_knn["y_pred"] if "KNN" in best_row["modelo"] else y_pred_dt

    print(f"\n  Mejor modelo en {variant_name}: {best_row['modelo']}")
    print(f"    Balanced Accuracy : {best_row['balanced_accuracy']:.4f}  ← métrica principal")
    print(f"    Accuracy          : {best_row['accuracy']:.4f}")
    print(f"    F1                : {best_row['f1']:.4f}")
    print(f"    AUC-ROC           : {best_row['auc_roc']:.4f}")

    plot_confusion_matrix(y_val, best_pred, class_names,
                          title=f"Mejor modelo — {best_row['modelo']} ({variant_name})",
                          filepath=output_dir / "best_model_cm.png")

    return {
        "variante"          : variant_name,
        "mejor_modelo"      : best_row["modelo"],
        "balanced_accuracy" : best_row["balanced_accuracy"],
        "accuracy"          : best_row["accuracy"],
        "f1"                : best_row["f1"],
        "auc_roc"           : best_row["auc_roc"],
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

summary = []
for v in VARIANTS:
    summary.append(run_variant(v))

print(f"\n{'='*60}")
print("  RESUMEN COMPARATIVO — TODAS LAS VARIANTES")
print(f"{'='*60}")
df_summary = pd.DataFrame(summary).round(4)
print(df_summary.to_string(index=False))

best_global = df_summary.loc[df_summary["balanced_accuracy"].idxmax()]
print(f"\nMejor combinación global:")
print(f"   Variante          : {best_global['variante']}")
print(f"   Modelo            : {best_global['mejor_modelo']}")
print(f"   Balanced Accuracy : {best_global['balanced_accuracy']:.4f}")
print(f"   Accuracy          : {best_global['accuracy']:.4f}")
print(f"   F1                : {best_global['f1']:.4f}")
print(f"   AUC-ROC           : {best_global['auc_roc']:.4f}")

df_summary.to_csv(VARIANTS_ROOT / "sk_variant_comparison.csv", index=False)
print(f"\n  Comparación guardada en: {VARIANTS_ROOT / 'sk_variant_comparison.csv'}")
