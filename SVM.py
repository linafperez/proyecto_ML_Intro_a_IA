from sklearn.svm import SVC
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report

DATA_PATH = "data/feature_matrix_variants/rgb_cell_chromatin/"

# Importar los datos de entrenamiento y validacion
x_train = np.load(os.path.join(DATA_PATH, "X_training.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_training.npy"))
x_val = np.load(os.path.join(DATA_PATH, "X_validation.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_validation.npy"))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

svm = SVC(kernel="rbf", C=0.6, class_weight="balanced")
svm.fit(x_train, y_train)

y_pred = svm.predict(x_val)
balanced_acc = balanced_accuracy_score(y_val, y_pred)

print('\n' + '='*40)
print('       MÉTRICAS SVM        ')
print('='*40)
print(f'Balanced Accuracy (BA) Oficial: {balanced_acc:.4f}')
print('-'*40)
print(classification_report(y_val, y_pred, target_names=['NMF (Normal)', 'AMF (Atípica)']))
