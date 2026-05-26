import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Input, Dropout
from keras.optimizers import Optimizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, classification_report

# Ruta de los datos
DATA_PATH = "data/processed/ml_ready/"

# Importar los datos de entrenamiento y validacion
x_train = np.load(os.path.join(DATA_PATH, "X_training.npy"))
y_train = np.load(os.path.join(DATA_PATH, "y_training.npy"))
x_val = np.load(os.path.join(DATA_PATH, "X_validation.npy"))
y_val = np.load(os.path.join(DATA_PATH, "y_validation.npy"))

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)

print("Caracteristicas de entrenamiento (x_train):", x_train.shape)
print("Etiquetas de entrenamiento (y_train):", y_train.shape)
#print("Caracteristicas de validacion (x_test):", x_val.shape)
print("Etiquetas de validacion (y_test):", y_val.shape)
print("Cantidad de imagenes de prueba: ", x_train.shape[0])

num_features = x_train.shape[1]

model = Sequential([
    Input(shape=(num_features,)),
    Dense(128, activation='relu'),
    Dropout(0.2),

    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

mod = model.fit(x_train, y_train, 
                epochs=30,
                batch_size=64,
                validation_data=(x_val, y_val),
                class_weight={0: 1.0, 1: 4.0})

y_pred_probs = model.predict(x_val)

y_pred = np.argmax(y_pred_probs, axis=1)

balanced_acc = balanced_accuracy_score(y_val, y_pred)

print('\n' + '='*40)
print('          MÉTRICAS DE EVALUACIÓN          ')
print('='*40)
print(f'Balanced Accuracy (BA) Oficial: {balanced_acc:.4f}')
print('-'*40)

print(classification_report(y_val, y_pred, target_names=['NMF (Normal)', 'AMF (Atípica)']))

#results = model.evaluate(x_val, y_val, verbose=0)
#print('Test loss, Test accuracy: ', results)
