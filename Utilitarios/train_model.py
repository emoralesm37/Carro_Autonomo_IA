# utils/train_model.py
"""
Entrena el MLP y guarda el modelo para producción.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PIPELINE DE ENTRENAMIENTO
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Cargar data/dataset.csv
  2. Separar features (X) y etiquetas (y)
  3. Split 80/20 estratificado (misma proporción de clases en train/test)
  4. StandardScaler → normalizar entradas (media=0, std=1)
  5. Entrenar MLPClassifier con early stopping
     Arquitectura: 4 entradas → 16 → 8 → 4 salidas (softmax)
  6. Evaluar en test set: accuracy, reporte por clase, matriz de confusión
  7. Guardar modelo + scaler en models/ con joblib

Los resultados de esta evaluación van en el reporte del proyecto bajo
"Datos de entrenamiento y datos de pruebas utilizados para la red neuronal".

Ejecutar: python utils/train_model.py
Salida:   models/modelo.pkl  models/scaler.pkl
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Ajustar path para importar config desde el directorio raíz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import ACCIONES

DATASET_PATH = "data/dataset.csv"
MODELO_PATH  = "models/modelo.pkl"
SCALER_PATH  = "models/scaler.pkl"
SEED         = 42


def main() -> None:
    # ── 1. Cargar dataset ─────────────────────────────────────
    if not os.path.exists(DATASET_PATH):
        print(f"[ERROR] No se encontró '{DATASET_PATH}'")
        print("        Ejecuta primero: python utils/generate_dataset.py")
        return

    df = pd.read_csv(DATASET_PATH)
    X  = df[["dist_frente_norm", "desviacion", "izq_libre", "der_libre"]].values
    y  = df["accion"].values.astype(int)

    print(f"\n[TRAIN] Dataset cargado: {len(df)} muestras")
    uniq, counts = np.unique(y, return_counts=True)
    for u, c in zip(uniq, counts):
        print(f"        Clase {u} ({ACCIONES[u]:12s}): {c} muestras")

    # ── 2. Split 80/20 estratificado ─────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=SEED
    )
    print(f"\n[SPLIT] Entrenamiento: {len(X_tr)} muestras")
    print(f"        Test         : {len(X_te)} muestras  (20%)\n")

    # ── 3. Normalización ──────────────────────────────────────
    # StandardScaler calcula media/std SOLO sobre train para evitar data leakage
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr)
    X_te   = scaler.transform(X_te)

    # ── 4. Entrenar MLP ───────────────────────────────────────
    # Arquitectura: 4 → 16 → 8 → 4
    # early_stopping: detiene si val_loss no mejora en 30 épocas consecutivas
    modelo = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        solver="adam",
        max_iter=2000,
        random_state=SEED,
        early_stopping=True,
        validation_fraction=0.10,   # 10% del train como validación interna
        n_iter_no_change=30,
        verbose=False
    )

    print("[TRAIN] Entrenando MLP (4 → 16 → 8 → 4)...")
    modelo.fit(X_tr, y_tr)
    print(f"        Épocas completadas : {modelo.n_iter_}")
    print(f"        Mejor loss val     : {modelo.best_loss_:.4f}\n")

    # ── 5. Evaluación en test set ─────────────────────────────
    y_pred = modelo.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)

    print(f"[EVAL]  Accuracy en test (20%) : {acc * 100:.1f}%")
    print()
    print(classification_report(y_te, y_pred, target_names=ACCIONES))

    print("[EVAL]  Matriz de confusión (filas=real, columnas=predicho):")
    cm = confusion_matrix(y_te, y_pred)
    header = f"{'':12}" + "".join(f"{a:>12}" for a in ACCIONES)
    print(header)
    for i, row in enumerate(cm):
        print(f"{ACCIONES[i]:12}" + "".join(f"{v:>12}" for v in row))
    print()

    # ── 6. Guardar modelo y scaler ────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(modelo, MODELO_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SAVE]  Modelo  guardado → {MODELO_PATH}")
    print(f"[SAVE]  Scaler  guardado → {SCALER_PATH}")
    print(f"\n✅ Listo. Copia el proyecto al Raspberry Pi y ejecuta main.py")


if __name__ == "__main__":
    main()
