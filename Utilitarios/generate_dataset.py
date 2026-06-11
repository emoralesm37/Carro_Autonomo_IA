# utils/generate_dataset.py
"""
Genera el dataset de entrenamiento para la red neuronal (MLP).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 CARACTERÍSTICAS DE ENTRADA (X) — 4 columnas
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  dist_frente_norm   float  distancia frontal normalizada [0.0 – 1.0]
  desviacion         float  desviación de carril          [-1.0 – +1.0]
  izq_libre          int    1 = izquierda libre, 0 = bloqueada
  der_libre          int    1 = derecha  libre, 0 = bloqueada

 ETIQUETA DE SALIDA (y) — 1 columna
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  accion             int    0=avanzar | 1=izquierda | 2=derecha | 3=retroceder

 DISEÑO DE MUESTRAS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Cada clase se genera con reglas deterministas + ruido Gaussiano.
  Se incluye un 10% de muestras ambiguas (casos donde izquierda o
  derecha son igualmente válidas) para evitar que la MLP memorice
  las reglas exactas y que las métricas de test sean honestas.

Ejecutar: python utils/generate_dataset.py
Salida:   data/dataset.csv
"""

import csv
import os
import random
import sys

import numpy as np

# Ajustar path para importar config desde el directorio raíz
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import DISTANCIA_SEGURA, DISTANCIA_FRENADO, DISTANCIA_MAX_NN

# ─────────────────────────────────────────────────────────────
# PARÁMETROS DE GENERACIÓN
# ─────────────────────────────────────────────────────────────
N_AVANZAR     = 1200
N_IZQUIERDA   =  600
N_DERECHA     =  600
N_RETROCEDER  =  400

RUIDO_DIST    = 3.0    # σ ruido en cm (antes de normalizar)
RUIDO_DESV    = 0.05   # σ ruido en desviación

OUTPUT        = "data/dataset.csv"
SEED          = 42

np.random.seed(SEED)
random.seed(SEED)


# ─────────────────────────────────────────────────────────────
# FUNCIONES AUXILIARES
# ─────────────────────────────────────────────────────────────

def norm(d: float) -> float:
    """Normaliza distancia con ruido a [0.0, 1.0]."""
    return float(np.clip(
        (d + np.random.normal(0, RUIDO_DIST)) / DISTANCIA_MAX_NN,
        0.0, 1.0
    ))


def desv_ruido(d: float) -> float:
    """Agrega ruido a la desviación de carril y recorta a [-1, 1]."""
    return float(np.clip(d + np.random.normal(0, RUIDO_DESV), -1.0, 1.0))


def uni(a: float, b: float) -> float:
    return random.uniform(a, b)


# ─────────────────────────────────────────────────────────────
# GENERACIÓN DE MUESTRAS POR CLASE
# ─────────────────────────────────────────────────────────────

def generar() -> list:
    filas = []

    # ── CLASE 0: avanzar ─────────────────────────────────────
    # Camino libre (dist > SEGURA) y carro centrado en carril (|desv| ≤ 0.14)
    for _ in range(N_AVANZAR):
        dist = uni(DISTANCIA_SEGURA + 1, DISTANCIA_MAX_NN)
        desv = uni(-0.14, 0.14)
        filas.append([norm(dist), desv_ruido(desv), 1, 1, 0])

    # ── CLASE 1: izquierda ───────────────────────────────────

    # Subclase A: obstáculo al frente, izquierda libre, derecha bloqueada
    for _ in range(N_IZQUIERDA // 2):
        dist = uni(DISTANCIA_FRENADO + 1, DISTANCIA_SEGURA - 1)
        desv = uni(-0.50, 0.50)
        filas.append([norm(dist), desv_ruido(desv), 1, 0, 1])

    # Subclase B: camino libre pero carro desviado a la derecha → corregir izq
    for _ in range(N_IZQUIERDA // 2):
        dist = uni(DISTANCIA_SEGURA + 1, DISTANCIA_MAX_NN)
        desv = uni(0.16, 1.0)    # desviación positiva = desviado a la derecha
        filas.append([norm(dist), desv_ruido(desv), 1, 1, 1])

    # ── CLASE 2: derecha ─────────────────────────────────────

    # Subclase A: obstáculo al frente, derecha libre, izquierda bloqueada
    for _ in range(N_DERECHA // 2):
        dist = uni(DISTANCIA_FRENADO + 1, DISTANCIA_SEGURA - 1)
        desv = uni(-0.50, 0.50)
        filas.append([norm(dist), desv_ruido(desv), 0, 1, 2])

    # Subclase B: camino libre pero carro desviado a la izquierda → corregir der
    for _ in range(N_DERECHA // 2):
        dist = uni(DISTANCIA_SEGURA + 1, DISTANCIA_MAX_NN)
        desv = uni(-1.0, -0.16)  # desviación negativa = desviado a la izquierda
        filas.append([norm(dist), desv_ruido(desv), 1, 1, 2])

    # ── CLASE 3: retroceder ───────────────────────────────────

    # Subclase A: obstáculo muy cercano (zona de frenado de emergencia)
    for _ in range(N_RETROCEDER // 2):
        dist = uni(2.0, DISTANCIA_FRENADO)
        desv = uni(-0.50, 0.50)
        filas.append([norm(dist), desv_ruido(desv), 1, 1, 3])

    # Subclase B: obstáculo frontal medio Y ambos lados bloqueados → sin escape
    for _ in range(N_RETROCEDER // 2):
        dist = uni(DISTANCIA_FRENADO, DISTANCIA_SEGURA - 1)
        desv = uni(-0.50, 0.50)
        filas.append([norm(dist), desv_ruido(desv), 0, 0, 3])

    # ── MUESTRAS AMBIGUAS (10%) ───────────────────────────────
    # Casos donde izquierda o derecha son igualmente válidas.
    # Se asigna una de las dos al azar para reflejar la incertidumbre real.
    # Esto evita que la MLP alcance 100% de accuracy (señal de memorización).
    n_ambig = int(len(filas) * 0.10)
    for _ in range(n_ambig):
        dist  = uni(DISTANCIA_FRENADO + 1, DISTANCIA_SEGURA - 1)
        desv  = uni(-0.10, 0.10)
        clase = random.choice([1, 2])   # izquierda o derecha, ambas válidas
        filas.append([norm(dist), desv_ruido(desv), 1, 1, clase])

    # ── Mezclar para eliminar sesgo de orden ──────────────────
    random.shuffle(filas)
    return filas


# ─────────────────────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────

def main() -> None:
    os.makedirs("data", exist_ok=True)
    filas = generar()

    with open(OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dist_frente_norm", "desviacion",
                         "izq_libre", "der_libre", "accion"])
        writer.writerows(filas)

    total  = len(filas)
    clases = ["avanzar", "izquierda", "derecha", "retroceder"]
    print(f"\n[GEN] Dataset generado: {OUTPUT}")
    print(f"      Total muestras : {total}")
    print(f"      Distribución por clase:")
    for i, nombre in enumerate(clases):
        n = sum(1 for r in filas if r[-1] == i)
        print(f"        Clase {i} ({nombre:12s}): {n:>5} ({n/total*100:.1f}%)")
    print("\n  Siguiente paso: python utils/train_model.py\n")


if __name__ == "__main__":
    main()
