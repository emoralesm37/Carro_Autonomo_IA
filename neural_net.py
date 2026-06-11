# neural_net.py
"""
Módulo de inferencia de la red neuronal (MLP).

Carga el modelo entrenado (scikit-learn MLPClassifier) y el
StandardScaler generados por utils/train_model.py.

Arquitectura del modelo:  4 entradas → 16 → 8 → 4 salidas (softmax)

Entradas (4 características):
  dist_frente_norm  float  distancia frontal normalizada [0.0 – 1.0]
  desviacion        float  desviación de carril          [-1.0 – +1.0]
  izq_libre         int    1 = izquierda despejada, 0 = bloqueada
  der_libre         int    1 = derecha  despejada, 0 = bloqueada

Salidas (4 clases):
  0 = avanzar | 1 = izquierda | 2 = derecha | 3 = retroceder

Precondición: ejecutar antes de usar main.py
  python utils/generate_dataset.py
  python utils/train_model.py
"""

import os

import joblib
import numpy as np

from config import RUTA_MODELO, RUTA_SCALER, DISTANCIA_MAX_NN, ACCIONES


class RedNeuronal:
    """
    Envuelve el MLPClassifier de scikit-learn para inferencia en tiempo real.

    Ejemplo:
        rn = RedNeuronal()
        accion, probs = rn.inferir(
            dist_frente=40.0, desviacion=0.1,
            izq_libre=True, der_libre=True
        )
        # → ("avanzar", array([0.85, 0.05, 0.07, 0.03]))
    """

    def __init__(self):
        if not os.path.exists(RUTA_MODELO) or not os.path.exists(RUTA_SCALER):
            raise FileNotFoundError(
                "\n[NN] Modelo no encontrado. Ejecuta primero:\n"
                "     python utils/generate_dataset.py\n"
                "     python utils/train_model.py\n"
            )
        self._modelo  = joblib.load(RUTA_MODELO)
        self._scaler  = joblib.load(RUTA_SCALER)
        print(f"[NN] Modelo cargado desde '{RUTA_MODELO}'")
        print(f"     Clases: {list(zip(range(4), ACCIONES))}")

    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _normalizar_dist(d: float) -> float:
        """
        Normaliza distancia a [0.0, 1.0].
        0.0 → obstáculo prácticamente tocando el sensor
        1.0 → distancia máxima (DISTANCIA_MAX_NN cm)
        """
        return float(np.clip(d / DISTANCIA_MAX_NN, 0.0, 1.0))

    def inferir(self,
                dist_frente: float,
                desviacion: float,
                izq_libre: bool,
                der_libre: bool) -> tuple:
        """
        Ejecuta la inferencia y retorna la acción recomendada.

        Args:
            dist_frente : distancia frontal medida por HC-SR04 (cm)
            desviacion  : desviación lateral del carril (-1.0 a +1.0)
            izq_libre   : True si el lado izquierdo está despejado
            der_libre   : True si el lado derecho está despejado

        Returns:
            accion      (str)        acción de movimiento
            probs       (np.ndarray) probabilidades por clase [shape=(4,)]
        """
        x = np.array([[
            self._normalizar_dist(dist_frente),
            float(np.clip(desviacion, -1.0, 1.0)),
            float(izq_libre),
            float(der_libre)
        ]])

        # Aplicar el mismo escalado usado durante el entrenamiento
        x_scaled = self._scaler.transform(x)

        idx   = int(self._modelo.predict(x_scaled)[0])
        probs = self._modelo.predict_proba(x_scaled)[0]

        return ACCIONES[idx], probs
