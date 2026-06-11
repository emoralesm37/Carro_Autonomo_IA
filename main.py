#!/usr/bin/env python3
# main.py
"""
Orquestador principal del carro autónomo.
Plataforma: Raspberry Pi 3 Modelo B — Raspberry Pi OS Bookworm

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 FLUJO DE CONTROL (≈ 20 Hz)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  1. Leer estado de visión  (semáforo, STOP, desviación de carril)
  2. Medir distancia frontal con HC-SR04
  3. [OVERRIDE] Semáforo rojo o señal STOP → detener
  4. [OVERRIDE] Obstáculo de emergencia < DISTANCIA_FRENADO → retroceder
  5. Si dist < DISTANCIA_SEGURA → escanear lateral con servo
  6. Consultar red neuronal: (dist, desv, izq_libre, der_libre) → acción
  7. Ejecutar movimiento; aplicar corrección suave de carril al avanzar

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 PRECONDICIONES (ejecutar una sola vez antes de usar)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python utils/generate_dataset.py   → genera data/dataset.csv
  python utils/train_model.py        → genera models/modelo.pkl

  (Opcional, si los colores no se detectan bien con la NoIR)
  python utils/calibrate_hsv.py --color rojo
  python utils/calibrate_hsv.py --color verde
"""

import signal
import sys
import time

from config import (
    VELOCIDAD_BASE,
    TIEMPO_GIRO,
    TIEMPO_RETROCESO,
    DISTANCIA_SEGURA,
    DISTANCIA_FRENADO,
)
from hardware    import Hardware
from vision      import Vision
from neural_net  import RedNeuronal
from streaming   import ServidorStreaming


def main() -> None:
    print("=" * 55)
    print("  CARRO AUTÓNOMO — Raspberry Pi 3 — Bookworm")
    print("=" * 55)

    # ── Inicializar subsistemas ───────────────────────────────
    hw  = Hardware()
    vis = Vision()
    rn  = RedNeuronal()
    srv = ServidorStreaming(vis)

    # Iniciar streaming en hilo background (no bloquea)
    srv.iniciar()

    print("[MAIN] Esperando cámara y sensores (3 s)...\n")
    time.sleep(3)

    # ── Manejo limpio de Ctrl+C y SIGTERM ────────────────────
    def shutdown(sig, frame):
        print("\n[MAIN] Señal de parada recibida — apagando...")
        hw.detener()
        vis.cerrar()
        hw.cerrar()
        sys.exit(0)

    signal.signal(signal.SIGINT,  shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # ── Variables de estado entre ciclos ─────────────────────
    izq_libre = True    # Resultado del último escaneo lateral
    der_libre = True
    ciclo     = 0

    print("[MAIN] Iniciando navegación autónoma...\n")
    print(f"{'CICLO':>6}  {'SEMAFORO':12}  {'STOP':5}  "
          f"{'DIST (cm)':>9}  {'DESV':>6}  ACCION")
    print("-" * 65)

    # ── Bucle principal ───────────────────────────────────────
    while True:
        ciclo += 1

        # ── 1. Estado de visión (thread-safe) ─────────────────
        semaforo, stop, desv = vis.leer_estado()

        # ── 2. Distancia frontal ──────────────────────────────
        dist_frente = hw.distancia_promedio(muestras=2)

        print(f"{ciclo:>6}  {semaforo:12}  {'SI' if stop else 'no':5}  "
              f"{dist_frente:>9.1f}  {desv:>+.3f}", end="  ")

        # ── 3. OVERRIDE: semáforo rojo o señal STOP ───────────
        #    El carro se detiene completamente y espera que cambie.
        if semaforo == "rojo" or stop:
            hw.detener()
            print("⛔ DETENIDO (semaforo/STOP)")
            time.sleep(0.15)
            continue

        # ── 4. OVERRIDE de emergencia: obstáculo muy cercano ──
        #    Retrocede automáticamente sin consultar la NN.
        if 0 < dist_frente < DISTANCIA_FRENADO:
            hw.detener()
            time.sleep(0.08)
            hw.retroceder(VELOCIDAD_BASE)
            time.sleep(TIEMPO_RETROCESO)
            hw.detener()
            print("⚠️  EMERGENCIA retroceso")
            time.sleep(0.2)
            continue

        # ── 5. Escaneo lateral (solo cuando hay obstáculo frontal)
        #    Operación lenta (~2 s); se salta si el camino está libre.
        if 0 < dist_frente < DISTANCIA_SEGURA:
            hw.detener()
            izq_libre, der_libre = hw.escanear_lateral()
        else:
            # Camino despejado: asumir ambos lados libres para la NN
            izq_libre = True
            der_libre = True

        # ── 6. Decisión de la red neuronal ────────────────────
        accion, probs = rn.inferir(
            dist_frente=dist_frente,
            desviacion=desv,
            izq_libre=izq_libre,
            der_libre=der_libre,
        )
        print(f"🧠 {accion:12}  P:{[f'{p:.2f}' for p in probs]}")

        # ── 7. Ejecutar acción ────────────────────────────────
        if accion == "avanzar":
            # Corrección proporcional de carril mientras avanza:
            #   desv > +0.15 → desviado a la derecha → curva a la izquierda
            #   desv < -0.15 → desviado a la izquierda → curva a la derecha
            #   |desv| ≤ 0.15 → centrado → avanzar recto
            if desv > 0.15:
                hw.corregir_izquierda(VELOCIDAD_BASE)
            elif desv < -0.15:
                hw.corregir_derecha(VELOCIDAD_BASE)
            else:
                hw.avanzar(VELOCIDAD_BASE)

        elif accion == "izquierda":
            hw.girar_izquierda()
            time.sleep(TIEMPO_GIRO)
            hw.detener()
            time.sleep(0.15)

        elif accion == "derecha":
            hw.girar_derecha()
            time.sleep(TIEMPO_GIRO)
            hw.detener()
            time.sleep(0.15)

        elif accion == "retroceder":
            hw.retroceder(VELOCIDAD_BASE)
            time.sleep(TIEMPO_RETROCESO)
            hw.detener()
            time.sleep(0.15)

        # Pequeña pausa para ~20 ciclos/segundo
        time.sleep(0.05)


if __name__ == "__main__":
    main()
