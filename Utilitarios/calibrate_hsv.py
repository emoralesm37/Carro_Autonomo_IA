# utils/calibrate_hsv.py
"""
Herramienta de calibración de rangos HSV para la cámara NoIR.

La cámara NoIR (sin filtro infrarrojo) puede mostrar tonos HSV
diferentes a una cámara estándar bajo luz artificial o LEDs IR.
Este script permite seleccionar visualmente una región de color
y actualiza los rangos en config.py automáticamente.

Uso:
  python utils/calibrate_hsv.py --color rojo
  python utils/calibrate_hsv.py --color verde

Controles en la ventana:
  Clic y arrastrar → seleccionar región del color a calibrar
  's'              → guardar rangos detectados en config.py
  'r'              → reiniciar selección
  'q'              → salir sin guardar
"""

import argparse
import os
import re
import sys

import cv2
import numpy as np
from picamera2 import Picamera2

# Ajustar path para importar config
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ─────────────────────────────────────────────────────────────
# Estado de selección de ROI con el mouse
# ─────────────────────────────────────────────────────────────
_roi_ini   = None
_roi_fin   = None
_arrastr   = False


def _callback_mouse(event, x, y, flags, _):
    global _roi_ini, _roi_fin, _arrastr
    if event == cv2.EVENT_LBUTTONDOWN:
        _roi_ini = (x, y)
        _arrastr = True
    elif event == cv2.EVENT_MOUSEMOVE and _arrastr:
        _roi_fin = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        _roi_fin = (x, y)
        _arrastr = False


def _calcular_rango_hsv(frame_bgr) -> tuple | None:
    """
    Extrae rango HSV (con margen) de la región seleccionada.
    Retorna (bajo, alto) como tuplas de 3 enteros, o None si la
    selección es demasiado pequeña.
    """
    if _roi_ini is None or _roi_fin is None:
        return None
    x1 = min(_roi_ini[0], _roi_fin[0])
    y1 = min(_roi_ini[1], _roi_fin[1])
    x2 = max(_roi_ini[0], _roi_fin[0])
    y2 = max(_roi_ini[1], _roi_fin[1])
    if x2 - x1 < 5 or y2 - y1 < 5:
        return None

    recorte = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h_min, s_min, v_min = recorte.min(axis=(0, 1))
    h_max, s_max, v_max = recorte.max(axis=(0, 1))

    # Margen de tolerancia para variaciones de iluminación
    mh, ms, mv = 10, 35, 45
    bajo = (int(max(0,   h_min - mh)),
            int(max(0,   s_min - ms)),
            int(max(0,   v_min - mv)))
    alto = (int(min(180, h_max + mh)),
            int(min(255, s_max + ms)),
            int(min(255, v_max + mv)))
    return bajo, alto


def _actualizar_config(color: str, bajo: tuple, alto: tuple) -> None:
    """Reemplaza los rangos HSV correspondientes en config.py."""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config.py")
    with open(config_path, "r", encoding="utf-8") as f:
        txt = f.read()

    if color == "verde":
        txt = re.sub(r"HSV_VERDE_BAJO\s*=\s*\(.*?\)",
                     f"HSV_VERDE_BAJO   = {bajo}", txt)
        txt = re.sub(r"HSV_VERDE_ALTO\s*=\s*\(.*?\)",
                     f"HSV_VERDE_ALTO   = {alto}", txt)
    elif color == "rojo":
        # Actualiza el primer rango rojo (el más frecuente)
        txt = re.sub(r"HSV_ROJO_BAJO_1\s*=\s*\(.*?\)",
                     f"HSV_ROJO_BAJO_1  = {bajo}", txt)
        txt = re.sub(r"HSV_ROJO_ALTO_1\s*=\s*\(.*?\)",
                     f"HSV_ROJO_ALTO_1  = {alto}", txt)

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(txt)

    print(f"\n[CAL] config.py actualizado:")
    print(f"      HSV_{color.upper()}_BAJO = {bajo}")
    print(f"      HSV_{color.upper()}_ALTO = {alto}")
    print("      Reinicia main.py para aplicar los cambios.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibración de rangos HSV para la cámara NoIR")
    parser.add_argument("--color", choices=["rojo", "verde"], required=True)
    args = parser.parse_args()

    cam = Picamera2()
    cam.configure(cam.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"}))
    cam.start()

    titulo = (f"Calibracion HSV — {args.color}  "
              f"| Selecciona region → 's' guardar  'r' reiniciar  'q' salir")
    cv2.namedWindow(titulo, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(titulo, _callback_mouse)

    print(f"\n[CAL] Calibrando: {args.color}")
    print("      1. Haz clic y arrastra sobre la región del color")
    print("      2. Presiona 's' para guardar en config.py")
    print("      3. Presiona 'q' para salir sin guardar\n")

    rango_final = None

    while True:
        frame_rgb = cam.capture_array()
        frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Dibujar ROI en curso
        if _roi_ini and _roi_fin:
            cv2.rectangle(frame, _roi_ini, _roi_fin, (0, 255, 255), 2)
            rango = _calcular_rango_hsv(frame)
            if rango:
                rango_final = rango
                bajo, alto  = rango
                cv2.putText(frame, f"BAJO HSV: {bajo}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"ALTO HSV: {alto}",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 255), 2)

                # Mostrar máscara del color detectado (preview)
                hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array(bajo), np.array(alto))
                frame[mask > 0] = [0, 200, 200]   # colorear píxeles detectados

        cv2.imshow(titulo, frame)
        k = cv2.waitKey(1) & 0xFF

        if k == ord('s'):
            if rango_final:
                _actualizar_config(args.color, rango_final[0], rango_final[1])
            else:
                print("[CAL] Selecciona una región primero.")
            break
        elif k == ord('r'):
            globals().update(_roi_ini=None, _roi_fin=None)
            rango_final = None
        elif k == ord('q'):
            print("[CAL] Saliendo sin guardar.")
            break

    cam.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
