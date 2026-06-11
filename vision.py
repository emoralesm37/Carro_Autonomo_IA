# vision.py
"""
Módulo de visión por computadora del carro autónomo.

Hardware: Raspberry Pi NoIR Camera Module
Librería: picamera2 (Bookworm / libcamera stack) + OpenCV

Para habilitar la cámara en Bookworm:
  sudo raspi-config → Interface Options → Camera → Enable
  O agregar en /boot/firmware/config.txt:
    camera_auto_detect=1

Capacidades:
  1. Detección de carril  — líneas blancas/amarillas (Canny + Hough)
  2. Detección de semáforo — rojo / verde por rango HSV
  3. Detección señal STOP — octágono rojo por contorno
  4. Provisión de frames JPEG para el servidor de streaming

⚠️  NOTA NoIR Camera:
  Sin filtro IR, los rangos HSV pueden diferir bajo luz artificial
  o LEDs IR. Calibrar antes del primer uso:
    python utils/calibrate_hsv.py --color rojo
    python utils/calibrate_hsv.py --color verde
"""

import threading
import time

import cv2
import numpy as np
from picamera2 import Picamera2

from config import (
    CAM_ANCHO, CAM_ALTO, CAM_FPS,
    ROI_CARRIL_INICIO,
    HSV_ROJO_BAJO_1, HSV_ROJO_ALTO_1,
    HSV_ROJO_BAJO_2, HSV_ROJO_ALTO_2,
    HSV_VERDE_BAJO,  HSV_VERDE_ALTO,
    HSV_UMBRAL_PIX,
    STREAM_CALIDAD,
)


class Vision:
    """
    Pipeline de visión ejecutado en hilo daemon independiente.
    El hilo actualiza los atributos de estado continuamente;
    el bucle principal los lee de forma thread-safe.

    Atributos públicos (acceder via leer_estado()):
        semaforo       str   "rojo" | "verde" | "desconocido"
        stop_detectado bool  True si se detecta señal STOP
        desviacion     float -1.0 (izq) → 0.0 (centro) → +1.0 (der)
    """

    def __init__(self):
        # Inicializar cámara con picamera2
        self._cam = Picamera2()
        cfg = self._cam.create_preview_configuration(
            main={"size": (CAM_ANCHO, CAM_ALTO), "format": "RGB888"},
            controls={"FrameRate": CAM_FPS}
        )
        self._cam.configure(cfg)
        self._cam.start()
        time.sleep(1.5)   # Warm-up: la cámara ajusta exposición automáticamente

        # Estado de detección (actualizado en hilo de visión)
        self.semaforo       = "desconocido"
        self.stop_detectado = False
        self.desviacion     = 0.0
        self.frame_anotado  = None

        # Lock para acceso concurrente thread-safe
        self._lock   = threading.Lock()
        self._activo = True

        # Hilo daemon: muere automáticamente si muere el proceso principal
        self._hilo = threading.Thread(target=self._loop, daemon=True)
        self._hilo.start()

        print("[VIS] Cámara picamera2 inicializada")

    # ─────────────────────────────────────────────────────────
    # LOOP INTERNO (hilo daemon)
    # ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Captura y procesa frames de forma continua (~20 fps)."""
        while self._activo:
            try:
                self._procesar()
            except Exception as e:
                print(f"[VIS] Error en loop: {e}")
            time.sleep(0.05)

    def _procesar(self) -> None:
        """
        Captura un frame y ejecuta el pipeline completo:
          1. Detectar semáforo
          2. Detectar señal STOP
          3. Detectar carril y calcular desviación
          4. Anotar resultado en el frame para streaming
        """
        # picamera2 devuelve formato RGB → convertir a BGR para OpenCV
        frame_rgb = self._cam.capture_array()
        frame     = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        semaforo       = self._detectar_semaforo(frame)
        stop           = self._detectar_stop(frame)
        desv, ann      = self._detectar_carril(frame)

        # ── Overlay de estado sobre el frame ─────────────────
        color_sem = {"verde": (0, 200, 0), "rojo": (0, 0, 220)}.get(
            semaforo, (160, 160, 160))

        cv2.putText(ann, f"Semaforo: {semaforo}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_sem, 2)
        cv2.putText(ann, f"STOP: {'SI' if stop else 'no'}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 0, 220) if stop else (160, 160, 160), 2)
        cv2.putText(ann, f"Desv: {desv:+.2f}",
                    (10, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 220), 2)

        with self._lock:
            self.semaforo       = semaforo
            self.stop_detectado = stop
            self.desviacion     = desv
            self.frame_anotado  = ann

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE CARRIL
    # ─────────────────────────────────────────────────────────

    def _detectar_carril(self, frame) -> tuple:
        """
        Detecta las líneas del carril en la mitad inferior del frame.

        Algoritmo:
          1. ROI: porción inferior (ROI_CARRIL_INICIO al 100% en Y)
          2. Escala de grises + blur → Canny (bordes)
          3. HoughLinesP → segmentos de línea
          4. Clasificar por pendiente: izquierda (pend < 0) / derecha (pend > 0)
          5. Calcular centro del carril y desviación respecto al centro del frame

        Retorna:
            (desviacion: float, frame_anotado: ndarray)
            desviacion > 0 → carro desviado a la derecha → corregir izquierda
            desviacion < 0 → carro desviado a la izquierda → corregir derecha
        """
        h, w  = frame.shape[:2]
        roi_y = int(h * ROI_CARRIL_INICIO)
        roi   = frame[roi_y:h, 0:w]

        gris   = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur   = cv2.GaussianBlur(gris, (5, 5), 0)
        bordes = cv2.Canny(blur, 50, 150)

        lineas = cv2.HoughLinesP(
            bordes,
            rho=1,
            theta=np.pi / 180,
            threshold=30,
            minLineLength=40,
            maxLineGap=100
        )

        ann = frame.copy()
        if lineas is None:
            return 0.0, ann

        cx = w / 2.0
        pts_izq, pts_der = [], []

        for seg in lineas:
            x1, y1, x2, y2 = seg[0]
            if x2 == x1:
                continue                       # vertical pura → ignorar
            pend   = (y2 - y1) / (x2 - x1)
            if abs(pend) < 0.3:
                continue                       # casi horizontal → ignorar
            medio_x = (x1 + x2) / 2.0
            roi_ann = ann[roi_y:h, 0:w]
            if pend < 0 and medio_x < cx:     # carril izquierdo
                pts_izq.append((x1, y1, x2, y2))
                cv2.line(roi_ann, (x1, y1), (x2, y2), (0, 255, 0), 2)
            elif pend > 0 and medio_x >= cx:  # carril derecho
                pts_der.append((x1, y1, x2, y2))
                cv2.line(roi_ann, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Posición X promedio de cada carril detectado
        def x_promedio(pts):
            if not pts:
                return None
            xs = [p[0] for p in pts] + [p[2] for p in pts]
            return int(np.mean(xs))

        x_izq = x_promedio(pts_izq)
        x_der = x_promedio(pts_der)

        if x_izq is not None and x_der is not None:
            centro_carril = (x_izq + x_der) / 2.0
        elif x_izq is not None:
            centro_carril = x_izq + w * 0.25   # estimar centro
        elif x_der is not None:
            centro_carril = x_der - w * 0.25
        else:
            return 0.0, ann

        # Desviación normalizada: -1.0 … +1.0
        desv = float(np.clip((centro_carril - cx) / cx, -1.0, 1.0))

        # Dibujar líneas de referencia sobre el frame original
        roi_ann = ann[roi_y:h, 0:w]
        cv2.line(roi_ann, (int(centro_carril), 0),
                 (int(centro_carril), h - roi_y), (0, 0, 255), 2)   # centro carril
        cv2.line(roi_ann, (int(cx), 0),
                 (int(cx), h - roi_y), (255, 255, 0), 1)             # centro frame

        return desv, ann

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE SEMÁFORO
    # ─────────────────────────────────────────────────────────

    def _detectar_semaforo(self, frame) -> str:
        """
        Detecta el color del semáforo en la mitad superior del frame
        usando segmentación HSV.

        Rojo usa dos rangos (el tono rojo cruza 0/180 en el espacio HSV).
        Se aplica apertura morfológica para eliminar ruido.

        Retorna: "rojo" | "verde" | "desconocido"
        """
        h   = frame.shape[0]
        roi = frame[0:int(h * 0.5), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k   = np.ones((5, 5), np.uint8)

        # Máscara roja (dos rangos)
        m_r1 = cv2.inRange(hsv,
                           np.array(HSV_ROJO_BAJO_1), np.array(HSV_ROJO_ALTO_1))
        m_r2 = cv2.inRange(hsv,
                           np.array(HSV_ROJO_BAJO_2), np.array(HSV_ROJO_ALTO_2))
        mask_rojo  = cv2.morphologyEx(cv2.bitwise_or(m_r1, m_r2),
                                       cv2.MORPH_OPEN, k)

        # Máscara verde
        mask_verde = cv2.inRange(hsv,
                                 np.array(HSV_VERDE_BAJO), np.array(HSV_VERDE_ALTO))
        mask_verde = cv2.morphologyEx(mask_verde, cv2.MORPH_OPEN, k)

        px_r = cv2.countNonZero(mask_rojo)
        px_v = cv2.countNonZero(mask_verde)

        if px_r > HSV_UMBRAL_PIX and px_r > px_v:
            return "rojo"
        if px_v > HSV_UMBRAL_PIX and px_v > px_r:
            return "verde"
        return "desconocido"

    # ─────────────────────────────────────────────────────────
    # DETECCIÓN DE SEÑAL STOP
    # ─────────────────────────────────────────────────────────

    def _detectar_stop(self, frame) -> bool:
        """
        Detecta la señal STOP mediante su forma octagonal roja.

        Algoritmo:
          1. ROI central del frame (evita bordes con ruido)
          2. Segmentación HSV del color rojo
          3. Cierre morfológico para consolidar la forma
          4. Buscar contornos y aproximar polígono
          5. Verificar: 6–9 lados Y circularidad > 0.6 (octágono)

        Retorna True si se detecta con suficiente confianza.
        """
        h, w   = frame.shape[:2]
        roi    = frame[int(h * 0.15):int(h * 0.75),
                       int(w * 0.10):int(w * 0.90)]
        hsv    = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        m1   = cv2.inRange(hsv, np.array(HSV_ROJO_BAJO_1), np.array(HSV_ROJO_ALTO_1))
        m2   = cv2.inRange(hsv, np.array(HSV_ROJO_BAJO_2), np.array(HSV_ROJO_ALTO_2))
        mask = cv2.morphologyEx(cv2.bitwise_or(m1, m2),
                                cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

        contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                         cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contornos:
            area = cv2.contourArea(cnt)
            if area < 500:                            # objeto demasiado pequeño
                continue
            perim = cv2.arcLength(cnt, True)
            aprox = cv2.approxPolyDP(cnt, 0.04 * perim, True)
            if 6 <= len(aprox) <= 9:                  # octágono aproximado
                circularidad = 4 * np.pi * area / (perim ** 2)
                if circularidad > 0.6:
                    return True
        return False

    # ─────────────────────────────────────────────────────────
    # ACCESO EXTERNO (thread-safe)
    # ─────────────────────────────────────────────────────────

    def leer_estado(self) -> tuple:
        """
        Retorna una copia thread-safe del estado actual de visión.
        Retorna: (semaforo: str, stop_detectado: bool, desviacion: float)
        """
        with self._lock:
            return self.semaforo, self.stop_detectado, self.desviacion

    def get_jpeg(self, calidad: int = STREAM_CALIDAD) -> bytes | None:
        """
        Retorna el último frame anotado codificado como JPEG.
        Usado por el servidor de streaming.
        """
        with self._lock:
            f = self.frame_anotado
        if f is None:
            return None
        _, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, calidad])
        return buf.tobytes()

    # ─────────────────────────────────────────────────────────
    # LIMPIEZA
    # ─────────────────────────────────────────────────────────

    def cerrar(self) -> None:
        """Detiene el hilo de visión y libera la cámara."""
        self._activo = False
        self._hilo.join(timeout=2)
        self._cam.stop()
        print("[VIS] Cámara detenida")
