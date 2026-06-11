# streaming.py
"""
Servidor de video en tiempo real via WiFi (Flask MJPEG).

Acceder desde cualquier dispositivo en la misma red:
  http://<IP_RASPBERRY>:5000

La IP del RPi se imprime en consola al iniciar.
Para conocer la IP del RPi: hostname -I

El stream muestra el frame anotado con:
  - Estado del semáforo (color)
  - Indicador de señal STOP
  - Valor de desviación de carril
  - Líneas de carril detectadas (verde)
  - Línea de centro del carril (rojo) y centro del frame (amarillo)
"""

import socket
import threading
import time

from flask import Flask, Response, render_template_string

from config import STREAM_HOST, STREAM_PORT, STREAM_FPS, STREAM_CALIDAD


# ─────────────────────────────────────────────────────────────
# PLANTILLA HTML DEL VISOR WEB
# ─────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Carro Autónomo — Vista en Vivo</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0d1117;
      color: #e6edf3;
      font-family: 'Courier New', monospace;
      display: flex;
      flex-direction: column;
      align-items: center;
      min-height: 100vh;
      padding: 16px;
    }
    h1 { font-size: 1.15rem; color: #58a6ff; margin-bottom: 12px; }
    img {
      border: 2px solid #30363d;
      border-radius: 6px;
      max-width: 95vw;
    }
    .info {
      margin-top: 10px;
      font-size: 0.72rem;
      color: #8b949e;
    }
  </style>
</head>
<body>
  <h1>&#x1F697; Carro Autónomo &mdash; Stream en Vivo</h1>
  <img src="/video" alt="stream de cámara">
  <p class="info">MJPEG &bull; {{ fps }} fps &bull; 640&times;480 &bull; NoIR Camera</p>
</body>
</html>"""


# ─────────────────────────────────────────────────────────────
# CLASE PRINCIPAL
# ─────────────────────────────────────────────────────────────

class ServidorStreaming:
    """
    Transmite frames MJPEG desde el módulo Vision por WiFi.

    Uso:
        srv = ServidorStreaming(vision)
        srv.iniciar()    # lanza Flask en hilo daemon (no bloquea)
    """

    def __init__(self, vision_ref):
        self._vision = vision_ref
        self._delay  = 1.0 / STREAM_FPS

        self._app = Flask(__name__)
        self._app.add_url_rule('/',      'index',  self._index)
        self._app.add_url_rule('/video', 'video',  self._video)

    # ── Rutas Flask ───────────────────────────────────────────

    def _index(self):
        return render_template_string(_HTML, fps=STREAM_FPS)

    def _generar(self):
        """
        Generador infinito de partes MJPEG.
        Cada iteración entrega un frame JPEG encapsulado en
        multipart/x-mixed-replace, que el navegador actualiza
        automáticamente sin recargar la página.
        """
        while True:
            jpeg = self._vision.get_jpeg(calidad=STREAM_CALIDAD)
            if jpeg:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n'
                       + jpeg +
                       b'\r\n')
            time.sleep(self._delay)

    def _video(self):
        return Response(
            self._generar(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    # ── Inicio en hilo daemon ─────────────────────────────────

    def iniciar(self) -> None:
        """
        Inicia el servidor Flask en un hilo daemon.
        No bloquea el bucle principal de control del carro.
        """
        hilo = threading.Thread(
            target=lambda: self._app.run(
                host=STREAM_HOST,
                port=STREAM_PORT,
                threaded=True,
                use_reloader=False   # OBLIGATORIO: evita doble arranque en hilo
            ),
            daemon=True
        )
        hilo.start()

        # Mostrar URL de acceso
        try:
            ip = socket.gethostbyname(socket.gethostname())
        except Exception:
            ip = "<IP_DEL_RPI>"
        print(f"[NET] Stream activo → http://{ip}:{STREAM_PORT}")
        print(f"      Abrir en el navegador desde cualquier PC en la misma red WiFi.")
