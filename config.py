# config.py
"""
Configuración centralizada del carro autónomo.
Modifica aquí todos los parámetros sin tocar el código principal.
"""

# ─────────────────────────────────────────────────────────────
# PINES GPIO (numeración BCM)
# ─────────────────────────────────────────────────────────────

# Motor izquierdo (L298N — lado A)
PIN_M_IZQ_IN1 = 17      # Dirección adelante
PIN_M_IZQ_IN2 = 18      # Dirección atrás
PIN_M_IZQ_ENA = 12      # Enable / PWM velocidad  ← hardware PWM

# Motor derecho (L298N — lado B)
PIN_M_DER_IN3 = 27      # Dirección adelante
PIN_M_DER_IN4 = 22      # Dirección atrás
PIN_M_DER_ENB = 13      # Enable / PWM velocidad  ← hardware PWM
# Habilitar hardware PWM: agregar en /boot/firmware/config.txt →
#   dtoverlay=pwm-2chan,pin=12,func=4,pin2=13,func2=4

# HC-SR04 — Sensor ultrasónico
#  ADVERTENCIA ELÉCTRICA: ECHO emite 5 V pero el GPIO del RPi3
#   soporta solo 3.3 V. Usa un divisor de voltaje antes del pin ECHO:
#     ECHO (HC-SR04) → R1=1 kΩ → nodo → GPIO 24
#                                nodo → R2=2 kΩ → GND
PIN_TRIG = 23
PIN_ECHO = 24

# Servo de escaneo (SG90 o similar)
PIN_SERVO = 25

# ─────────────────────────────────────────────────────────────
# PARÁMETROS DE MOVIMIENTO
# ─────────────────────────────────────────────────────────────
VELOCIDAD_BASE     = 0.60   # 0.0 – 1.0 (escala gpiozero)
VELOCIDAD_GIRO     = 0.45
VELOCIDAD_CURVA    = 0.38   # Rueda interior en corrección de carril

TIEMPO_GIRO        = 0.30   # Segundos por giro en sitio
TIEMPO_RETROCESO   = 0.40   # Segundos de retroceso al evitar obstáculo

# ─────────────────────────────────────────────────────────────
# PARÁMETROS DE DISTANCIA (cm)
# ─────────────────────────────────────────────────────────────
DISTANCIA_SEGURA   = 25.0   # Frente libre  → avanzar normalmente
DISTANCIA_FRENADO  = 12.0   # Override de emergencia → retroceder
DISTANCIA_MAX_NN   = 60.0   # Límite de normalización para la NN

# ─────────────────────────────────────────────────────────────
# SERVO — ÁNGULOS DE ESCANEO
# ─────────────────────────────────────────────────────────────
SERVO_CENTRO       = 90     # Mirando al frente
SERVO_IZQUIERDA    = 135    # 45° a la izquierda
SERVO_DERECHA      = 45     # 45° a la derecha

# ─────────────────────────────────────────────────────────────
# CÁMARA
# ─────────────────────────────────────────────────────────────
CAM_ANCHO          = 640
CAM_ALTO           = 480
CAM_FPS            = 30
ROI_CARRIL_INICIO  = 0.55   # Fracción vertical donde empieza ROI de carril

# ─────────────────────────────────────────────────────────────
# VISIÓN — RANGOS HSV
# ─────────────────────────────────────────────────────────────
#   NOTA NoIR: la cámara sin filtro IR puede alterar los tonos
#     bajo iluminación artificial o IR. Si los valores no funcionan,
#     calibra con:  python utils/calibrate_hsv.py --color rojo
#                   python utils/calibrate_hsv.py --color verde

# Rojo (dos rangos porque el rojo cruza el límite 0/180 en HSV)
HSV_ROJO_BAJO_1  = (  0, 120,  70)
HSV_ROJO_ALTO_1  = ( 10, 255, 255)
HSV_ROJO_BAJO_2  = (170, 120,  70)
HSV_ROJO_ALTO_2  = (180, 255, 255)

# Verde semáforo
HSV_VERDE_BAJO   = ( 40,  50,  50)
HSV_VERDE_ALTO   = ( 90, 255, 255)

# Píxeles mínimos detectados para confirmar color (reduce falsos positivos)
HSV_UMBRAL_PIX   = 300

# ─────────────────────────────────────────────────────────────
# RED NEURONAL
# ─────────────────────────────────────────────────────────────
RUTA_MODELO      = "models/modelo.pkl"
RUTA_SCALER      = "models/scaler.pkl"
ACCIONES         = ["avanzar", "izquierda", "derecha", "retroceder"]

# ─────────────────────────────────────────────────────────────
# STREAMING FLASK (video en tiempo real por WiFi)
# ─────────────────────────────────────────────────────────────
STREAM_HOST      = "0.0.0.0"   # Escuchar en todas las interfaces
STREAM_PORT      = 5000
STREAM_FPS       = 20          # Frames por segundo del stream MJPEG
STREAM_CALIDAD   = 70          # Calidad JPEG 0–100
