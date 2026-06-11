# hardware.py
"""
Control de hardware del carro autónomo.
Librería: gpiozero con backend lgpio (nativo en Bookworm).

Componentes gestionados:
  - 2× Motorreductor DC via módulo L298N
  - Servomotor SG90 para escaneo lateral
  - Sensor ultrasónico HC-SR04

Esquema de conexión L298N:
  ┌─────────────┬────────────┐
  │ L298N pin   │ RPi GPIO   │
  ├─────────────┼────────────┤
  │ ENA         │ GPIO 12    │  ← hardware PWM
  │ IN1         │ GPIO 17    │
  │ IN2         │ GPIO 18    │
  │ ENB         │ GPIO 13    │  ← hardware PWM
  │ IN3         │ GPIO 27    │
  │ IN4         │ GPIO 22    │
  │ GND         │ GND RPi    │
  │ VIN         │ Batería 6–12 V (SEPARADA del RPi)  │
  └─────────────┴────────────┘

Esquema HC-SR04 (divisor de voltaje obligatorio en ECHO):
  ECHO(5V) → R1=1kΩ → nodo → GPIO 24
                       nodo → R2=2kΩ → GND
  TRIG     → GPIO 23  (3.3 V es suficiente)
  VCC      → Pin 5V del RPi
  GND      → GND RPi

Servo SG90:
  Signal → GPIO 25
  VCC    → Pin 5V
  GND    → GND
"""

import time

from gpiozero import PWMOutputDevice, OutputDevice, AngularServo, DistanceSensor
from gpiozero.pins.lgpio import LGPIOFactory
from gpiozero import Device

from config import (
    PIN_M_IZQ_IN1, PIN_M_IZQ_IN2, PIN_M_IZQ_ENA,
    PIN_M_DER_IN3, PIN_M_DER_IN4, PIN_M_DER_ENB,
    PIN_TRIG, PIN_ECHO, PIN_SERVO,
    VELOCIDAD_BASE, VELOCIDAD_GIRO, VELOCIDAD_CURVA,
    TIEMPO_GIRO, TIEMPO_RETROCESO,
    SERVO_CENTRO, SERVO_IZQUIERDA, SERVO_DERECHA,
)

# Forzar backend lgpio (nativo en Bookworm; no requiere instalación extra)
Device.pin_factory = LGPIOFactory()


# ─────────────────────────────────────────────────────────────
# CLASE AUXILIAR: Motor DC individual
# ─────────────────────────────────────────────────────────────

class MotorDC:
    """
    Controla un motor DC via L298N usando dos pines de dirección
    (IN1/IN2 o IN3/IN4) y un pin PWM de habilitación (ENA/ENB).
    """

    def __init__(self, pin_in1: int, pin_in2: int, pin_en: int):
        self._in1 = OutputDevice(pin_in1, active_high=True, initial_value=False)
        self._in2 = OutputDevice(pin_in2, active_high=True, initial_value=False)
        # PWMOutputDevice: frequency=1000 Hz para suavidad de giro
        self._en  = PWMOutputDevice(pin_en, frequency=1000, initial_value=0)

    def adelante(self, velocidad: float = VELOCIDAD_BASE) -> None:
        """Gira hacia adelante. velocidad: 0.0 – 1.0"""
        velocidad = float(max(0.0, min(1.0, velocidad)))
        self._in1.on()
        self._in2.off()
        self._en.value = velocidad

    def atras(self, velocidad: float = VELOCIDAD_BASE) -> None:
        """Gira hacia atrás. velocidad: 0.0 – 1.0"""
        velocidad = float(max(0.0, min(1.0, velocidad)))
        self._in1.off()
        self._in2.on()
        self._en.value = velocidad

    def parar(self) -> None:
        self._in1.off()
        self._in2.off()
        self._en.value = 0

    def cerrar(self) -> None:
        self.parar()
        self._en.close()
        self._in1.close()
        self._in2.close()


# ─────────────────────────────────────────────────────────────
# CLASE PRINCIPAL: Hardware
# ─────────────────────────────────────────────────────────────

class Hardware:
    """
    Interfaz unificada de alto nivel para todos los actuadores
    y sensores del carro.

    Uso típico:
        hw = Hardware()
        hw.avanzar()
        dist = hw.distancia_promedio()
        hw.cerrar()   # siempre llamar al terminar
    """

    def __init__(self):
        # ── Motores ──────────────────────────────────────────
        self.motor_izq = MotorDC(PIN_M_IZQ_IN1, PIN_M_IZQ_IN2, PIN_M_IZQ_ENA)
        self.motor_der = MotorDC(PIN_M_DER_IN3, PIN_M_DER_IN4, PIN_M_DER_ENB)

        # ── Servo (SG90 — pulsos 0.5 ms a 2.5 ms) ───────────
        self.servo = AngularServo(
            PIN_SERVO,
            min_angle=0,
            max_angle=180,
            min_pulse_width=0.0005,   # 500 µs  → 0°
            max_pulse_width=0.0025    # 2500 µs → 180°
        )
        self.servo.angle = SERVO_CENTRO
        time.sleep(0.4)  # estabilizar servo al centro

        # ── Sensor HC-SR04 ───────────────────────────────────
        # gpiozero.DistanceSensor maneja el timing con lgpio a nivel
        # de microsegundos (más preciso que time.time() en Python).
        # max_distance en metros → 4.0 m = 400 cm
        self.sensor = DistanceSensor(
            echo=PIN_ECHO,
            trigger=PIN_TRIG,
            max_distance=4.0
        )

        print("[HW] Hardware inicializado (gpiozero + lgpio)")

    # ─────────────────────────────────────────────────────────
    # COMANDOS DE MOVIMIENTO
    # ─────────────────────────────────────────────────────────

    def avanzar(self, v: float = VELOCIDAD_BASE) -> None:
        """Ambos motores al frente a la misma velocidad."""
        self.motor_izq.adelante(v)
        self.motor_der.adelante(v)

    def retroceder(self, v: float = VELOCIDAD_BASE) -> None:
        """Ambos motores atrás a la misma velocidad."""
        self.motor_izq.atras(v)
        self.motor_der.atras(v)

    def girar_izquierda(self, v: float = VELOCIDAD_GIRO) -> None:
        """Giro en sitio: motor izq atrás, motor der adelante."""
        self.motor_izq.atras(v)
        self.motor_der.adelante(v)

    def girar_derecha(self, v: float = VELOCIDAD_GIRO) -> None:
        """Giro en sitio: motor izq adelante, motor der atrás."""
        self.motor_izq.adelante(v)
        self.motor_der.atras(v)

    def corregir_izquierda(self, v: float = VELOCIDAD_BASE) -> None:
        """
        Curva suave a la izquierda para corrección de carril.
        Rueda izquierda más lenta → carro vira hacia la izquierda.
        Usado cuando el carro se desvía a la derecha del carril.
        """
        self.motor_izq.adelante(VELOCIDAD_CURVA)
        self.motor_der.adelante(v)

    def corregir_derecha(self, v: float = VELOCIDAD_BASE) -> None:
        """
        Curva suave a la derecha para corrección de carril.
        Rueda derecha más lenta → carro vira hacia la derecha.
        Usado cuando el carro se desvía a la izquierda del carril.
        """
        self.motor_izq.adelante(v)
        self.motor_der.adelante(VELOCIDAD_CURVA)

    def detener(self) -> None:
        """Para ambos motores inmediatamente."""
        self.motor_izq.parar()
        self.motor_der.parar()

    # ─────────────────────────────────────────────────────────
    # SERVO
    # ─────────────────────────────────────────────────────────

    def servo_angulo(self, angulo: float) -> None:
        """Mueve el servo al ángulo indicado (0°–180°) y espera."""
        angulo = float(max(0.0, min(180.0, angulo)))
        self.servo.angle = angulo
        time.sleep(0.35)  # tiempo de estabilización mecánica

    # ─────────────────────────────────────────────────────────
    # SENSOR ULTRASÓNICO
    # ─────────────────────────────────────────────────────────

    def medir_distancia(self) -> float:
        """
        Retorna la distancia frontal en cm.
        gpiozero.DistanceSensor devuelve metros → multiplicar ×100.
        Retorna 0.0 si la lectura está fuera del rango válido (2–400 cm).
        """
        try:
            d_cm = self.sensor.distance * 100.0
            return round(d_cm, 1) if 2.0 <= d_cm <= 400.0 else 0.0
        except Exception:
            return 0.0

    def distancia_promedio(self, muestras: int = 3) -> float:
        """
        Promedia N lecturas para reducir ruido del HC-SR04.
        Descarta lecturas inválidas (0.0).
        """
        lecturas = []
        for _ in range(muestras):
            d = self.medir_distancia()
            if d > 0:
                lecturas.append(d)
            time.sleep(0.04)
        return round(sum(lecturas) / len(lecturas), 1) if lecturas else 0.0

    def escanear_lateral(self) -> tuple:
        """
        Gira el servo izquierda y derecha para detectar espacio libre.
        Solo se llama cuando hay obstáculo al frente (operación lenta ~2 s).

        Retorna:
            (izq_libre: bool, der_libre: bool)
            True  → lado despejado (distancia > umbral o sin lectura)
            False → lado bloqueado
        """
        UMBRAL_LATERAL = 30.0  # cm mínimos para considerar el lado libre

        # Escanear izquierda
        self.servo_angulo(SERVO_IZQUIERDA)
        d_izq = self.distancia_promedio(muestras=2)

        # Volver al centro para evitar interferencia cruzada
        self.servo_angulo(SERVO_CENTRO)
        time.sleep(0.1)

        # Escanear derecha
        self.servo_angulo(SERVO_DERECHA)
        d_der = self.distancia_promedio(muestras=2)

        # Volver al centro
        self.servo_angulo(SERVO_CENTRO)

        izq_libre = (d_izq > UMBRAL_LATERAL) or (d_izq == 0.0)
        der_libre = (d_der > UMBRAL_LATERAL) or (d_der == 0.0)

        return izq_libre, der_libre

    # ─────────────────────────────────────────────────────────
    # LIMPIEZA
    # ─────────────────────────────────────────────────────────

    def cerrar(self) -> None:
        """Libera todos los recursos GPIO. Llamar siempre al terminar."""
        self.detener()
        self.servo.close()
        self.sensor.close()
        self.motor_izq.cerrar()
        self.motor_der.cerrar()
        print("[HW] GPIO liberado")
