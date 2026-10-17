"""
Módulo de Configuración Central
Define las variables globales, brokers MQTT, parámetros de simulación y umbrales agronómicos.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Conexión MQTT
MQTT_BROKER = os.getenv("MQTT_BROKER", "broker.emqx.io")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC_PREFIX = os.getenv("MQTT_TOPIC_PREFIX", "agricultura")
MQTT_TOPIC_SUB = os.getenv("MQTT_TOPIC_SUB", "agricultura/#")
MQTT_TOPIC_PRED = os.getenv("MQTT_TOPIC_PRED", "agricultura/predicciones")

# Variables objetivo
TARGETS = [
    "temperature_air",
    "humidity",
    "precipitation",
    "soil_ph"
]

TARGET_LABELS = {
    "temperature_air": "Temperatura (°C)",
    "humidity": "Humedad (%)",
    "precipitation": "Precipitación (mm)",
    "soil_ph": "pH Suelo"
}

# Definición de parcelas monitoreadas
PARCELAS = [
    {"id": "parcela_cafe_1", "crop": "coffee", "label": "Café - Parcela 1"},
    {"id": "parcela_cafe_2", "crop": "coffee", "label": "Café - Parcela 2"},
    {"id": "parcela_cafe_3", "crop": "coffee", "label": "Café - Parcela 3"},
    {"id": "parcela_cafe_4", "crop": "coffee", "label": "Café - Parcela 4"},
    {"id": "parcela_palma_1", "crop": "oil palm", "label": "Palma - Parcela 1"},
    {"id": "parcela_palma_2", "crop": "oil palm", "label": "Palma - Parcela 2"},
    {"id": "parcela_palma_3", "crop": "oil palm", "label": "Palma - Parcela 3"},
    {"id": "parcela_palma_4", "crop": "oil palm", "label": "Palma - Parcela 4"}
]

PARCELAS_ORDEN = [p["id"] for p in PARCELAS]

# Umbrales agronómicos críticos según diseño técnico del cultivo
UMBRALES_AGRONOMICOS = {
    "coffee": {
        "temperature_air": {"optimo": (18.0, 22.0), "critico_inf": 15.0, "critico_sup": 25.0},
        "humidity": {"optimo": (70.0, 80.0), "critico_inf": 60.0, "critico_sup": 85.0},
        "precipitation": {"optimo": (1500.0, 1800.0), "critico_inf": 1000.0, "critico_sup": 2200.0},
        "soil_ph": {"optimo": (5.0, 5.5), "critico_inf": 4.5, "critico_sup": 6.0}
    },
    "oil palm": {
        "temperature_air": {"optimo": (25.0, 28.0), "critico_inf": 21.0, "critico_sup": 32.0},
        "humidity": {"optimo": (75.0, 95.0), "critico_inf": 65.0, "critico_sup": None},
        "precipitation": {"optimo": (2000.0, 3000.0), "critico_inf": 1600.0, "critico_sup": None},
        "soil_ph": {"optimo": (4.5, 6.0), "critico_inf": 4.0, "critico_sup": 6.5}
    }
}

# Estilos visuales del dashboard
COLORES_DASHBOARD = {
    "real": "#00FFCC",
    "pred": "#FF9500",
    "mae": "#FFD700",
    "umbral": "#FF4444",
    "bg": "#0d0d1a"
}

# Parámetros para Online ML
MAX_PUNTOS = 120
MAX_ERROR_HISTORY = 250
N_WARM = 10
ANOMALY_THRESHOLD = 3.0

GB_RETRAIN_EVERY = 60
GB_MIN_BUFFER = 80
GB_BUFFER_SIZE = 400

ONLINE_WEIGHT_INIT = 0.85
GB_WEIGHT_INIT = 0.15
