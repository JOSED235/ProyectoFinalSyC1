"""
Módulo de Simulación de Sensores IoT
Genera telemetría agroclimática realista basada en datasets de campo y publica
los datos hacia un broker MQTT (local o remoto).
"""

import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import paho.mqtt.client as mqtt

try:
    from .config import MQTT_BROKER, MQTT_PORT, PARCELAS
except ImportError:
    from config import MQTT_BROKER, MQTT_PORT, PARCELAS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("Simulator")


def cargar_dataset():
    """
    Carga el dataset de base climática para simulación.
    Intenta cargarlo localmente o desde la fuente en Google Drive / Sheets.
    """
    ruta_local = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_agroclimatico.csv")
    
    if os.path.exists(ruta_local):
        logger.info(f"Cargando dataset local desde {ruta_local}")
        df = pd.read_csv(ruta_local)
    else:
        sheet_id = "19qVUpzmgD6SwpefaWN2RTipjNm-dNPXKSPUw-t35bDw"
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
        logger.info("Descargando dataset desde Google Sheets...")
        df = pd.read_csv(csv_url)

    df["Crop_Type"] = df["Crop_Type"].str.lower()
    df = df[df["Crop_Type"].isin(["coffee", "oil palm"])]
    return df


def sensor_simulator(row, parcela):
    """
    Simula una lectura con ruido de sensor calibrado.
    """
    noise = np.random.normal(0, 0.3)
    return {
        "timestamp": datetime.now().isoformat(),
        "parcela": parcela["id"],
        "crop_type": parcela["crop"],
        "temperature_air": round(float(row["Temperature"]) + noise, 2),
        "humidity": round(float(row["Humidity"]) + noise, 2),
        "precipitation": round(max(0.0, float(row["Rainfall"]) + noise), 2),
        "soil_ph": round(float(row["Soil_pH"]) + (noise * 0.1), 2),
        "compatible": row.get("Compatible", True)
    }


def iniciar_simulacion():
    df = cargar_dataset()
    logger.info(f"Dataset cargado con {len(df)} registros para cultivos: café y palma de aceite.")

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)

    logger.info(f"Conectando al broker MQTT en {MQTT_BROKER}:{MQTT_PORT}...")
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
        client.loop_start()
        logger.info("Conexión MQTT establecida exitosamente.")
    except Exception as e:
        logger.error(f"Error conectando al broker MQTT: {e}")
        sys.exit(1)

    intervalo = float(os.getenv("SIMULATION_INTERVAL", 1.0))
    logger.info(f"Iniciando ciclo de publicación (intervalo={intervalo}s)...")

    try:
        while True:
            for _, row in df.iterrows():
                start_time = time.time()
                for parcela in PARCELAS:
                    if row["Crop_Type"] != parcela["crop"]:
                        continue

                    sensor_data = sensor_simulator(row, parcela)
                    crop_key = parcela["crop"].replace(" ", "_")
                    topic = f"agricultura/{crop_key}/{parcela['id']}"
                    payload_json = json.dumps(sensor_data)

                    result = client.publish(topic, payload_json)
                    if result.rc == 0:
                        logger.debug(f"Publicado en {topic}: {payload_json}")
                    else:
                        logger.warning(f"Fallo al publicar en {topic}")

                elapsed = time.time() - start_time
                time.sleep(max(0.0, intervalo - elapsed))

    except KeyboardInterrupt:
        logger.info("Simulación detenida por el usuario.")
    finally:
        client.loop_stop()
        client.disconnect()
        logger.info("Desconectado del broker MQTT.")


if __name__ == "__main__":
    iniciar_simulacion()
