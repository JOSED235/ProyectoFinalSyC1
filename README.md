# AgroIoT & Machine Learning: Sistema de Monitoreo Agroclimático en Tiempo Real

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![MQTT](https://img.shields.io/badge/Protocol-MQTT%20v5-660066?style=for-the-badge&logo=eclipsemosquitto&logoColor=white)](https://mqtt.org/)
[![InfluxDB](https://img.shields.io/badge/Database-InfluxDB%20v2-22ADF6?style=for-the-badge&logo=influxdb&logoColor=white)](https://www.influxdata.com/)
[![Node-RED](https://img.shields.io/badge/Gateway-Node--RED-8F0000?style=for-the-badge&logo=nodered&logoColor=white)](https://nodered.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

Plataforma integral de **Internet de las Cosas (IoT)** y **Machine Learning en Línea (Streaming Analytics)** diseñada para el monitoreo agroclimático de precisión en cultivos tropicales colombianos: **Café (*Coffea arabica*)** y **Palma de Aceite (*Elaeis guineensis*)**. 

El sistema cubre desde la telemetría distribuida de sensores y mensajería Publish/Subscribe (MQTT) hasta la ingesta en bases de datos de series de tiempo (InfluxDB), automatización de alertas agronómicas (Node-RED) e inferencia adaptativa en tiempo real mediante algoritmos de regresión online y boosting.

---

## Contexto del Proyecto y Problemática

El cambio climático y la variabilidad microclimática en regiones tropicales impactan directamente el rendimiento agrícola:
* **Café (*Coffea arabica*):** Requiere un equilibrio estricto de temperatura (18 °C – 22 °C), humedad (70% – 80%) y pH ácido (5.0 – 5.5). Temperaturas superiores a 25 °C provocan estrés térmico y aceleran de forma perjudicial la maduración del grano, deteriorando la calidad en taza.
* **Palma de Aceite (*Elaeis guineensis*):** Exige regímenes térmicos cálidos (25 °C – 28 °C) y alta precipitación. Caídas de temperatura por debajo de 21 °C paralizan el desarrollo foliar y la producción de racimos.

Este proyecto implementa una infraestructura telemática robusta para sensar, transmitir, alertar y predecir estas fluctuaciones antes de que generen pérdidas agronómicas irreparables.

---

## Arquitectura de la Solución

```text
┌───────────────────────┐
│  Sensores IoT         │ (8 Parcelas: 4 Café / 4 Palma)
│  (DHT22, pH, Lluvia)  │ ──> Publicación JSON
└──────────┬────────────┘
           │
           ▼ [Puerto 1883]
┌──────────────────────────────────────────────┐
│  Broker MQTT (Eclipse Mosquitto / EMQX)      │
│  Tópicos: agricultura/{cultivo}/{parcela_id} │
└──────────┬───────────────────────────┬───────┘
           │                           │
           │ (Subscripción)            │ (Subscripción Telemetría)
           ▼                           ▼
┌───────────────────────────┐ ┌──────────────────────────────────────┐
│  Gateway IoT (Node-RED)   │ │  Online Machine Learning Engine      │
│  - Limpieza & Filtro      │ │  - PassiveAggressive Regressor       │
│  - Umbrales Agronómicos   │ │  - HistGradientBoosting (Warm Replay)│
│  - Generación de Alertas  │ │  - Detección de Anomalías Dinámicas  │
└──────────┬────────────────┘ └──────────────────────────────────────┘
           │
           ▼
┌───────────────────────────┐
│  InfluxDB v2 (TSDB)       │
│  Bucket: agricultura-iot  │
└───────────────────────────┘
```

---

## Variables Sensadas y Umbrales Críticos

| Variable Agroclimática | Cultivo | Rango Óptimo | Umbral Crítico Inferior | Umbral Crítico Superior | Dispositivo Típico |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Temperatura del aire** | Café | 18 °C – 22 °C | < 15 °C | > 25 °C | DHT22 / BME280 |
| **Humedad relativa** | Café | 70% – 80% | < 60% | > 85% | Capacitivo ambiental |
| **Precipitación anual** | Café | 1500 – 1800 mm | < 1000 mm | > 2200 mm | Pluviómetro de balancín |
| **pH del suelo** | Café | 5.0 – 5.5 | < 4.5 | > 6.0 | Electrodo selectivo H+ |
| **Temperatura del aire** | Palma | 25 °C – 28 °C | < 21 °C | > 32 °C | Termistor NTC / SHT31 |
| **Humedad relativa** | Palma | > 75% | < 65% | N/A | Sensor higrómetro |
| **Precipitación anual** | Palma | > 2000 mm | < 1600 mm | N/A | Pluviómetro de pulsos |
| **Humedad del suelo** | Palma | 60% – 80% CC | < 45% CC | N/A | Sonda capacitiva FDR |

---

## Pipeline de Machine Learning

### 1. Modelado Batch Offline
Se compararon algoritmos de regresión supervisada para evaluar la correlación entre variables microclimáticas:
* **Regresión Lineal Múltiple:** Modelo base de interpretabilidad.
* **Random Forest Regressor:** Detección de dependencias no lineales y estimación de importancia de variables (*Feature Importance*).
* **Gradient Boosting Regressor:** Máximo rendimiento con minimización de $RMSE$ y $R^2 \approx 0.94$.

### 2. Online / Streaming Machine Learning (`src/online_monitor.py`)
Para streaming en vivo sobre MQTT se utiliza un esquema híbrido:
* **PassiveAggressiveRegressor (Online):** Actualización incremental punto a punto con latencia de sub-milisegundo.
* **HistGradientBoostingRegressor (Background Replay):** Reentrenamiento periódico sobre búfer deslizante (`buffer_replay`) para corregir deriva conceptual (*Concept Drift*).
* **Inferencia de Anomalías:** Cálculo dinámico de Z-Score ($|z| > 3.0$) sobre la distribución de error en tiempo real.

---

## Estructura del Repositorio

```text
├── .env.example                     # Plantilla de variables de entorno (MQTT, InfluxDB, etc.)
├── .gitignore                       # Filtro de dependencias, artefactos y entornos virtuales
├── docker-compose.yml               # Orquestación de Mosquitto, InfluxDB v2, Node-RED y Simulador
├── requirements.txt                 # Dependencias Python
├── data/
│   └── dataset_agroclimatico.csv    # Dataset base para simulación y análisis
├── docs/
│   └── TrabajoFinal_Siscom.pdf      # Informe técnico y diseño de ingeniería telemática
├── notebooks/
│   └── analisis_agroclimatico_ml.ipynb # Notebook interactivo (Fases 2 a 6)
├── docker/
│   └── Dockerfile.simulator         # Contenedor para el simulador de telemetría IoT
└── src/
    ├── __init__.py
    ├── config.py                    # Configuración central, tópicos y umbrales agronómicos
    ├── simulator.py                 # Publicador de telemetría MQTT multicultivo
    ├── batch_models.py              # Script de entrenamiento y comparativa offline
    └── online_monitor.py            # Dashboard y motor adaptativo de Online ML
```

---

## Despliegue y Ejecución

### Opción 1: Despliegue Completo con Docker Compose (Recomendado)

Inicia todos los servicios del ecosistema IoT (Broker MQTT, Base de Datos InfluxDB v2, Gateway Node-RED y Simulador de sensores):

```bash
docker compose up -d
```

Servicios levantados:
* **Node-RED UI:** [http://localhost:1880](http://localhost:1880)
* **InfluxDB Dashboard:** [http://localhost:8086](http://localhost:8086) (Org: `icesi`, Bucket: `agricultura-iot`)
* **Mosquitto MQTT:** `localhost:1883`

---

### Opción 2: Ejecución Local en Python

1. **Crear y activar entorno virtual:**
   ```bash
   python -m venv venv
   # En Windows:
   venv\Scripts\activate
   # En Linux/macOS:
   source venv/bin/activate
   ```

2. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ejecutar el simulador de sensores IoT:**
   ```bash
   python src/simulator.py
   ```

4. **Lanzar el Monitor de Online ML & Dashboard en Tiempo Real:**
   ```bash
   python src/online_monitor.py
   ```

5. **Ejecutar la comparativa offline de modelos ML:**
   ```bash
   python src/batch_models.py
   ```

---

## Créditos y Autoría

* **Autor:** Jose David Rodríguez Pinilla  
* **Programa:** Ingeniería Telemática – Universidad Icesi  
* **Asignatura:** Sistemas y Comunicaciones I  
* **Profesor:** Gonzalo Llano R., Ph.D.  
