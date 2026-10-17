import json
import collections
import threading
import time
from datetime import datetime

import numpy as np
import paho.mqtt.client as mqtt

import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from matplotlib.widgets import Button
from matplotlib.gridspec import GridSpec

from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

# CONFIG

MQTT_BROKER     = "broker.emqx.io"
MQTT_PORT       = 1883
MQTT_TOPIC_SUB  = "agricultura/#"
MQTT_TOPIC_PRED = "agricultura/predicciones"

TARGETS = [
    "temperature_air",
    "humidity",
    "precipitation",
    "soil_ph"
]

TARGET_LBL = [
    "Temperatura (°C)",
    "Humedad (%)",
    "Precipitación (mm)",
    "pH Suelo"
]

PARCELAS_ORDEN = [
    "parcela_cafe_1",
    "parcela_cafe_2",
    "parcela_cafe_3",
    "parcela_cafe_4",
    "parcela_palma_1",
    "parcela_palma_2",
    "parcela_palma_3",
    "parcela_palma_4",
]

COLORES = {
    "real":      "#00FFCC",
    "pred":      "#FF9500",
    "mae":       "#FFD700",
    "umbral":    "#FF4444",
    "bg":        "#0d0d1a",
}

MAX_PUNTOS            = 120
MAX_ERROR_HISTORY     = 250

N_WARM                = 10
ANOMALY_THRESHOLD     = 3.0

# Hybrid ML
GB_RETRAIN_EVERY      = 60
GB_MIN_BUFFER         = 80
GB_BUFFER_SIZE        = 400

ONLINE_WEIGHT_INIT    = 0.85
GB_WEIGHT_INIT        = 0.15

# FEATURES

def features_ingenieria(hum, prec, ph, temp, hora):
    """
    Features mejoradas para estabilidad temporal.
    """

    hora_sin = np.sin(2 * np.pi * hora / 24)
    hora_cos = np.cos(2 * np.pi * hora / 24)

    hum_prec = hum * np.log1p(abs(prec) + 1)

    ph_sq = ph ** 2

    # Features MUY importantes para precipitación
    prec_log = np.log1p(abs(prec))
    hum_sq = hum ** 2

    return [[
        hum,
        prec,
        ph,
        temp,

        hora_sin,
        hora_cos,

        hum_prec,
        ph_sq,

        prec_log,
        hum_sq
    ]]

# MODELOS

def crear_modelo_online(target):
    """
    Modelo online especializado por variable.
    """

    if target == "precipitation":
        return PassiveAggressiveRegressor(
            C=0.08,
            max_iter=1,
            tol=None,
            loss="epsilon_insensitive",
            epsilon=0.02,
            shuffle=False,
            warm_start=True,
            average=25
        )

    return PassiveAggressiveRegressor(
        C=0.25,
        max_iter=1,
        tol=None,
        loss="epsilon_insensitive",
        epsilon=0.05,
        shuffle=False,
        warm_start=True,
        average=15
    )

def crear_modelo_gb(target):
    """
    Modelo batch de alta precisión.
    """

    if target == "precipitation":
        return HistGradientBoostingRegressor(
            max_iter=160,
            learning_rate=0.03,
            max_depth=6,
            min_samples_leaf=5,
            l2_regularization=0.2,
            random_state=42
        )

    return HistGradientBoostingRegressor(
        max_iter=120,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=6,
        l2_regularization=0.1,
        random_state=42
    )

# ESTRUCTURAS

modelos_online_parcelas = {}
modelos_online_cultivos = {}

modelos_gb_parcelas = {}
modelos_gb_cultivos = {}

scalers_parcelas = {}
scalers_cultivos = {}

stats_parcelas = {}
stats_cultivos = {}

buffer_replay = collections.defaultdict(
    lambda: collections.deque(maxlen=GB_BUFFER_SIZE)
)

_lock_model = threading.Lock()

# STATS

def crear_stats():

    stats = {}

    for t in TARGETS:
        stats[t] = {
            "errores": collections.deque(maxlen=MAX_ERROR_HISTORY),
            "n": 0
        }

    return stats

# INIT ENTIDAD

def _inicializar_entidad(entidad,
                         online_models,
                         gb_models,
                         scalers,
                         stats):

    online_models[entidad] = {
        t: crear_modelo_online(t)
        for t in TARGETS
    }

    gb_models[entidad] = {
        t: None
        for t in TARGETS
    }

    scalers[entidad] = {
        t: StandardScaler()
        for t in TARGETS
    }

    stats[entidad] = crear_stats()

# GB BACKGROUND TRAIN

def entrenar_gradient_boosting_background(entidad,
                                          online_models,
                                          gb_models,
                                          scalers):

    try:

        buffer_local = list(buffer_replay[entidad])

        if len(buffer_local) < GB_MIN_BUFFER:
            return

        for t in TARGETS:

            X_all = []
            y_all = []

            for hum, prec, ph, temp, hora in buffer_local:

                reales = {
                    "temperature_air": temp,
                    "humidity": hum,
                    "precipitation": prec,
                    "soil_ph": ph
                }

                otros = {
                    k: v for k, v in reales.items()
                    if k != t
                }

                X = features_ingenieria(
                    otros.get("humidity", hum),
                    otros.get("precipitation", prec),
                    otros.get("soil_ph", ph),
                    otros.get("temperature_air", temp),
                    hora
                )

                X_all.append(X[0])
                y_all.append(reales[t])

            scaler = scalers[entidad][t]

            X_sc = scaler.transform(X_all)

            gb = crear_modelo_gb(t)

            gb.fit(X_sc, y_all)

            with _lock_model:
                gb_models[entidad][t] = gb

        print(f"[GB] Reentrenado -> {entidad}")

    except Exception as e:
        print(f"Error GB background: {e}")

# PRED + TRAIN

def _pred_online(entidad,
                 X,
                 y_real,
                 target,
                 online_models,
                 gb_models,
                 scalers,
                 stats):

    scaler = scalers[entidad][target]

    scaler.partial_fit(X)

    X_sc = scaler.transform(X)

    modelo_online = online_models[entidad][target]

    # Warmup inicial

    n_seen = stats[entidad][target]["n"]

    if n_seen < 2:
        modelo_online.partial_fit(X_sc, [y_real])
        return float(y_real)

    # Online prediction

    pred_online = modelo_online.predict(X_sc)[0]

    # Gradient Boosting prediction

    pred_final = pred_online

    gb_model = gb_models[entidad][target]

    if gb_model is not None and n_seen > GB_MIN_BUFFER:

        try:

            pred_gb = gb_model.predict(X_sc)[0]

            mae_hist = stats[entidad][target]["errores"]

            mae_actual = np.mean(mae_hist) if len(mae_hist) > 5 else 999

            # Peso dinámico
            if mae_actual < 0.5:
                w_online = 0.45
                w_gb = 0.55
            else:
                w_online = ONLINE_WEIGHT_INIT
                w_gb = GB_WEIGHT_INIT

            pred_final = (
                pred_online * w_online
                +
                pred_gb * w_gb
            )

        except Exception:
            pass

    # Train online model

    modelo_online.partial_fit(X_sc, [y_real])

    return float(pred_final)

# MAIN PROCESS

def procesar_muestra(parcela,
                     cultivo,
                     hum,
                     prec,
                     ph,
                     temp,
                     hora):

    for entidad, online_mods, gb_mods, scalers, stats in [

        (
            parcela,
            modelos_online_parcelas,
            modelos_gb_parcelas,
            scalers_parcelas,
            stats_parcelas
        ),

        (
            cultivo,
            modelos_online_cultivos,
            modelos_gb_cultivos,
            scalers_cultivos,
            stats_cultivos
        )
    ]:

        if entidad not in online_mods:

            _inicializar_entidad(
                entidad,
                online_mods,
                gb_mods,
                scalers,
                stats
            )

    reales = {
        "temperature_air": temp,
        "humidity": hum,
        "precipitation": prec,
        "soil_ph": ph
    }

    preds_p = {}
    errs_p = {}
    maes_p = {}

    for t in TARGETS:

        y_real = reales[t]

        otros = {
            k: v for k, v in reales.items()
            if k != t
        }

        X = features_ingenieria(
            otros.get("humidity", hum),
            otros.get("precipitation", prec),
            otros.get("soil_ph", ph),
            otros.get("temperature_air", temp),
            hora
        )

        pred_p = _pred_online(
            parcela,
            X,
            y_real,
            t,
            modelos_online_parcelas,
            modelos_gb_parcelas,
            scalers_parcelas,
            stats_parcelas
        )

        err_p = abs(y_real - pred_p)

        stats_parcelas[parcela][t]["errores"].append(err_p)
        stats_parcelas[parcela][t]["n"] += 1

        preds_p[t] = pred_p
        errs_p[t] = err_p

        # MAE suavizado

        errores = list(stats_parcelas[parcela][t]["errores"])

        if len(errores) > 8:
            mae = np.mean(errores[-20:])
        else:
            mae = np.mean(errores)

        maes_p[t] = float(mae)

    # BUFFER

    buffer_replay[parcela].append(
        (hum, prec, ph, temp, hora)
    )

    n_total = stats_parcelas[parcela]["temperature_air"]["n"]

    # REENTRENAMIENTO GB

    if (
        n_total % GB_RETRAIN_EVERY == 0
        and
        len(buffer_replay[parcela]) >= GB_MIN_BUFFER
    ):

        threading.Thread(
            target=entrenar_gradient_boosting_background,
            args=(
                parcela,
                modelos_online_parcelas,
                modelos_gb_parcelas,
                scalers_parcelas
            ),
            daemon=True
        ).start()

    return preds_p, errs_p, maes_p

# BUFFERS VISUALES

_lock = threading.Lock()

def _nuevo_buffer():

    base = {
        "paso": 0,
        "nuevo": False
    }

    for t in TARGETS:

        base[f"t_{t}"] = collections.deque(maxlen=MAX_PUNTOS)

        base[f"real_{t}"] = collections.deque(maxlen=MAX_PUNTOS)

        base[f"pred_{t}"] = collections.deque(maxlen=MAX_PUNTOS)

        base[f"err_{t}"] = collections.deque(maxlen=MAX_PUNTOS)

        base[f"mae_{t}"] = collections.deque(maxlen=MAX_PUNTOS)

    return base

buffers = {
    p: _nuevo_buffer()
    for p in PARCELAS_ORDEN
}

# HELPERS

def extraer_hora(ts):

    try:
        dt = datetime.fromisoformat(ts)

    except Exception:
        dt = datetime.now()

    return dt.hour + dt.minute / 60.0

# MQTT

def on_message(client, userdata, msg):

    if "predicciones" in msg.topic:
        return

    try:

        d = json.loads(msg.payload.decode())

        parcela = d.get("parcela")
        cultivo = d.get("crop_type")

        hum = float(d.get("humidity"))
        prec = float(d.get("precipitation"))
        ph = float(d.get("soil_ph"))
        temp = float(d.get("temperature_air"))

        ts = d.get(
            "timestamp",
            datetime.now().isoformat()
        )

        if parcela not in PARCELAS_ORDEN:
            return

        hora = extraer_hora(ts)

        preds_p, errs_p, maes_p = procesar_muestra(
            parcela,
            cultivo,
            hum,
            prec,
            ph,
            temp,
            hora
        )

        n_p = stats_parcelas[parcela]["temperature_air"]["n"]

        err_temp = errs_p["temperature_air"]

        es_anomalia = (
            n_p > N_WARM
            and
            err_temp > ANOMALY_THRESHOLD
        )

        client.publish(
            f"{MQTT_TOPIC_PRED}/{parcela}",
            json.dumps({

                "parcela": parcela,
                "cultivo": cultivo,

                **{
                    f"pred_{t}": round(preds_p[t], 2)
                    for t in TARGETS
                },

                **{
                    f"error_{t}": round(errs_p[t], 2)
                    for t in TARGETS
                },

                **{
                    f"mae_{t}": round(maes_p[t], 2)
                    for t in TARGETS
                },

                "anomalia_detectada": bool(es_anomalia)

            }),
            qos=1
        )

        reales = {
            "temperature_air": temp,
            "humidity": hum,
            "precipitation": prec,
            "soil_ph": ph
        }

        with _lock:

            buf = buffers[parcela]

            buf["paso"] += 1

            for t in TARGETS:

                buf[f"t_{t}"].append(buf["paso"])

                buf[f"real_{t}"].append(reales[t])

                buf[f"pred_{t}"].append(preds_p[t])

                buf[f"err_{t}"].append(errs_p[t])

                buf[f"mae_{t}"].append(maes_p[t])

            buf["nuevo"] = True

        estado = "ANOMALIA" if es_anomalia else "OK"

        print(
            f"[{parcela}] "
            f"T:{temp:.2f} "
            f"H:{hum:.2f} "
            f"P:{prec:.2f} "
            f"pH:{ph:.2f} "
            f"| {estado}"
        )

    except Exception as e:
        print(f"Error on_message: {e}")

def on_connect(client, userdata, flags, rc, props):

    print(f"MQTT conectado: {rc}")

    client.subscribe(MQTT_TOPIC_SUB)

def on_disconnect(client, userdata, flags, rc, props):

    print("Broker desconectado...")

    while True:

        try:
            client.reconnect()
            print("Reconectado")
            break

        except Exception:
            time.sleep(5)

mqtt_client = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.on_disconnect = on_disconnect

mqtt_client.connect(
    MQTT_BROKER,
    MQTT_PORT,
    60
)

mqtt_client.loop_start()

# GRAFICAS

plt.style.use("dark_background")

fig = plt.figure(
    figsize=(18, 10),
    facecolor=COLORES["bg"]
)

fig.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.90,
    bottom=0.13,
    hspace=0.55,
    wspace=0.32
)

gs = GridSpec(
    2,
    3,
    figure=fig
)

ax_temp = fig.add_subplot(gs[0, 0])
ax_hum = fig.add_subplot(gs[0, 1])
ax_prec = fig.add_subplot(gs[0, 2])

ax_ph = fig.add_subplot(gs[1, 0])

ax_err = fig.add_subplot(gs[1, 1:])

AXES_MAP = {
    "temperature_air": ax_temp,
    "humidity": ax_hum,
    "precipitation": ax_prec,
    "soil_ph": ax_ph
}

def _init_ax(ax, ylabel, title):

    ax.set_facecolor(COLORES["bg"])

    ax.set_ylabel(
        ylabel,
        color="white",
        fontsize=8
    )

    ax.set_title(
        title,
        color="#aaaacc",
        fontsize=9
    )

    ax.tick_params(
        colors="white",
        labelsize=7
    )

    ax.grid(
        True,
        alpha=0.15
    )

    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")

for t, lbl in zip(TARGETS, TARGET_LBL):

    _init_ax(
        AXES_MAP[t],
        lbl,
        lbl
    )

_init_ax(
    ax_err,
    "MAE",
    "Evolución Error del Modelo"
)

# LINEAS

lineas_real = {}
lineas_pred = {}

for t in TARGETS:

    ax = AXES_MAP[t]

    lineas_real[t], = ax.plot(
        [],
        [],
        color=COLORES["real"],
        lw=2.0,
        label="Real"
    )

    lineas_pred[t], = ax.plot(
        [],
        [],
        color=COLORES["pred"],
        lw=1.8,
        ls="--",
        label="Pred"
    )

    ax.legend(
        loc="upper left",
        fontsize=7
    )

# MAE

lineas_mae = {}

MAE_COLORS = [
    "#FF9500",
    "#00FFCC",
    "#AA44FF",
    "#FF4488"
]

for t, col in zip(TARGETS, MAE_COLORS):

    lineas_mae[t], = ax_err.plot(
        [],
        [],
        color=col,
        lw=2,
        label=t.replace("_", " ")
    )

ax_err.legend(
    loc="upper right",
    fontsize=7
)

titulo = fig.suptitle(
    "",
    fontsize=12,
    color="#00FFCC",
    fontweight="bold"
)

# NAV

nav = {
    "idx": 0
}

def parcela_actual():
    return PARCELAS_ORDEN[nav["idx"]]

def actualizar_titulo():

    p = parcela_actual()

    cultivo = (
        "Café"
        if "cafe" in p
        else "Palma"
    )

    titulo.set_text(
        f"Hybrid ML IoT | {p} | {cultivo}"
    )

actualizar_titulo()

# BOTONES

ax_prev = fig.add_axes(
    [0.38, 0.03, 0.10, 0.045]
)

ax_next = fig.add_axes(
    [0.52, 0.03, 0.10, 0.045]
)

btn_prev = Button(
    ax_prev,
    "◀ Anterior"
)

btn_next = Button(
    ax_next,
    "Siguiente ▶"
)

def ir_anterior(event):

    nav["idx"] = (
        nav["idx"] - 1
    ) % len(PARCELAS_ORDEN)

    actualizar_titulo()

def ir_siguiente(event):

    nav["idx"] = (
        nav["idx"] + 1
    ) % len(PARCELAS_ORDEN)

    actualizar_titulo()

btn_prev.on_clicked(ir_anterior)
btn_next.on_clicked(ir_siguiente)

# INFO

ax_info = fig.add_axes(
    [0.06, 0.03, 0.28, 0.045]
)

ax_info.set_axis_off()

texto_estado = ax_info.text(
    0.0,
    0.5,
    "",
    ha="left",
    va="center",
    fontsize=9,
    fontweight="bold",
    color="#00FFCC",
    transform=ax_info.transAxes
)

# UPDATE GRAPH

def actualizar_grafica(frame):

    p = parcela_actual()

    with _lock:

        buf = buffers[p]

        if not buf["t_temperature_air"]:

            texto_estado.set_text(
                "Esperando datos..."
            )

            return

        datos = {}

        for t in TARGETS:

            datos[f"t_{t}"] = list(
                buf[f"t_{t}"]
            )

            datos[f"real_{t}"] = list(
                buf[f"real_{t}"]
            )

            datos[f"pred_{t}"] = list(
                buf[f"pred_{t}"]
            )

            datos[f"mae_{t}"] = list(
                buf[f"mae_{t}"]
            )

        paso = buf["paso"]

    # SERIES

    for t in TARGETS:

        tv = datos[f"t_{t}"]

        real = datos[f"real_{t}"]

        pred = datos[f"pred_{t}"]

        lineas_real[t].set_data(tv, real)

        lineas_pred[t].set_data(tv, pred)

        ax = AXES_MAP[t]

        ax.relim()

        ax.autoscale_view()

    # MAE

    max_mae = 0.0

    for t in TARGETS:

        tv = datos[f"t_{t}"]

        mae = datos[f"mae_{t}"]

        lineas_mae[t].set_data(tv, mae)

        if len(mae):
            max_mae = max(max_mae, max(mae))

    ax_err.relim()

    ax_err.autoscale_view()

    # IMPORTANTE:
    # arregla el problema visual de precipitacion

    if max_mae < 1:
        ax_err.set_ylim(0, 1.2)
    else:
        ax_err.set_ylim(0, max_mae * 1.15)

    # STATUS

    mae_temp = (
        datos["mae_temperature_air"][-1]
        if datos["mae_temperature_air"]
        else 0
    )

    texto_estado.set_text(
        f"Muestras={paso} | "
        f"MAE Temp={mae_temp:.3f}"
    )

    fig.canvas.draw_idle()

ani = animation.FuncAnimation(
    fig,
    actualizar_grafica,
    interval=500,
    blit=False,
    cache_frame_data=False
)

plt.show()

print("\nApagando MQTT...")

mqtt_client.loop_stop()

mqtt_client.disconnect()