"""
Evaluación y Comparación de Modelos de Machine Learning (Batch)
Entrena y evalúa Linear Regression, Random Forest y Gradient Boosting
utilizando datos agroclimáticos reales para la estimación de variables críticas.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from .config import UMBRALES_AGRONOMICOS
except ImportError:
    from config import UMBRALES_AGRONOMICOS


def cargar_datos_procesados():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "dataset_agroclimatico.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el dataset en {data_path}")
    
    df = pd.read_csv(data_path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    col_map = {
        "temperature": "temperature_air",
        "rainfall": "precipitation",
        "soil_ph": "soil_ph",
        "humidity": "humidity"
    }
    df = df.rename(columns=col_map)
    return df


def entrenar_comparativa():
    df = cargar_datos_procesados()
    features = ["humidity", "precipitation", "soil_ph"]
    target = "temperature_air"

    df_clean = df.dropna(subset=features + [target])

    X = df_clean[features].values
    y = df_clean[target].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    modelos = {
        "Regresión Lineal": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }

    print(f"\n{'MODELO':<25} {'MAE':>8} {'RMSE':>8} {'R2':>8} {'CV R2':>10}")
    print("-" * 65)

    resultados = {}
    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        cv = cross_val_score(modelo, X_scaled, y, cv=5, scoring="r2").mean()

        resultados[nombre] = {"modelo": modelo, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv}
        print(f"{nombre:<25} {mae:>8.3f} {rmse:>8.3f} {r2:>8.3f} {cv:>10.3f}")

    mejor_nombre = max(resultados, key=lambda k: resultados[k]["R2"])
    print(f"\nMejor modelo seleccionado: {mejor_nombre} (R2 = {resultados[mejor_nombre]['R2']:.3f})")
    return resultados


if __name__ == "__main__":
    entrenar_comparativa()
