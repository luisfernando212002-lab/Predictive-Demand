import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def entrenar_random_forest(df_procesado, sku_objetivo):
    """
    Entrena el modelo Random Forest y retorna:
    - rf: Modelo entrenado
    - y_test: Valores reales del conjunto de evaluación (20%)
    - predicciones_test: Predicciones sobre el conjunto de evaluación (para métricas)
    - predicciones_totales: Predicciones sobre todo el conjunto X (para Gurobi)
    """
    # Filtrar datos para el SKU objetivo
    df_sku = df_procesado[df_procesado['sku'] == sku_objetivo].copy()
    df_sku = df_sku.sort_values(by='fecha')

    # Lista total de características potenciales
    caracteristicas_deseadas = [
        'ventas_lag_1', 'ventas_lag_7', 'media_movil_7d',
        'es_quincena', 'es_fin_de_semana', 'es_fin_semana',
        'precio_caja', 'dia_semana', 'mes'
    ]

    # Filtrar dinámicamente solo las columnas que realmente existen
    caracteristicas = [c for c in caracteristicas_deseadas if c in df_sku.columns]

    X = df_sku[caracteristicas]
    y = df_sku['demanda']

    # División 80/20 respetando el orden temporal
    split_idx = int(len(df_sku) * 0.8)
    if split_idx < 1 or split_idx == len(df_sku):
        split_idx = max(1, len(df_sku) - 1)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Entrenar el modelo
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predicciones separadas: prueba (para métricas) vs total (para Gurobi)
    predicciones_test = rf.predict(X_test)
    predicciones_totales = rf.predict(X)

    return rf, y_test, predicciones_test, predicciones_totales


def calcular_metricas(y_true, y_pred):
    """
    Calcula MAE, RMSE, MAPE y WAPE de las predicciones.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE evitando división por cero
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

    # WAPE
    wape = (np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(y_true), 1)) * 100

    return mae, rmse, mape, wape