import pandas as pd
import numpy as np

def cargar_datos_historicos(ruta_o_archivo):
    """
    Carga datos desde una ruta de archivo o un objeto UploadedFile de Streamlit.
    """
    if hasattr(ruta_o_archivo, 'name'):
        if ruta_o_archivo.name.endswith('.csv'):
            return pd.read_csv(ruta_o_archivo)
        else:
            return pd.read_excel(ruta_o_archivo)
    elif isinstance(ruta_o_archivo, str):
        if ruta_o_archivo.endswith('.csv'):
            return pd.read_csv(ruta_o_archivo)
        else:
            return pd.read_excel(ruta_o_archivo)
    return pd.DataFrame()


def enriquecer_datos_temporales(df, columna_fecha='fecha'):
    """
    Normaliza columnas y genera todas las variables predictivas requeridas por el modelo:
    - Lags (1 y 7 días)
    - Media móvil (7 días)
    - Indicadores de calendario (es_fin_de_semana, es_quincena, mes, dia_semana)
    - Precio unitario por caja
    """
    df = df.copy()

    # 1. Identificar y normalizar Fecha
    col_fecha = None
    for col in df.columns:
        c_low = str(col).lower().strip()
        if c_low in ['fecha_corte', 'fecha', 'date', 'fechas'] or ('fecha' in c_low and 'corte' in c_low):
            col_fecha = col
            break

    if col_fecha:
        df['fecha'] = pd.to_datetime(df[col_fecha])
    elif columna_fecha in df.columns:
        df['fecha'] = pd.to_datetime(df[columna_fecha])
    else:
        raise KeyError("No se encontró la columna de fecha en el archivo Excel.")

    # 2. Identificar y normalizar SKU
    col_sku = None
    for col in df.columns:
        c_low = str(col).lower().strip()
        if c_low in ['sku', 'codigo', 'producto_id', 'nombre_producto']:
            col_sku = col
            break

    if col_sku:
        df['sku'] = df[col_sku]

    # 3. Identificar y normalizar Demanda / Ventas
    col_demanda = None
    for candidato in ['demanda_total_estimada', 'cantidad_vendida', 'ventas_despachadas', 'demanda', 'ventas']:
        for col in df.columns:
            if str(col).lower().strip() == candidato:
                col_demanda = col
                break
        if col_demanda:
            break

    if col_demanda:
        df['demanda'] = df[col_demanda]
        df['cantidad_vendida'] = df[col_demanda]
    else:
        raise KeyError("No se encontró ninguna columna de demanda o ventas en el Excel.")

    # 4. Identificar y normalizar Precio de Caja
    col_precio = None
    for candidato in ['costo_unitario_compra_cop', 'precio_unitario_venta_cop', 'precio_caja', 'precio', 'costo']:
        for col in df.columns:
            if str(col).lower().strip() == candidato:
                col_precio = col
                break
        if col_precio:
            break

    if col_precio:
        df['precio_caja'] = df[col_precio]
    else:
        df['precio_caja'] = 45000.0

    # Ordenar cronológicamente por SKU
    df = df.sort_values(by=['sku', 'fecha'])

    # 5. Generar Rezagos (Lags)
    df['ventas_lag_1'] = df.groupby('sku')['demanda'].shift(1)
    df['ventas_lag_7'] = df.groupby('sku')['demanda'].shift(7)
    df['ventas_lag_1'] = df['ventas_lag_1'].fillna(df['demanda'])
    df['ventas_lag_7'] = df['ventas_lag_7'].fillna(df['demanda'])

    # 6. Generar Media Móvil (7 días)
    df['media_movil_7d'] = df.groupby('sku')['demanda'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['media_movil_7d'] = df['media_movil_7d'].fillna(df['demanda'])

    # 7. Generar Variables de Calendario
    df['dia_semana'] = df['fecha'].dt.dayofweek
    df['dia'] = df['fecha'].dt.day
    df['mes'] = df['fecha'].dt.month
    
    # Nombres exactos de variables solicitadas por predict.py
    df['es_fin_de_semana'] = df['dia_semana'].apply(lambda x: 1 if x >= 5 else 0)
    df['es_fin_semana'] = df['es_fin_de_semana']  # Alias de seguridad
    df['es_quincena'] = df['dia'].apply(lambda x: 1 if x in [14, 15, 28, 29, 30, 31] else 0)

    return df