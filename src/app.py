import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_prep import enriquecer_datos_temporales
from predict import entrenar_random_forest, calcular_metricas
from optimize import optimizar_abastecimiento

st.set_page_config(page_title="Optimizador Logístico", layout="wide", page_icon="📦")

st.title("📦 Plataforma de Inteligencia Logística")
st.markdown("Integración de Machine Learning (Random Forest) y MILP (Gurobi) adaptado al marco metodológico PRISMA.")
st.divider()

# --- INICIALIZACIÓN DE ESTADO PARA PARÁMETROS FINANCIEROS Y OPERATIVOS ---
if "c_orden" not in st.session_state:
    st.session_state["c_orden"] = 50000
if "c_compra" not in st.session_state:
    st.session_state["c_compra"] = 45000
if "c_inv" not in st.session_state:
    st.session_state["c_inv"] = 500
if "c_faltante" not in st.session_state:
    st.session_state["c_faltante"] = 80000
if "capacidad" not in st.session_state:
    st.session_state["capacidad"] = 500

# --- PANEL PRINCIPAL: CARGA DE ARCHIVO Y PROCESAMIENTO ---
st.subheader("1. Carga de Datos y Selección de Producto")
archivo_subido = st.file_uploader("Arrastra tu archivo Excel o CSV con el historial de ventas", type=["xlsx", "csv"])

df = None
producto_seleccionado = None

if archivo_subido is not None:
    try:
        if archivo_subido.name.endswith('.csv'):
            df = pd.read_csv(archivo_subido)
        else:
            df = pd.read_excel(archivo_subido)
            
        # Detección inteligente y renombrado de columnas según el Excel de Valencia
        columnas_renombrar = {}
        for col in df.columns:
            c_low = str(col).lower().strip()
            if c_low in ['fecha_corte', 'date', 'fechas'] or ('fecha' in c_low and 'corte' in c_low):
                columnas_renombrar[col] = 'fecha'
            elif c_low in ['sku', 'codigo', 'producto_id']:
                columnas_renombrar[col] = 'sku'
            elif c_low in ['demanda_total_estimada', 'ventas_despachadas', 'cantidad_vendida', 'demanda', 'ventas']:
                if 'demanda' not in columnas_renombrar.values():
                    columnas_renombrar[col] = 'demanda'

        df = df.rename(columns=columnas_renombrar)
        
        # Garantizar que 'cantidad_vendida' y 'demanda' existan siempre
        if 'demanda' in df.columns:
            df['cantidad_vendida'] = df['demanda']

        # Validación de SKU
        if 'sku' not in df.columns:
            cols_sku = [c for c in df.columns if 'sku' in str(c).lower() or 'producto' in str(c).lower()]
            if cols_sku:
                df = df.rename(columns={cols_sku[0]: 'sku'})
            else:
                st.error("⚠️ No se encontró la columna de producto/SKU en el archivo.")
                st.stop()

        # Selección de Producto (SKU)
        lista_productos = df['sku'].unique().tolist()
        producto_seleccionado = st.selectbox("Seleccione el producto a optimizar (SKU):", lista_productos)
        
        # Extraer parámetros financieros desde el Excel para el SKU seleccionado
        df_prod = df[df['sku'] == producto_seleccionado]
        
        col_compra = [c for c in df.columns if 'costo' in c.lower() and ('compra' in c.lower() or 'unitario' in c.lower()) and 'cedi' not in c.lower()]
        col_inv = [c for c in df.columns if 'costo' in c.lower() and ('almacenamiento' in c.lower() or 'inventario' in c.lower())]
        col_faltante = [c for c in df.columns if 'faltante' in c.lower() or 'penalizacion' in c.lower() or 'quiebre' in c.lower()]
        col_orden = [c for c in df.columns if 'orden' in c.lower() or 'flete' in c.lower()]
        col_capacidad = [c for c in df.columns if 'capacidad' in c.lower() or 'bodega' in c.lower()]

        # Actualizar la sesión si cambia el producto o archivo
        llave_rastreo = f"{archivo_subido.name}_{producto_seleccionado}"
        if st.session_state.get("ultima_llave_procesada") != llave_rastreo:
            st.session_state["ultima_llave_procesada"] = llave_rastreo
            
            # Costo de Compra por Caja
            if col_compra and not df_prod[col_compra[0]].dropna().empty:
                st.session_state["c_compra"] = int(df_prod[col_compra[0]].dropna().iloc[0])
                
            # Costo de Almacenamiento por Caja
            if col_inv and not df_prod[col_inv[0]].dropna().empty:
                raw_inv = float(df_prod[col_inv[0]].dropna().iloc[0])
                if 'inventario_disponible' in df_prod.columns and df_prod['inventario_disponible'].iloc[0] > 0:
                    val_inv = raw_inv / float(df_prod['inventario_disponible'].iloc[0])
                elif raw_inv > 10000:
                    val_inv = 500
                else:
                    val_inv = raw_inv
                st.session_state["c_inv"] = int(val_inv)
                
            # Penalización por Faltante
            if col_faltante and not df_prod[col_faltante[0]].dropna().empty:
                val_faltante = float(df_prod[col_faltante[0]].dropna().iloc[0])
                if abs(val_faltante) > 0:
                    st.session_state["c_faltante"] = int(abs(val_faltante))
                    
            # Costo por Emitir Orden
            if col_orden and not df_prod[col_orden[0]].dropna().empty:
                st.session_state["c_orden"] = int(df_prod[col_orden[0]].dropna().iloc[0])
                
            # Capacidad de Bodega
            if col_capacidad and not df_prod[col_capacidad[0]].dropna().empty:
                st.session_state["capacidad"] = int(df_prod[col_capacidad[0]].dropna().iloc[0])

            st.toast(f"✅ Costos extraídos del Excel para: {producto_seleccionado}")

    except Exception as e:
        st.error(f"❌ Error al procesar el archivo: {e}")
        st.stop()

# --- BARRA LATERAL: INPUTS DEL USUARIO (EDITABLES) ---
st.sidebar.header("⚙️ Parámetros Financieros (COP)")
st.sidebar.caption("Sincronizados con el archivo subido y totalmente editables:")

c_orden = st.sidebar.number_input(
    "Costo por Emitir Orden ($)", 
    value=int(st.session_state["c_orden"]), 
    step=1000
)

c_compra = st.sidebar.number_input(
    "Costo de Compra por Caja ($)", 
    value=int(st.session_state["c_compra"]), 
    step=1000
)

c_inv = st.sidebar.number_input(
    "Costo de Almacenamiento ($)", 
    value=int(st.session_state["c_inv"]), 
    step=100
)

c_faltante = st.sidebar.number_input(
    "Penalización por Faltante ($)", 
    value=int(st.session_state["c_faltante"]), 
    step=1000
)

capacidad = st.sidebar.number_input(
    "Capacidad de Bodega (m3)", 
    value=int(st.session_state["capacidad"]), 
    step=50
)

st.sidebar.divider()
st.sidebar.header("📅 Horizonte de Planificación")
dias_proyeccion = st.sidebar.slider("Días a Proyectar", min_value=1, max_value=30, value=7)

# --- EJECUCIÓN DEL PIPELINE DE IA Y OPTIMIZACIÓN ---
if archivo_subido is not None and df is not None and producto_seleccionado is not None:
    st.divider()
    
    if st.button(f"🚀 Ejecutar IA y Optimización para: {producto_seleccionado}", use_container_width=True):
        
        with st.spinner("Calculando variables rezagadas (Lags) y calendario..."):
            df_procesado = enriquecer_datos_temporales(df, 'fecha')
            
        with st.spinner("Entrenando modelo Random Forest Regressor..."):
            modelo_rf, y_test, predicciones_test, predicciones_totales = entrenar_random_forest(df_procesado, sku_objetivo=producto_seleccionado)
            
            # Calcular métricas exactamente sobre la muestra de evaluación (12 registros)
            mae, rmse, mape, wape = calcular_metricas(y_test, predicciones_test)
            
            st.subheader("📊 Rendimiento Predictivo (Ecuaciones 3-6)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("MAE (Cajas)", f"{mae:.2f}")
            col2.metric("RMSE (Cajas)", f"{rmse:.2f}")
            col3.metric("MAPE (%)", f"{mape:.2f}%")
            col4.metric("WAPE (%)", f"{wape:.2f}%")
            
        with st.spinner(f"Ejecutando solver Gurobi para {dias_proyeccion} días..."):
            horizonte_prediccion = predicciones_totales[-dias_proyeccion:] 
            
            # Usar los parámetros de la barra lateral
            tabla_resultados, costo_total = optimizar_abastecimiento(
                horizonte_prediccion, c_orden, c_compra, c_inv, c_faltante, capacidad
            )
            
            st.divider()
            st.subheader(f"📋 Plan Prescriptivo de Abastecimiento (Próximos {dias_proyeccion} días)")
            st.metric(label="Costo Total Logístico Proyectado (Función Objetivo)", value=f"${costo_total:,.0f} COP")
            
            st.dataframe(tabla_resultados, use_container_width=True, hide_index=True)
            st.success("¡Pipeline matemático ejecutado con éxito!")
else:
    st.info("Esperando la carga del archivo de ventas histórico...")