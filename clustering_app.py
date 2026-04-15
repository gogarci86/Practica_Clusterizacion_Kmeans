import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.datasets import fetch_openml
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Configuración de página
st.set_page_config(page_title="Credit Card Clustering Dashboard", layout="wide")

st.title("💳 Dashboard de Segmentación de Clientes HEDY LAMARR")
st.markdown("""
Esta aplicación analiza y segmenta clientes de tarjetas de crédito basándose en su comportamiento financiero,
utilizando algoritmos de **Machine Learning** y técnicas de **Reducción de Dimensionalidad**.
""")

# --- 1. Carga y Procesamiento de Datos (Cacheado) ---
@st.cache_data
def load_and_process_data():
    # Descarga
    credit_data = fetch_openml(data_id=42477, as_frame=True, parser='auto')
    df = credit_data.frame
    
    # Renombrado de negocio
    nombres_negocio = {
        'x1': 'limite_credito', 'x2': 'genero', 'x3': 'educacion', 'x4': 'estado_civil', 'x5': 'edad',
        'x6': 'estado_pago_sep', 'x7': 'estado_pago_ago', 'x8': 'estado_pago_jul',
        'x9': 'estado_pago_jun', 'x10': 'estado_pago_may', 'x11': 'estado_pago_abr',
        'x12': 'deuda_sep', 'x13': 'deuda_ago', 'x14': 'deuda_jul',
        'x15': 'deuda_jun', 'x16': 'deuda_may', 'x17': 'deuda_abr',
        'x18': 'pagado_sep', 'x19': 'pagado_ago', 'x20': 'pagado_jul',
        'x21': 'pagado_jun', 'x22': 'pagado_may', 'x23': 'pagado_abr',
        'y': 'objetivo_default'
    }
    df.rename(columns=nombres_negocio, inplace=True)
    
    # Ingeniería de Variables
    df.drop_duplicates(inplace=True)
    cols_deuda = ['deuda_sep', 'deuda_ago', 'deuda_jul', 'deuda_jun', 'deuda_may', 'deuda_abr']
    cols_pagado = ['pagado_sep', 'pagado_ago', 'pagado_jul', 'pagado_jun', 'pagado_may', 'pagado_abr']
    
    df['total_deuda'] = df[cols_deuda].sum(axis=1)
    df['total_pagado'] = df[cols_pagado].sum(axis=1)
    df['diferencia_deuda_pago'] = df['total_deuda'] - df['total_pagado']
    df['dentro_del_limite'] = (df['total_deuda'] <= df['limite_credito']).astype(int)
    
    return df

with st.spinner('Cargando y procesando datos financieros...'):
    df = load_and_process_data()

# --- 2. Selección de Variables y Entrenamiento ---
features_selection = ['limite_credito', 'edad', 'total_deuda', 'total_pagado', 'diferencia_deuda_pago', 'dentro_del_limite', 'estado_pago_sep']
X_subset = df[features_selection]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

# Entrenamiento K-Means (Fijado a K=3 por decisión de negocio)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)
df['Cluster'] = df['Cluster'].astype(str)

# --- 3. Sidebar (Filtros) ---
st.sidebar.header("🛠️ Filtros de Análisis")

# Filtro por Cluster
selected_clusters = st.sidebar.multiselect(
    "Selecciona Clusters para visualizar:",
    options=sorted(df['Cluster'].unique()),
    default=sorted(df['Cluster'].unique())
)

# Filtro por Edad
min_age, max_age = int(df['edad'].min()), int(df['edad'].max())
age_range = st.sidebar.slider("Rango de Edad", min_age, max_age, (min_age, max_age))

# Filtro por Límite de Crédito
min_lim, max_lim = float(df['limite_credito'].min()), float(df['limite_credito'].max())
lim_range = st.sidebar.slider("Límite de Crédito", min_lim, max_lim, (min_lim, max_lim))

# Aplicar filtros
df_filtered = df[
    (df['Cluster'].isin(selected_clusters)) &
    (df['edad'].between(age_range[0], age_range[1])) &
    (df['limite_credito'].between(lim_range[0], lim_range[1]))
]

# --- 4. KPIs Principales ---
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Clientes", f"{len(df_filtered):,}")
with col2:
    st.metric("Deuda Promedio", f"${df_filtered['total_deuda'].mean():,.2f}")
with col3:
    st.metric("Límite Promedio", f"${df_filtered['limite_credito'].mean():,.2f}")
with col4:
    st.metric("% Dentro del Límite", f"{(df_filtered['dentro_del_limite'].mean()*100):.1f}%")

# --- 5. Visualizaciones PCA (2D y 3D) ---
st.subheader("🚀 Proyecciones PCA")

tab1, tab2 = st.tabs(["Vista 2D", "Vista 3D"])

with tab1:
    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_scaled)
    # Solo mostrar una muestra en el gráfico si es muy grande para fluidez (opcional)
    # Por ahora usamos el filtrado
    
    # Necesitamos unir el PCA con los datos filtrados para el hover
    # Lo más seguro es recalcular o mapear índices
    indices_filtrados = df_filtered.index
    # Nota: X_scaled y PCA se calculan sobre el total, aquí filtramos para el gráfico
    temp_pca = pd.DataFrame(X_pca_2d, index=df.index, columns=['PC1', 'PC2'])
    df_plot_2d = pd.concat([df_filtered, temp_pca.loc[indices_filtrados]], axis=1)

    fig_2d = px.scatter(
        df_plot_2d, x='PC1', y='PC2', color='Cluster',
        hover_data=['edad', 'limite_credito', 'total_deuda'],
        title="Visualización 2D (Fronteras de Segmentos)",
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.6
    )
    st.plotly_chart(fig_2d, use_container_width=True)

with tab2:
    pca_3d = PCA(n_components=3)
    X_pca_3d = pca_3d.fit_transform(X_scaled)
    temp_pca_3d = pd.DataFrame(X_pca_3d, index=df.index, columns=['PC1', 'PC2', 'PC3'])
    df_plot_3d = pd.concat([df_filtered, temp_pca_3d.loc[indices_filtrados]], axis=1)

    fig_3d = px.scatter_3d(
        df_plot_3d, x='PC1', y='PC2', z='PC3', color='Cluster',
        hover_data=['edad', 'limite_credito', 'total_deuda'],
        title="Visualización 3D (Estructura Espacial)",
        color_discrete_sequence=px.colors.qualitative.Safe,
        opacity=0.7, height=700
    )
    st.plotly_chart(fig_3d, use_container_width=True)

# --- 6. Tabla de Datos Detallada ---
st.subheader("📋 Datos del Segmento Seleccionado")
st.dataframe(df_filtered[features_selection + ['Cluster']].head(100), use_container_width=True)
