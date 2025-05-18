import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Configuración inicial de la página
st.set_page_config(page_title="Optimización Energética Industrial", layout="wide")
st.title("Análisis de Consumo Energético Industrial")

# Carga de datos
@st.cache_data  # Cache para mejorar rendimiento
def load_data():
    try:
        df = pd.read_csv('industrial_consumption_dataset.csv')
        # Validación de datos
        if not all(col in df.columns for col in ['Temperature', 'Velocity', 'Hours']):
            st.error("El dataset no contiene las columnas requeridas. Usando datos generados.")
            raise ValueError
        return df
    except Exception as e:
        st.warning("No se pudo cargar el archivo. Generando datos sintéticos...")
        np.random.seed(42)
        data = {
            'Temperature': np.random.uniform(10, 50, 100),
            'Velocity': np.random.uniform(1, 20, 100),
            'Hours': np.random.randint(1, 24, 100)
        }
        return pd.DataFrame(data)

df = load_data()

# Cálculo de energía y derivadas
df['Energy'] = 0.5*df['Temperature']**2 + 0.1*df['Velocity']**3 + 2*df['Hours'] + 0.3*df['Temperature']*df['Velocity']
df['dEdT'] = df['Temperature'] + 0.3*df['Velocity']  # Derivada parcial teórica
df['dEdV'] = 0.3*df['Velocity']**2 + 0.3*df['Temperature']  # Derivada parcial teórica

# Sidebar para controles
with st.sidebar:
    st.header("Parámetros de Control")
    T = st.slider("Temperatura (°C)", 10.0, 50.0, 25.0, 0.1)
    V = st.slider("Velocidad (unidades/min)", 1.0, 20.0, 10.0, 0.1)
    H = st.slider("Horas de operación", 1, 24, 8)
    st.divider()
    st.caption("Proyecto de Cálculo II - Optimización Energética")

# Cálculos principales
E = 0.5*T**2 + 0.1*V**3 + 2*H + 0.3*T*V
dEdT = T + 0.3*V
dEdV = 0.3*V**2 + 0.3*T

# Filtrado de datos similares
similar_data = df[
    (df['Temperature'].between(T-2, T+2)) & 
    (df['Velocity'].between(V-1, V+1))
]

# Visualización en pestañas
tab1, tab2, tab3 = st.tabs(["Resultados", "Validación", "Análisis"])

with tab1:
    st.header("Resultados Principales")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Consumo Energético", f"{E:.2f} kW/h")
    with col2:
        st.metric("Sensibilidad a Temperatura", f"{dEdT:.2f} kW/h/°C")
    with col3:
        st.metric("Sensibilidad a Velocidad", f"{dEdV:.2f} kW/h/(unidad/min)")
    
    # Gráfico 3D opcional (requiere plotly)
    st.write("Distribución del consumo en los datos:")
    fig, ax = plt.subplots()
    scatter = ax.scatter(df['Temperature'], df['Velocity'], c=df['Energy'], cmap='viridis')
    ax.scatter(T, V, c='red', s=100, label='Punto seleccionado')
    plt.colorbar(scatter, label='Consumo Energético (kW/h)')
    ax.set_xlabel('Temperatura (°C)')
    ax.set_ylabel('Velocidad (unidades/min)')
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.header("Validación con Datos Simulados")
    
    if len(similar_data) > 0:
        st.write(f"Datos encontrados con parámetros similares: {len(similar_data)} registros")
        
        # Cálculo de derivadas numéricas
        numeric_dEdT = similar_data['Energy'].diff() / similar_data['Temperature'].diff()
        numeric_dEdV = similar_data['Energy'].diff() / similar_data['Velocity'].diff()
        
        st.subheader("Comparación de Derivadas")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**∂E/∂T (Temperatura):**")
            st.write(f"Teórica: {dEdT:.2f}")
            st.write(f"Numérica: {numeric_dEdT.mean():.2f}")
        with col2:
            st.write("**∂E/∂V (Velocidad):**")
            st.write(f"Teórica: {dEdV:.2f}")
            st.write(f"Numérica: {numeric_dEdV.mean():.2f}")
        
        st.subheader("Datos Similares")
        st.dataframe(similar_data[['Temperature', 'Velocity', 'Hours', 'Energy']].head())
    else:
        st.warning("No se encontraron datos con parámetros similares")

with tab3:
    st.header("Análisis Avanzado")
    
    st.subheader("Campo Vectorial del Gradiente")
    st.write("Visualización de cómo cambia el consumo energético en diferentes condiciones:")
    
    # Muestra solo 15 puntos para mejor visualización
    sample = df.sample(15, random_state=42)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.quiver(
        sample['Temperature'],
        sample['Velocity'],
        sample['dEdT'],
        sample['dEdV'],
        scale=300,
        color='blue',
        width=0.003
    )
    ax.scatter(T, V, c='red', s=100, label='Punto actual')
    ax.set_xlabel('Temperatura (°C)')
    ax.set_ylabel('Velocidad (unidades/min)')
    ax.set_title('Campo Vectorial del Gradiente ∇E(T,V)')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("Distribución del Consumo")
    st.line_chart(df.groupby('Temperature')['Energy'].mean())