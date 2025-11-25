import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

# --------------------------
# FASE 1: CARGA DE DATOS
# --------------------------

st.title("🏡 Análisis Interactivo – California Housing Prices")

# Cargar dataset
data = fetch_california_housing()
df_california = pd.DataFrame(data.data, columns=data.feature_names)
df_california["MedHouseVal"] = data.target

st.subheader("📥 Vista Inicial de los Datos")
st.dataframe(df_california.head())

st.write("**Tipos de datos:**")
st.write(df_california.dtypes)

st.write("**Valores faltantes por columna:**")
st.write(df_california.isnull().sum())

# --------------------------
# FASE 2: WIDGETS INTERACTIVOS
# --------------------------

st.sidebar.markdown("## 🎛️ Filtros de Exploración")
st.sidebar.write("Ajusta los filtros para explorar el dataset")

# Filtro: Edad de la vivienda
min_age = int(df_california["HouseAge"].min())
max_age = int(df_california["HouseAge"].max())

age_range = st.sidebar.slider(
    "Selecciona el rango de *HouseAge*:",
    min_age, max_age, (min_age, max_age)
)

df_filtered = df_california[
    (df_california["HouseAge"] >= age_range[0]) &
    (df_california["HouseAge"] <= age_range[1])
]

# Filtro: Latitud mínima
lat_min = st.sidebar.number_input(
    "Latitud mínima:",
    min_value=float(df_california["Latitude"].min()),
    max_value=float(df_california["Latitude"].max()),
    value=float(df_california["Latitude"].min())
)

df_filtered = df_filtered[df_filtered["Latitude"] >= lat_min]

# Resumen estadístico filtrado
st.subheader("📊 Resumen del Valor de la Vivienda (Filtrado)")
mediana_val = df_filtered["MedHouseVal"].median()
rango_val = df_filtered["MedHouseVal"].max() - df_filtered["MedHouseVal"].min()

st.write(f"**Mediana del valor (MedHouseVal):** {mediana_val:.2f}")
st.write(f"**Rango (max - min):** {rango_val:.2f}")

# --------------------------
# FASE 3: VISUALIZACIONES
# --------------------------

# Histograma
st.subheader("📈 Histograma del Valor de la Vivienda")

fig_hist, ax_hist = plt.subplots()
ax_hist.hist(df_filtered["MedHouseVal"], bins=30, color="skyblue", edgecolor="black")
ax_hist.set_xlabel("MedHouseVal")
ax_hist.set_ylabel("Frecuencia")
ax_hist.set_title("Distribución del Valor de las Viviendas")

st.pyplot(fig_hist)

# Scatter plot MedInc vs MedHouseVal
st.subheader("📌 Relación entre Ingresos (MedInc) y Valor de Vivienda (MedHouseVal)")

fig_scatter, ax_scatter = plt.subplots()
ax_scatter.scatter(df_filtered["MedInc"], df_filtered["MedHouseVal"], alpha=0.5)
ax_scatter.set_xlabel("MedInc (Mediana de Ingresos)")
ax_scatter.set_ylabel("MedHouseVal (Valor Mediano)")
ax_scatter.set_title("MedInc vs MedHouseVal")

st.pyplot(fig_scatter)

# --------------------------
# MAPA GEOGRÁFICO (CORREGIDO)
# --------------------------

st.subheader("🗺️ Mapa de las viviendas filtradas")

# Renombrar columnas a minúsculas porque st.map exige "latitude" y "longitude"
df_map = df_filtered.rename(columns={"Latitude": "latitude", "Longitude": "longitude"})

# Mostrar mapa si hay datos
if not df_map.empty:
    st.map(df_map[["latitude", "longitude"]])
else:
    st.write("No hay datos para mostrar en el mapa con los filtros seleccionados.")
