import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

st.set_page_config(layout="wide", page_title="Prediksi Permintaan Produk Retail")

st.title("📊 Prediksi Permintaan Produk Retail di Jakarta Timur")
st.caption("Menggunakan Algoritma Regresi Linear")

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Harga Satuan'] = pd.to_numeric(df['Harga Satuan'], errors='coerce')
    df['Stok Tersedia'] = pd.to_numeric(df['Stok Tersedia'], errors='coerce')
    df['Penjualan (Unit)'] = pd.to_numeric(df['Penjualan (Unit)'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar input
st.sidebar.header("🎯 Prediksi Manual")
kategori = st.sidebar.selectbox("Kategori Produk", df['Kategori Produk'].unique())
harga = st.sidebar.slider("Harga Satuan (Rp)", int(df['Harga Satuan'].min()), int(df['Harga Satuan'].max()), int(df['Harga Satuan'].mean()))
stok = st.sidebar.slider("Stok Tersedia", int(df['Stok Tersedia'].min()), int(df['Stok Tersedia'].max()), int(df['Stok Tersedia'].mean()))

# Preprocessing
X = df[['Harga Satuan', 'Stok Tersedia']]
y = df['Penjualan (Unit)']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Prediksi manual
manual_pred = model.predict([[harga, stok]])[0]
st.sidebar.metric("📈 Prediksi Penjualan", f"{manual_pred:.0f} unit")

# Tabs
with st.expander("📊 Data Ringkasan"):
    st.dataframe(df.head())
    st.write(f"Jumlah data: {df.shape[0]} baris")

with st.expander("📉 Evaluasi Model"):
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    st.metric("R-Squared", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")

with st.expander("🧪 Uji Asumsi Klasik"):
    X_train_sm = sm.add_constant(X_train)
    X_train_sm = X_train_sm.astype(float)
    y_train = y_train.astype(float)
    model_sm = sm.OLS(y_train, X_train_sm).fit()
    resid = model_sm.resid

    # Uji normalitas
    stat, p_normal = shapiro(resid)
    st.write("### Uji Normalitas (Shapiro-Wilk)")
    st.write(f"p-value = {p_normal:.4f}")
    st.success("Residual terdistribusi normal") if p_normal > 0.05 else st.error("Residual tidak normal")

    # Uji homoskedastisitas
    _, p_bp, _, _ = het_breuschpagan(resid, X_train_sm)
    st.write("### Uji Homoskedastisitas (Breusch-Pagan)")
    st.write(f"p-value = {p_bp:.4f}")
    st.success("Homoskedastisitas terpenuhi") if p_bp > 0.05 else st.error("Terdapat heteroskedastisitas")

    # VIF
    st.write("### Multikolinearitas (VIF)")
    X_vif = X_train.astype(float).dropna()
    vif_df = pd.DataFrame()
    vif_df['Fitur'] = X_vif.columns
    vif_df['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
    st.dataframe(vif_df)

with st.expander("📈 Visualisasi"):
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Scatter Plot")
        fig1, ax1 = plt.subplots(figsize=(5,3))
        sns.scatterplot(x=y_test, y=y_pred, ax=ax1)
        ax1.set_xlabel("Aktual")
        ax1.set_ylabel("Prediksi")
        ax1.set_title("Aktual vs Prediksi")
        st.pyplot(fig1)
    with col2:
        st.write("### Residual Histogram")
        fig2, ax2 = plt.subplots(figsize=(5,3))
        sns.histplot(resid, kde=True, ax=ax2)
        ax2.set_title("Distribusi Residual")
        st.pyplot(fig2)

st.markdown("---")
st.caption("Dashboard ini disesuaikan dengan struktur dan isi presentasi Anda.")
