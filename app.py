import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera

# Load data
@st.cache_data

def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

df = load_data()

# Persiapan data regresi
X = sm.add_constant(df[['Harga Satuan', 'Stok Tersedia']], has_constant='add')
y = df['Penjualan (Unit)']
model = sm.OLS(y, X).fit()

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail")

menu = st.sidebar.selectbox("Pilih Menu", [
    "Visualisasi Data", 
    "Evaluasi Model", 
    "Uji Asumsi Klasik", 
    "Koefisien Model", 
    "Simulasi Prediksi 3 Bulan ke Depan"
])

# VISUALISASI
if menu == "Visualisasi Data":
    st.subheader("📈 Penjualan per Bulan")
    df_bulanan = df.resample('M', on='Tanggal').sum(numeric_only=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    df_bulanan['Penjualan (Unit)'].plot(kind='bar', ax=ax, color='teal')
    ax.set_title("Total Penjualan per Bulan")
    ax.set_ylabel("Unit Terjual")
    st.pyplot(fig)

# EVALUASI MODEL
elif menu == "Evaluasi Model":
    st.subheader("📌 Evaluasi Model Regresi")
    y_pred = model.predict(X)
    mse = np.mean((y - y_pred)**2)
    mae = np.mean(np.abs(y - y_pred))
    r2 = model.rsquared
    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared", f"{r2:.3f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    st.markdown("---")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.scatter(y, y_pred, alpha=0.6, color='green')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Aktual vs Prediksi")
    st.pyplot(fig)

# UJI ASUMSI
elif menu == "Uji Asumsi Klasik":
    st.subheader("🧪 Uji Asumsi Klasik")
    residuals = y - y_pred
    jb_stat, jb_p, _, _ = jarque_bera(residuals)
    _, bp_p, _, _ = het_breuschpagan(residuals, X)
    vif = pd.DataFrame()
    vif["Fitur"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    st.write("### Uji Normalitas (Jarque-Bera)")
    st.write(f"P-Value: {jb_p:.4f}")
    st.success("Lolos uji normalitas" if jb_p > 0.05 else "Tidak lolos uji normalitas")

    st.write("### Uji Homoskedastisitas (Breusch-Pagan)")
    st.write(f"P-Value: {bp_p:.4f}")
    st.success("Lolos homoskedastisitas" if bp_p > 0.05 else "Tidak lolos homoskedastisitas")

    st.write("### Uji Multikolinearitas (VIF)")
    st.dataframe(vif)

# KOEFISIEN
elif menu == "Koefisien Model":
    st.subheader("📌 Koefisien dan Signifikansi")
    summary = model.summary2().tables[1]
    summary = summary.rename(columns={"Coef.": "Koefisien", "P>|t|": "P-Value"})
    summary = summary[['Koefisien', 'P-Value']]
    summary['Signifikan'] = summary['P-Value'] < 0.05
    st.dataframe(summary)
    st.write(f"**P-Value Uji F (Signifikansi Model Keseluruhan):** {model.f_pvalue:.4f}")

# PREDIKSI SIMULASI
elif menu == "Simulasi Prediksi 3 Bulan ke Depan":
    st.subheader("📅 Simulasi Prediksi Jan–Mar 2025")

    dates = pd.date_range("2025-01-01", "2025-03-31")
    harga_simulasi = np.random.normal(df['Harga Satuan'].mean(), df['Harga Satuan'].std(), len(dates))
    stok_simulasi = np.random.normal(df['Stok Tersedia'].mean(), df['Stok Tersedia'].std(), len(dates))

    df_pred = pd.DataFrame({
        "Tanggal": dates,
        "Harga Satuan": harga_simulasi,
        "Stok Tersedia": stok_simulasi
    })
    X_pred = sm.add_constant(df_pred[['Harga Satuan', 'Stok Tersedia']], has_constant='add')
    X_pred = X_pred[X.columns]  # pastikan urutan kolom sama
    df_pred['Prediksi'] = model.predict(X_pred)

    df_bulanan_pred = df_pred.groupby(df_pred['Tanggal'].dt.to_period('M')).sum(numeric_only=True)
    df_bulanan_pred.index = df_bulanan_pred.index.to_timestamp()

    st.write("### Tabel Prediksi Harian (Top 30)")
    st.dataframe(df_pred.head(30))

    st.write("### Total Prediksi per Bulan")
    fig, ax = plt.subplots(figsize=(4, 3))
    df_bulanan_pred['Prediksi'].plot(kind='bar', ax=ax, color='orange')
    ax.set_title("Prediksi Penjualan per Bulan (2025)")
    ax.set_ylabel("Unit")
    st.pyplot(fig)
