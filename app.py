ort streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

st.set_page_config(page_title="Dashboard Prediksi Permintaan Retail", layout="wide")
st.title("📊 Dashboard Business Intelligence: Prediksi Permintaan Produk Retail di Jakarta Timur")
st.markdown("Dashboard ini dikembangkan untuk menganalisis dan memprediksi permintaan produk retail menggunakan algoritma Regresi Linear berdasarkan data penjualan harian di Jakarta Timur.")

# Load data
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Bulan'] = df['Tanggal'].dt.to_period('M')

# ========================== Sidebar Dropdown ==========================
menu = st.sidebar.selectbox("Pilih Halaman Dashboard", [
    "Visualisasi Penjualan Bulanan",
    "Evaluasi Model",
    "Uji Asumsi Klasik",
    "Uji Signifikansi",
    "Visualisasi Aktual vs Prediksi",
    "Prediksi Berdasarkan Kategori"
])

# ========================== Model Setup =============================
X = df[["Harga Satuan", "Stok Tersedia"]]
y = df["Penjualan (Unit)"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pred = model.predict(X)
residuals = y - pred

# Drop kolom const untuk VIF
X_vif = X.drop(columns='const')

# ========================== PAGE 1 =============================
if menu == "Visualisasi Penjualan Bulanan":
    st.header("1. Visualisasi Tren Penjualan per Bulan")
    monthly_sales = df.groupby('Bulan')["Penjualan (Unit)"].sum()
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    monthly_sales.plot(kind='bar', color='skyblue', ax=ax1)
    ax1.set_xlabel("Bulan")
    ax1.set_ylabel("Jumlah Penjualan")
    ax1.set_title("Total Penjualan per Bulan")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

# ========================== PAGE 2 =============================
elif menu == "Evaluasi Model":
    st.header("2. Evaluasi Kinerja Model")
    mse = mean_squared_error(y, pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y, pred)
    r2 = model.rsquared

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", f"{r2:.3f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("MAE", f"{mae:.2f}")

# ========================== PAGE 3 =============================
elif menu == "Uji Asumsi Klasik":
    st.header("3. Uji Asumsi Klasik")

    with st.expander("3.1 Linearitas"):
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=pred, y=residuals, ax=ax2)
        ax2.axhline(0, color='red', linestyle='--')
        ax2.set_xlabel("Prediksi")
        ax2.set_ylabel("Residual")
        ax2.set_title("Scatterplot Prediksi vs Residual")
        st.pyplot(fig2)

    with st.expander("3.2 Normalitas (Shapiro-Wilk)"):
        shapiro_test = shapiro(residuals)
        st.write(f"p-value: {shapiro_test.pvalue:.4f}")
        if shapiro_test.pvalue > 0.05:
            st.success("✅ Residual terdistribusi normal")
        else:
            st.error("❌ Tidak normal")

    with st.expander("3.3 Homoskedastisitas (Breusch-Pagan)"):
        , pval_bp, _, __ = het_breuschpagan(residuals, X)
        st.write(f"p-value: {pval_bp:.4f}")
        if pval_bp > 0.05:
            st.success("✅ Tidak ada heteroskedastisitas")
        else:
            st.error("❌ Ada indikasi heteroskedastisitas")

    with st.expander("3.4 Multikolinearitas (VIF)"):
        vif_df = pd.DataFrame()
        vif_df['Fitur'] = X_vif.columns
        vif_df['VIF'] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        st.dataframe(vif_df)
        if (vif_df['VIF'] > 5).any():
            st.error("❌ Ada multikolinearitas tinggi")
        else:
            st.success("✅ Tidak ada multikolinearitas signifikan")

# ========================== PAGE 4 =============================
elif menu == "Uji Signifikansi":
    st.header("4. Uji Signifikansi Model")
    st.markdown("*Uji F:*")
    st.write(f"p-value: {model.f_pvalue:.4f}")
    if model.f_pvalue < 0.05:
        st.success("✅ Model signifikan secara keseluruhan")
    else:
        st.error("❌ Model tidak signifikan")

    st.markdown("*Uji t (Koefisien):*")
    signifikan = model.pvalues.drop('const') < 0.05
    for var, sig in signifikan.items():
        if sig:
            st.markdown(f"- ✅ {var}: signifikan")
        else:
            st.markdown(f"- ❌ {var}: tidak signifikan")

# ========================== PAGE 5 =============================
elif menu == "Visualisasi Aktual vs Prediksi":
    st.header("5. Visualisasi Aktual vs Prediksi")
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.scatter(y, pred, alpha=0.5, color='green')
    ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax3.set_xlabel("Aktual")
    ax3.set_ylabel("Prediksi")
    ax3.set_title("Perbandingan Penjualan Aktual vs Prediksi")
    st.pyplot(fig3)

# ========================== PAGE 6 =============================
elif menu == "Prediksi Berdasarkan Kategori":
    st.header("6. Prediksi Permintaan Berdasarkan Kategori Produk")
    kategori = st.selectbox("Pilih Kategori Produk", df['Kategori Produk'].unique())
    df_kat = df[df['Kategori Produk'] == kategori]
    X_kat = sm.add_constant(df_kat[['Harga Satuan', 'Stok Tersedia']], has_constant='add')
    X_kat = X_kat[X.columns]  # pastikan urutan kolom sama
    pred_kat = model.predict(X_kat)

    df_kat = df_kat.copy()
    df_kat['Prediksi Penjualan'] = pred_kat

    st.dataframe(df_kat[['Tanggal', 'Harga Satuan', 'Stok Tersedia', 'Penjualan (Unit)', 'Prediksi Penjualan']].head(20))

    st.subheader("Visualisasi Aktual vs Prediksi")
    fig5, ax5 = plt.subplots(figsize=(6, 4))
    ax5.plot(df_kat['Tanggal'], df_kat['Penjualan (Unit)'], label='Aktual')
    ax5.plot(df_kat['Tanggal'], df_kat['Prediksi Penjualan'], label='Prediksi', linestyle='--')
    ax5.set_title(f"Aktual vs Prediksi - {kategori}")
    ax5.set_xlabel("Tanggal")
    ax5.set_ylabel("Unit")
    ax5.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig5)
