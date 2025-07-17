import streamlit as st
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

# ========================== 1. Visualisasi Penjualan Bulanan ==============================
st.header("1. Visualisasi Tren Penjualan per Bulan")
monthly_sales = df.groupby('Bulan')["Penjualan (Unit)"].sum()
fig1, ax1 = plt.subplots()
monthly_sales.plot(kind='bar', color='skyblue', ax=ax1)
ax1.set_xlabel("Bulan")
ax1.set_ylabel("Jumlah Penjualan")
ax1.set_title("Total Penjualan per Bulan")
plt.xticks(rotation=45)
st.pyplot(fig1)

# ========================== 2. Pelatihan Model ============================================
X = df[["Harga Satuan", "Stok Tersedia"]]
y = df["Penjualan (Unit)"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
pred = model.predict(X)
residuals = y - pred

# ========================== 3. Evaluasi Model ============================================
st.header("2. Evaluasi Kinerja Model")
mse = mean_squared_error(y, pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y, pred)
r2 = model.rsquared

col1, col2, col3, col4 = st.columns(4)
col1.metric("R-squared", f"{r2:.3f}", help="Proporsi variasi penjualan yang dapat dijelaskan oleh model")
col2.metric("MSE", f"{mse:.2f}")
col3.metric("RMSE", f"{rmse:.2f}")
col4.metric("MAE", f"{mae:.2f}")

# ========================== 4. Uji Asumsi Klasik ============================================
st.header("3. Uji Asumsi Klasik")

with st.expander("3.1 Uji Linearitas (Scatterplot Residual vs Prediksi)"):
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=pred, y=residuals, ax=ax2)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("Residual")
    ax2.set_title("Scatterplot Prediksi vs Residual")
    st.pyplot(fig2)
    st.markdown("✅ Pola acak di sekitar nol menunjukkan linearitas terpenuhi.")

with st.expander("3.2 Uji Normalitas (Shapiro-Wilk)"):
    shapiro_test = shapiro(residuals)
    st.write(f"p-value: {shapiro_test.pvalue:.4f}")
    if shapiro_test.pvalue > 0.05:
        st.success("✅ Residual terdistribusi normal (p > 0.05)")
    else:
        st.error("❌ Residual tidak normal")

with st.expander("3.3 Uji Homoskedastisitas (Breusch-Pagan)"):
    _, pval_bp, __, __ = het_breuschpagan(residuals, X)
    st.write(f"p-value: {pval_bp:.4f}")
    if pval_bp > 0.05:
        st.success("✅ Tidak terdapat heteroskedastisitas")
    else:
        st.error("❌ Ada indikasi heteroskedastisitas")

with st.expander("3.4 Uji Multikolinearitas (VIF)"):
    vif_df = pd.DataFrame()
    vif_df['Fitur'] = X.columns
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_df)
    if (vif_df['VIF'] > 5).any():
        st.error("❌ Ada indikasi multikolinearitas tinggi")
    else:
        st.success("✅ Tidak ada multikolinearitas yang mengganggu")

# ========================== 5. Uji Signifikansi ============================================
st.header("4. Uji Signifikansi Model")
st.markdown("**Ringkasan Hasil Regresi Linear:**")
st.text(model.summary())
st.markdown("- **Uji F:** Model signifikan jika p-value < 0.05")
st.markdown("- **Uji t:** Menilai pengaruh masing-masing variabel terhadap target")

# ========================== 6. Visualisasi Aktual vs Prediksi ===============================
st.header("5. Visualisasi Hasil Prediksi")
fig3, ax3 = plt.subplots()
ax3.scatter(y, pred, alpha=0.5, color='green')
ax3.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax3.set_xlabel("Aktual")
ax3.set_ylabel("Prediksi")
ax3.set_title("Perbandingan Penjualan Aktual vs Prediksi")
st.pyplot(fig3)
st.markdown("Titik-titik yang mendekati garis merah menunjukkan prediksi mendekati nilai aktual.")

# ========================== 7. Input Prediksi Manual ========================================
st.header("6. Prediksi Permintaan (Input Manual)")
with st.form("manual_input"):
    harga = st.number_input("Harga Satuan", min_value=0.0, step=100.0)
    stok = st.number_input("Stok Tersedia", min_value=0.0, step=1.0)
    submit = st.form_submit_button("Prediksi")
    if submit:
        new_X = pd.DataFrame({"const": [1], "Harga Satuan": [harga], "Stok Tersedia": [stok]})
        pred_manual = model.predict(new_X)[0]
        st.success(f"Prediksi Penjualan: {pred_manual:.2f} unit")
