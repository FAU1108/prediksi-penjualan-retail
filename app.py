import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.formula.api import ols
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

st.set_page_config(page_title="Dashboard Prediksi Permintaan Retail", layout="wide")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail di Jakarta Timur")

# Load data
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Bulan'] = df['Tanggal'].dt.to_period('M')

# Visualisasi penjualan per bulan
st.header("1. Penjualan per Bulan")
monthly_sales = df.groupby('Bulan')["Penjualan (Unit)"].sum()
fig, ax = plt.subplots()
monthly_sales.plot(kind='bar', color='skyblue', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Preprocessing untuk model
X = df[["Harga Satuan", "Stok Tersedia"]]
y = df["Penjualan (Unit)"]
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()

# Uji Normalitas
st.header("2. Uji Asumsi Klasik")
st.subheader("2.1 Uji Normalitas (Shapiro-Wilk)")
residuals = model.resid
shapiro_test = shapiro(residuals)
st.write(f"Statistik Shapiro: {shapiro_test.statistic:.4f}, p-value: {shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue > 0.05:
    st.success("Residual terdistribusi normal (p > 0.05)")
else:
    st.error("Residual tidak terdistribusi normal (p <= 0.05)")

# Uji Multikolinearitas
st.subheader("2.2 Uji Multikolinearitas (VIF)")
vif_data = pd.DataFrame()
vif_data['feature'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
st.dataframe(vif_data)

# Uji Homoskedastisitas
st.subheader("2.3 Uji Homoskedastisitas (Breusch-Pagan)")
_, pval, __, f_pval = het_breuschpagan(residuals, X)
st.write(f"p-value: {pval:.4f}")
if pval > 0.05:
    st.success("Tidak terjadi heteroskedastisitas (p > 0.05)")
else:
    st.error("Terdapat indikasi heteroskedastisitas (p <= 0.05)")

# Uji Signifikansi
st.header("3. Uji Signifikansi Model")
st.text(model.summary())

# Evaluasi Model
st.header("4. Evaluasi Model")
pred = model.predict(X)
mse = mean_squared_error(y, pred)
rmse = math.sqrt(mse)
mae = mean_absolute_error(y, pred)
r2 = model.rsquared

st.write(f"**R-squared:** {r2:.4f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# Input Manual
st.header("5. Prediksi Permintaan Manual")
st.write("Masukkan data untuk memprediksi permintaan produk:")
harga_input = st.number_input("Harga Satuan", min_value=0.0, step=100.0)
stok_input = st.number_input("Stok Tersedia", min_value=0.0, step=1.0)

if st.button("Prediksi"):
    X_new = pd.DataFrame({
        "const": [1],
        "Harga Satuan": [harga_input],
        "Stok Tersedia": [stok_input]
    })
    prediksi = model.predict(X_new)[0]
    st.success(f"Prediksi Permintaan: {prediksi:.2f} unit")
