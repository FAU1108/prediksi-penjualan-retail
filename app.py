import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

df = load_data()

# Setup regresi
X = sm.add_constant(df[['Harga Satuan', 'Stok Tersedia']], has_constant='add')
y = df['Penjualan (Unit)']
model = sm.OLS(y, X).fit()
y_pred = model.predict(X)
residuals = y - y_pred

# Streamlit UI
st.set_page_config(layout="wide")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail")

menu = st.sidebar.radio("Navigasi", [
    "Evaluasi Model", 
    "Uji Asumsi Klasik", 
    "Visualisasi Prediksi", 
    "Simulasi Prediksi Manual", 
    "Data Historis"
])

if menu == "Evaluasi Model":
    st.header("📌 Evaluasi Model Regresi")
    mse = np.mean((y - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y - y_pred))
    r2 = model.rsquared
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-squared", f"{r2:.3f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("MAE", f"{mae:.2f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y, y_pred, alpha=0.6, color='blue')
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    ax.set_title("Aktual vs Prediksi")
    st.pyplot(fig)

elif menu == "Uji Asumsi Klasik":
    st.header("🧪 Uji Asumsi Klasik")
    shapiro_p = shapiro(residuals)[1]
    bp_p = het_breuschpagan(residuals, X)[1]
    vif_df = pd.DataFrame()
    vif_df["Fitur"] = X.columns
    vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    st.write("### Normalitas Residual (Shapiro-Wilk)")
    st.write(f"P-Value: {shapiro_p:.4f}")
    st.success("✅ Lolos uji normalitas" if shapiro_p > 0.05 else "❌ Tidak lolos")

    st.write("### Homoskedastisitas (Breusch-Pagan)")
    st.write(f"P-Value: {bp_p:.4f}")
    st.success("✅ Lolos homoskedastisitas" if bp_p > 0.05 else "❌ Tidak lolos")

    st.write("### Multikolinearitas (VIF)")
    st.dataframe(vif_df)

    st.write("### Uji Signifikansi (F & T)")
    st.write(f"**P-Value Uji F:** {model.f_pvalue:.4e}")
    signifikan = model.pvalues < 0.05
    hasil = pd.DataFrame({
        "Fitur": model.params.index,
        "Koefisien": model.params.values,
        "P-Value": model.pvalues.values,
        "Signifikan": signifikan.values
    })
    hasil = hasil[hasil['Fitur'] != 'const']
    st.dataframe(hasil)

elif menu == "Visualisasi Prediksi":
    st.header("📈 Visualisasi Penjualan Bulanan")
    df_bulanan = df.resample('M', on='Tanggal').sum(numeric_only=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    df_bulanan['Penjualan (Unit)'].plot(kind='bar', ax=ax, color='orange')
    ax.set_title("Total Penjualan per Bulan")
    ax.set_ylabel("Unit Terjual")
    st.pyplot(fig)

elif menu == "Simulasi Prediksi Manual":
    st.header("📌 Simulasi Prediksi Manual")
    st.write("Masukkan harga dan stok untuk memprediksi penjualan.")
    harga = st.number_input("Harga Satuan", min_value=0, value=int(df['Harga Satuan'].mean()))
    stok = st.number_input("Stok Tersedia", min_value=0, value=int(df['Stok Tersedia'].mean()))

    input_df = pd.DataFrame({
        "const": [1],
        "Harga Satuan": [harga],
        "Stok Tersedia": [stok]
    })

    pred = model.predict(input_df)[0]
    st.metric("📈 Prediksi Penjualan", f"{pred:.0f} unit")

elif menu == "Data Historis":
    st.header("📂 Data Historis Penjualan")
    st.dataframe(df)
