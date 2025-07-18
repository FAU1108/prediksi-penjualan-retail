import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import shapiro

st.set_page_config(page_title="Dashboard Analisis Permintaan", layout="wide")
st.title("📊 Dashboard Analisis Permintaan Produk Retail")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")

df = load_data()

# Persiapan data
df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df.rename(columns={
    "Kategori Produk": "Kategori",
    "Penjualan (Unit)": "Unit_Terjual",
    "Stok Tersedia": "Stok_Tersedia"
}, inplace=True)

# One-hot encoding untuk kategori
df = pd.get_dummies(df, columns=["Kategori"], drop_first=True)

# Fitur dan target
X = df.drop(columns=["Tanggal", "Lokasi", "Harga Satuan", "Unit_Terjual"], errors="ignore")
y = df["Unit_Terjual"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

with st.expander("📈 Evaluasi Model"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{mae:.2f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("R²", f"{r2:.3f}")

with st.expander("📉 Visualisasi Prediksi vs Aktual"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    st.pyplot(fig)

with st.expander("📌 Uji Asumsi Klasik (Normalitas Residual)"):
    residuals = y_test - y_pred
    stat, p = shapiro(residuals)
    st.write(f"Shapiro-Wilk p-value: `{p:.4f}`")

    fig2, ax2 = plt.subplots()
    ax2.hist(residuals, bins=20, edgecolor='k')
    ax2.set_title("Distribusi Residual")
    st.pyplot(fig2)

    if p > 0.05:
        st.success("✅ Residual berdistribusi normal (lolos uji normalitas)")
    else:
        st.error("❌ Residual tidak normal (gagal uji normalitas)")

with st.expander("📐 Uji Signifikansi (F & T Test)"):
    # Hanya ambil kolom numerik untuk OLS
    X2 = sm.add_constant(X_train.select_dtypes(include=[np.number]))
    ols = sm.OLS(y_train, X2).fit()
    st.text(ols.summary())

    if ols.f_pvalue < 0.05:
        st.success(f"✅ Uji F signifikan (p = {ols.f_pvalue:.4f})")
    else:
        st.error(f"❌ Uji F tidak signifikan (p = {ols.f_pvalue:.4f})")
