import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import shapiro

# Load data
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Bulan'] = df['Tanggal'].dt.month
df['Kategori Produk'] = df['Kategori Produk'].astype(str)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Kategori Produk"], prefix="Kategori")
X = df_encoded.drop(columns=["Tanggal", "Lokasi", "Penjualan (Unit)"])
y = df_encoded["Penjualan (Unit)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Residuals
residuals = y_test - y_pred

# Layout
st.set_page_config(page_title="Prediksi Penjualan", layout="wide")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail – Jakarta Timur")

# === UJI ASUMSI KLASIK ===
with st.expander("🧪 Uji Asumsi Klasik"):
    st.subheader("1. Uji Linearitas (Scatter Residual vs Nilai Prediksi)")
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Nilai Prediksi")
    ax.set_ylabel("Residual")
    ax.set_title("Scatter Residual vs Nilai Prediksi (Linearitas)")
    st.pyplot(fig)
    st.info("✅ Lolos jika tidak terlihat pola yang jelas.")

    st.subheader("2. Uji Normalitas Residual (Shapiro-Wilk Test)")
    stat, p = shapiro(residuals)
    st.write(f"Shapiro-Wilk statistic: `{stat:.4f}`")
    st.write(f"p-value: `{p:.4f}`")
    if p > 0.05:
        st.success("✅ Residual berdistribusi normal (lolos uji normalitas)")
    else:
        st.error("❌ Residual tidak normal")

    st.subheader("3. Uji Homoskedastisitas (Breusch-Pagan)")
    train_pred = model.predict(X_train)
    train_residuals = y_train - train_pred
    X_bp = sm.add_constant(X_train.select_dtypes(include=[np.number]))
    bp_test = het_breuschpagan(train_residuals, X_bp)
    p_bp = bp_test[1]
    st.write(f"Breusch-Pagan p-value: `{p_bp:.4f}`")
    if p_bp > 0.05:
        st.success("✅ Homoskedastisitas terpenuhi (residual konstan)")
    else:
        st.error("❌ Terjadi heteroskedastisitas")

# === UJI SIGNIFIKANSI ===
with st.expander("📊 Uji Signifikansi Model"):
    st.subheader("Uji F dan Uji t")
    X_train_ols = sm.add_constant(X_train).astype(float)
    y_train_ols = y_train.astype(float)
    ols = sm.OLS(y_train_ols, X_train_ols).fit()
    f_pvalue = ols.f_pvalue
    t_pvalues = ols.pvalues[1:]  # tanpa intercept

    st.write(f"Uji F p-value: `{f_pvalue:.4f}`")
    if f_pvalue < 0.05:
        st.success("✅ Model signifikan secara simultan (Uji F)")
    else:
        st.error("❌ Model tidak signifikan (Uji F)")

    if all(t_pvalues < 0.05):
        st.success("✅ Semua variabel signifikan secara parsial (Uji t)")
    else:
        st.warning("⚠️ Beberapa variabel tidak signifikan (Uji t)")

# === EVALUASI MODEL ===
with st.expander("📈 Evaluasi Model"):
    st.subheader("Evaluasi Kinerja")
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    st.write(f"MAE: `{mae:.2f}`")
    st.write(f"MSE: `{mse:.2f}`")
    st.write(f"RMSE: `{rmse:.2f}`")
    st.write(f"R² Score: `{r2:.2f}`")

# === VISUALISASI TREN PENJUALAN ===
with st.expander("📅 Visualisasi Tren Penjualan per Bulan"):
    st.subheader("Rata-rata Penjualan Tiap Bulan")
    bulan_map = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
        5: "Mei", 6: "Jun", 7: "Jul", 8: "Agu",
        9: "Sep", 10: "Okt", 11: "Nov", 12: "Des"
    }
    df['Bulan_Nama'] = df['Bulan'].map(bulan_map)
    trend_df = df.groupby('Bulan_Nama')['Penjualan (Unit)'].mean().reindex(
        ['Jan', 'Feb', 'Mar', 'Apr', 'Mei', 'Jun', 'Jul', 'Agu', 'Sep', 'Okt', 'Nov', 'Des']
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(trend_df.index, trend_df.values, marker='o', linestyle='-', color='blue')
    ax.set_title("Tren Rata-rata Penjualan per Bulan")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata Penjualan")
    ax.grid(True)
    st.pyplot(fig)
