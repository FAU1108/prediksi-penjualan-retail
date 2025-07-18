import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro
import statsmodels.api as sm

st.set_page_config(page_title="Dashboard Prediksi Retail", layout="wide")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail – Jakarta Timur")

@st.cache_data
def load_data():
    return pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")

df = load_data()

# Rename kolom agar seragam
df.rename(columns={
    "Kategori Produk": "Kategori",
    "Penjualan (Unit)": "Unit_Terjual",
    "Stok Tersedia": "Stok_Tersedia"
}, inplace=True)

df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Kategori'] = df['Kategori'].astype(str)

with st.expander("🗃️ Data Penjualan"):
    st.write("📋 Kolom:", df.columns.tolist())
    st.dataframe(df)

# Filter Kategori
st.sidebar.header("🔍 Filter Data")
selected_kategori = st.sidebar.selectbox("Pilih Kategori Produk", df['Kategori'].unique())
df_filtered = df[df['Kategori'] == selected_kategori]

# One-hot encoding
df_encoded = pd.get_dummies(df_filtered, columns=['Kategori'], drop_first=True)

# Fitur dan target
X = df_encoded.drop(columns=['Tanggal', 'Lokasi', 'Harga Satuan', 'Unit_Terjual'], errors='ignore')
y = df_encoded['Unit_Terjual']
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

with st.expander("📈 Evaluasi Model"):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MSE", f"{mse:.2f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    col4.metric("R²", f"{r2:.3f}")

with st.expander("📊 Visualisasi Prediksi vs Aktual"):
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Aktual")
    ax.set_ylabel("Prediksi")
    st.pyplot(fig)

with st.expander("📌 Uji Asumsi Klasik"):
    residuals = y_test - y_pred
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_pred, residuals)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_title("Scatter Residual")
    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("Residual")
    st.pyplot(fig2)

    shapiro_stat, shapiro_p = shapiro(residuals)
    st.markdown(f"**Uji Normalitas (Shapiro-Wilk)**: p-value = `{shapiro_p:.4f}` {'✅ Normal' if shapiro_p > 0.05 else '❌ Tidak Normal'}")

with st.expander("📐 Uji F dan Uji T"):
    X2 = sm.add_constant(X_train)
    est = sm.OLS(y_train, X2).fit()
    st.text(est.summary())

with st.expander("🔮 Prediksi Permintaan Manual"):
    stok_input = st.number_input("Masukkan Stok Tersedia", min_value=0)
    kategori_input = st.selectbox("Pilih Kategori", df['Kategori'].unique())

    input_data = {'Stok_Tersedia': stok_input}
    for col in df_encoded.columns:
        if col.startswith("Kategori_"):
            input_data[col] = 1 if col == f"Kategori_{kategori_input}" else 0

    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    if st.button("Prediksi"):
        hasil = model.predict(input_df)[0]
        st.success(f"📦 Prediksi penjualan untuk kategori **{kategori_input}** adalah: `{hasil:.0f}` unit")
