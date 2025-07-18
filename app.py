import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="Dashboard Analisis Permintaan", layout="wide")
st.title("📊 Dashboard Analisis Permintaan Produk Retail (Versi Notebook Asli)")

with st.expander("📥 Eksekusi Seluruh Notebook"):
    exec("""
# Mulai dari sini isi notebook kamu dieksekusi persis seperti aslinya

# Contoh kode — GANTI dengan isi kode dari notebook kamu
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")

df["Tanggal"] = pd.to_datetime(df["Tanggal"])
df = df.rename(columns={
    "Kategori Produk": "Kategori",
    "Penjualan (Unit)": "Unit_Terjual",
    "Stok Tersedia": "Stok_Tersedia"
})

# Visualisasi sederhana
fig, ax = plt.subplots()
df["Unit_Terjual"].hist(bins=20, ax=ax)
st.pyplot(fig)

# Encoding
df = pd.get_dummies(df, columns=["Kategori"], drop_first=True)

# Split data
X = df.drop(columns=["Tanggal", "Lokasi", "Harga Satuan", "Unit_Terjual"], errors="ignore")
y = df["Unit_Terjual"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluasi
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("📊 Evaluasi Model:")
st.write(f"MAE: {mae:.2f}")
st.write(f"RMSE: {rmse:.2f}")
st.write(f"R²: {r2:.3f}")
""", globals())
