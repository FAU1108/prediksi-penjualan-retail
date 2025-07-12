import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load model dan data
model = joblib.load("model_regresi.pkl")
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")

# Title
st.title("📊 Prediksi Penjualan Produk Retail")
st.markdown("Aplikasi prediksi ini dibangun menggunakan model regresi linear dan diterapkan pada data penjualan retail di Jakarta Timur.")

# Visualisasi Tren Penjualan Aktual
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df_sorted = df.sort_values('Tanggal')

st.subheader("Tren Penjualan Aktual (2024)")
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_sorted, x='Tanggal', y='Penjualan (Unit)', ax=ax)
plt.xticks(rotation=45)
plt.title("Tren Penjualan Harian")
plt.xlabel("Tanggal")
plt.ylabel("Unit Terjual")
plt.tight_layout()
st.pyplot(fig)

# Input user
st.subheader("Prediksi Penjualan Berdasarkan Input")
stok = st.number_input("Masukkan jumlah stok tersedia", min_value=0, max_value=500, step=1)
kategori = st.selectbox("Pilih kategori produk", ["Elektronik", "Kebutuhan Rumah", "Makanan", "Minuman", "Perawatan Pribadi"])

# Encoding input
input_data = {
    "Stok Tersedia": stok,
    "Kategori Produk_Kebutuhan Rumah": 1 if kategori == "Kebutuhan Rumah" else 0,
    "Kategori Produk_Makanan": 1 if kategori == "Makanan" else 0,
    "Kategori Produk_Minuman": 1 if kategori == "Minuman" else 0,
    "Kategori Produk_Perawatan Pribadi": 1 if kategori == "Perawatan Pribadi" else 0
}
input_df = pd.DataFrame([input_data])

# Prediksi
if st.button("Prediksi Penjualan"):
    hasil = model.predict(input_df)[0]
    st.success(f"Prediksi penjualan: {hasil:.0f} unit")

# Simulasi Tren Prediksi
st.subheader("Simulasi Tren Prediksi 30 Hari")
stok_simulasi = st.slider("Jumlah stok untuk simulasi", 10, 200, 100)
kategori_simulasi = st.selectbox("Kategori simulasi", ["Elektronik", "Kebutuhan Rumah", "Makanan", "Minuman", "Perawatan Pribadi"])

# Generate prediksi 30 hari ke depan
dates = pd.date_range(start='2024-08-01', periods=30)
data_simulasi = []

for tgl in dates:
    sim_input = {
        "Stok Tersedia": stok_simulasi,
        "Kategori Produk_Kebutuhan Rumah": 1 if kategori_simulasi == "Kebutuhan Rumah" else 0,
        "Kategori Produk_Makanan": 1 if kategori_simulasi == "Makanan" else 0,
        "Kategori Produk_Minuman": 1 if kategori_simulasi == "Minuman" else 0,
        "Kategori Produk_Perawatan Pribadi": 1 if kategori_simulasi == "Perawatan Pribadi" else 0
    }
    df_sim = pd.DataFrame([sim_input])
    pred = model.predict(df_sim)[0]
    data_simulasi.append({"Tanggal": tgl, "Prediksi": pred})

df_simulasi = pd.DataFrame(data_simulasi)

# Plot tren prediksi
fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.lineplot(data=df_simulasi, x='Tanggal', y='Prediksi', ax=ax2)
plt.xticks(rotation=45)
plt.title("Tren Prediksi Penjualan 30 Hari")
plt.xlabel("Tanggal")
plt.ylabel("Prediksi Unit Terjual")
plt.tight_layout()
st.pyplot(fig2)
