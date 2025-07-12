
import streamlit as st
import joblib
import pandas as pd

# Load model regresi yang sudah dilatih
model = joblib.load("model_regresi.pkl")

# Judul aplikasi
st.title("Prediksi Penjualan Produk Retail")
st.subheader("Jakarta Timur - Model Regresi Linear")

# Input pengguna
stok = st.number_input("Masukkan stok tersedia", min_value=0, max_value=500, step=1)
kategori = st.selectbox("Pilih kategori produk", [
    "Elektronik", "Kebutuhan Rumah", "Makanan", "Minuman", "Perawatan Pribadi"
])

# Siapkan data input sesuai dummy variabel
input_data = {
    "Stok Tersedia": stok,
    "Kategori Produk_Kebutuhan Rumah": 1 if kategori == "Kebutuhan Rumah" else 0,
    "Kategori Produk_Makanan": 1 if kategori == "Makanan" else 0,
    "Kategori Produk_Minuman": 1 if kategori == "Minuman" else 0,
    "Kategori Produk_Perawatan Pribadi": 1 if kategori == "Perawatan Pribadi" else 0
}
input_df = pd.DataFrame([input_data])

# Tombol prediksi
if st.button("Prediksi Penjualan"):
    hasil = model.predict(input_df)[0]
    st.success(f"Prediksi penjualan: {hasil:.0f} unit")
