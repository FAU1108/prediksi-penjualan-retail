import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro
import statsmodels.api as sm

st.set_page_config(page_title="Prediksi Permintaan Retail Jakarta Timur", layout="wide")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail – Jakarta Timur")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")

df = load_data()

# Tampilkan kolom untuk debug
st.write("📋 Kolom dalam dataset:", df.columns.tolist())

# Deteksi nama kolom kategori secara otomatis
possible_kategori_cols = ['Kategori_Produk', 'Kategori', 'kategori_produk']
kategori_col = next((col for col in possible_kategori_cols if col in df.columns), None)

if kategori_col is None:
    st.error("❌ Kolom kategori produk tidak ditemukan dalam data.")
    st.stop()

# Tampilkan data
st.subheader("🗃️ Dataset Penjualan Retail")
st.dataframe(df.head())

# Pra-pemrosesan
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df[kategori_col] = df[kategori_col].astype(str)

# One-Hot Encoding untuk kategori
df_encoded = pd.get_dummies(df, columns=[kategori_col], drop_first=True)

# Fitur dan target
required_columns = ['Tanggal', 'Nama_Produk', 'Unit_Terjual']
missing = [col for col in required_columns if col not in df_encoded.columns]
if missing:
    st.error(f"❌ Kolom berikut tidak ditemukan: {missing}")
    st.stop()

X = df_encoded.drop(columns=['Tanggal', 'Nama_Produk', 'Unit_Terjual'])
y = df_encoded['Unit_Terjual']

# Handle NaN jika ada
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

st.subheader("📈 Evaluasi Model Regresi Linear")
col1, col2, col3, col4 = st.columns(4)
col1.metric("MSE", f"{mse:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("MAE", f"{mae:.2f}")
col4.metric("R-squared (R²)", f"{r2:.3f}")

# Visualisasi prediksi vs aktual
st.subheader("🔍 Visualisasi: Aktual vs Prediksi")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title("Prediksi vs Aktual")
st.pyplot(fig)

# Uji asumsi klasik
st.subheader("📌 Uji Asumsi Klasik")
residuals = y_test - y_pred

fig2, ax2 = plt.subplots()
ax2.scatter(y_pred, residuals)
ax2.axhline(0, color='red', linestyle='--')
ax2.set_title("Scatter Residual: Uji Linearitas & Homoskedastisitas")
ax2.set_xlabel("Prediksi")
ax2.set_ylabel("Residual")
st.pyplot(fig2)

shapiro_stat, shapiro_p = shapiro(residuals)
st.markdown(f"**Uji Normalitas (Shapiro-Wilk)**: p-value = `{shapiro_p:.4f}` {'✅ Normal' if shapiro_p > 0.05 else '❌ Tidak Normal'}")

# Uji F dan T
X2 = sm.add_constant(X_train)
est = sm.OLS(y_train, X2).fit()
st.subheader("📊 Statistik Regresi: Uji F & T")
st.text(est.summary())

# Prediksi interaktif
st.subheader("🔮 Prediksi Permintaan Manual")
stok = st.number_input("Jumlah stok tersedia", min_value=0, value=10)

# Dropdown kategori (dari data asli)
kategori_list = df[kategori_col].unique()
kategori_input = st.selectbox("Kategori Produk", kategori_list)

# Buat input satu baris untuk prediksi
input_data = {'Stok_Tersedia': stok}
for col in df_encoded.columns:
    if col.startswith(f"{kategori_col}_"):
        input_data[col] = 1 if col == f"{kategori_col}_{kategori_input}" else 0

input_df = pd.DataFrame([input_data])
input_df = input_df.reindex(columns=X.columns, fill_value=0)

if st.button("Prediksi"):
    hasil_prediksi = model.predict(input_df)[0]
    st.success(f"📦 Prediksi unit terjual: {hasil_prediksi:.0f} unit")
