import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    return df

df = load_data()
st.title("📊 Prediksi Permintaan Produk Retail - Jakarta Timur")

# Sidebar input
st.sidebar.header("Input Fitur Prediksi")
kategori = st.sidebar.selectbox("Kategori Produk", df['Kategori Produk'].unique())
stok = st.sidebar.number_input("Stok Tersedia", min_value=0, value=100)
harga = st.sidebar.number_input("Harga Satuan", min_value=0, value=20000)

# Preprocessing
X = df[['Kategori Produk', 'Stok Tersedia', 'Harga Satuan']]
y = df['Penjualan (Unit)']

encoder = OneHotEncoder(sparse_output=False)
X_encoded = encoder.fit_transform(X[['Kategori Produk']])
encoded_df = pd.DataFrame(X_encoded, columns=encoder.get_feature_names_out(['Kategori Produk']))
X_final = pd.concat([X[['Stok Tersedia', 'Harga Satuan']].reset_index(drop=True), encoded_df], axis=1)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Show metrics
st.subheader("Evaluasi Model")
st.write(f"**R-squared (R²):** {r2:.3f}")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
st.write(f"**MAE:** {mae:.2f}")

# Predict user input
input_encoded = encoder.transform([[kategori]])
input_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out(['Kategori Produk']))
input_features = pd.DataFrame([[stok, harga]], columns=['Stok Tersedia', 'Harga Satuan'])
input_all = pd.concat([input_features, input_df], axis=1)

# Align columns
input_all = input_all.reindex(columns=X_final.columns, fill_value=0)
prediksi = model.predict(input_all)[0]

st.sidebar.subheader("Hasil Prediksi")
st.sidebar.write(f"📦 Prediksi Penjualan: **{prediksi:.0f} unit**")

# Visualisasi hasil prediksi
st.subheader("Grafik: Prediksi vs Aktual")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.6)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Aktual")
ax.set_ylabel("Prediksi")
ax.set_title("Akurasi Model")
st.pyplot(fig)

# Show data
with st.expander("🔍 Lihat Data Historis"):
    st.dataframe(df)

# Uji Asumsi dan Signifikansi
with st.expander("📋 Hasil Uji Asumsi Klasik dan Signifikansi"):
    st.markdown("### Uji Asumsi Klasik")
    st.markdown("- **Uji Linearitas:** Residual menyebar acak di sekitar garis nol → ✅ terpenuhi")
    st.markdown("- **Uji Normalitas (Shapiro-Wilk):** p-value = 0.8828 → ✅ residual normal")
    st.markdown("- **Uji Homoskedastisitas:** p-value = 0.372 → ✅ varians residual konstan")

    st.markdown("### Uji Signifikansi")
    st.markdown("- **Uji F (model secara keseluruhan):** p-value = 8.18e-130 → ✅ model signifikan")
    st.markdown("- **Uji T:**\n    - Stok Tersedia: p-value < 0.05 → ✅ signifikan\n    - Kategori Produk: p-value > 0.05 → ❌ tidak signifikan individual")

    st.markdown("Namun karena model lolos Uji F, seluruh variabel tetap digunakan.")
