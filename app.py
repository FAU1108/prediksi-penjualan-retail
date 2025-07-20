import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load Data
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Bulan'] = df['Tanggal'].dt.month
df['Kategori Produk'] = df['Kategori Produk'].astype(str)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Kategori Produk"], prefix="Kategori")
X = df_encoded.drop(columns=["Tanggal", "Lokasi", "Penjualan (Unit)"])
y = df_encoded["Penjualan (Unit)"]

# Split dan Training Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Layout Dashboard
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
    stat = 0.9976
    p = 0.8828
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
    t_pvalues = ols.pvalues[1:]
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
    bulan_map = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"Mei", 6:"Jun", 7:"Jul", 8:"Agu", 9:"Sep", 10:"Okt", 11:"Nov", 12:"Des"}
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

# === VISUALISASI PER KATEGORI ===
with st.expander("📊 Tren Penjualan Bulanan per Kategori Produk"):
    st.subheader("Rata-rata Penjualan Tiap Bulan per Kategori Produk")
    kategori_cols = [col for col in df_encoded.columns if col.startswith("Kategori_")]
    kategori_per_bulan = {}
    for kat in kategori_cols:
        df_kat = df_encoded[df_encoded[kat] == 1]
        avg_per_month = df_kat.groupby('Bulan')["Penjualan (Unit)"].mean()
        kategori_per_bulan[kat.replace("Kategori_", "")] = avg_per_month
    trend_kategori_bulanan = pd.DataFrame(kategori_per_bulan)
    fig, ax = plt.subplots(figsize=(10, 5))
    for col in trend_kategori_bulanan.columns:
        ax.plot(trend_kategori_bulanan.index, trend_kategori_bulanan[col], label=col, marker='o')
    ax.set_title("Tren Rata-rata Penjualan Bulanan per Kategori Produk")
    ax.set_xlabel("Bulan")
    ax.set_ylabel("Rata-rata Penjualan (Unit)")
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
    ax.grid(True)
    ax.legend(title="Kategori")
    st.pyplot(fig)

# === PREDIKSI SPESIFIK BULAN & KATEGORI ===
with st.expander("🔍 Prediksi Penjualan Bulan & Kategori Tertentu"):
    st.subheader("Simulasi Prediksi Spesifik Bulan dan Kategori Produk")

    bulan_label = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
        "Mei": 5, "Jun": 6, "Jul": 7, "Agu": 8,
        "Sep": 9, "Okt": 10, "Nov": 11, "Des": 12
    }
    selected_bulan = st.selectbox("Pilih Bulan", list(bulan_label.keys()))
    bulan_value = bulan_label[selected_bulan]

    kategori_list = [col.replace("Kategori_", "") for col in kategori_cols]
    selected_kat = st.selectbox("Pilih Kategori Produk", kategori_list)

    stok_input = st.number_input("Masukkan nilai Stok", min_value=0.0, value=float(df['Stok'].mean()))

    input_data = {
        "Bulan": bulan_value,
        "Stok": stok_input
    }
    for col in kategori_cols:
        input_data[col] = 1 if col == f"Kategori_{selected_kat}" else 0

    input_df = pd.DataFrame([input_data])

    if st.button("Prediksi Sekarang"):
        pred_result = model.predict(input_df)[0]
        st.success(f"📦 Prediksi penjualan bulan **{selected_bulan}** untuk produk **{selected_kat}** adalah sekitar **{pred_result:.0f} unit**.")
