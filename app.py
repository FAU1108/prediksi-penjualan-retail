# === IMPORTS ===
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor

# === PAGE CONFIG ===
st.set_page_config(layout="wide", page_title="Dashboard Prediksi Penjualan Retail")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail")

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Stok Tersedia'] = pd.to_numeric(df['Stok Tersedia'], errors='coerce')
    df['Harga Satuan'] = pd.to_numeric(df['Harga Satuan'], errors='coerce')
    df['Penjualan (Unit)'] = pd.to_numeric(df['Penjualan (Unit)'], errors='coerce')
    df.dropna(subset=['Stok Tersedia', 'Harga Satuan', 'Penjualan (Unit)'], inplace=True)
    return df

df = load_data()
st.subheader("Cuplikan Data")
st.dataframe(df.head())

# === PREPROCESSING ===
df_encoded = pd.get_dummies(df, columns=['Kategori Produk'], drop_first=True)
X = df_encoded.drop(columns=['Tanggal', 'Lokasi', 'Penjualan (Unit)'])
y = df_encoded['Penjualan (Unit)']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL TRAINING ===
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# === METRICS ===
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# === UJI ASUMSI ===
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
resid = model_sm.resid

shapiro_stat, shapiro_pvalue = shapiro(resid)
bp_test = het_breuschpagan(resid, model_sm.model.exog)
bp_pvalue = bp_test[1]

vif_df = pd.DataFrame()
vif_df['Fitur'] = X_train.columns
vif_df['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# === INPUT PREDIKSI ===
st.sidebar.header("🧮 Simulasi Prediksi")
kategori = st.sidebar.selectbox("Kategori Produk", df['Kategori Produk'].unique())
harga = st.sidebar.number_input("Harga Satuan", min_value=1000, max_value=100000, value=25000)
stok = st.sidebar.slider("Stok Tersedia", 0, 500, 100)

input_df = pd.DataFrame({
    'Harga Satuan': [harga],
    'Stok Tersedia': [stok],
})

for cat in X.columns:
    if "Kategori Produk_" in cat:
        input_df[cat] = 1 if cat == f"Kategori Produk_{kategori}" else 0

for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]  # urutkan
prediksi = model.predict(input_df)[0]

st.sidebar.metric("📈 Prediksi Penjualan", f"{prediksi:.0f} unit")

# === TABS ===
tab1, tab2, tab3 = st.tabs(["Evaluasi Model", "Uji Asumsi", "Visualisasi"])

with tab1:
    st.subheader("Evaluasi Model")
    st.metric("R-squared", f"{r2:.3f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("RMSE", f"{rmse:.2f}")

with tab2:
    st.subheader("Uji Asumsi Klasik")
    col1, col2 = st.columns(2)

    with col1:
        st.write("### Uji Normalitas (Shapiro-Wilk)")
        st.write(f"P-value: {shapiro_pvalue:.4f}")
        st.success("Lolos" if shapiro_pvalue > 0.05 else "Tidak Lolos")

        st.write("### Uji Homoskedastisitas (Breusch-Pagan)")
        st.write(f"P-value: {bp_pvalue:.4f}")
        st.success("Lolos" if bp_pvalue > 0.05 else "Tidak Lolos")

    with col2:
        st.write("### Multikolinearitas (VIF)")
        st.dataframe(vif_df)

with tab3:
    st.subheader("Distribusi Residual")
    fig, ax = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Aktual vs Prediksi")
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Aktual")
    ax2.set_ylabel("Prediksi")
    st.pyplot(fig2)

st.caption("Dashboard ini menyatukan simulasi prediksi + validasi asumsi model seperti di laporan PPT.")
