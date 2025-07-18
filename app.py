import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import statsmodels.api as sm

st.set_page_config(layout="wide", page_title="Dashboard Prediksi Penjualan Retail")
st.title("📈 Dashboard Prediksi Penjualan Produk Retail")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Harga Satuan'] = pd.to_numeric(df['Harga Satuan'], errors='coerce')
    df['Stok Tersedia'] = pd.to_numeric(df['Stok Tersedia'], errors='coerce')
    df['Jumlah Penjualan'] = pd.to_numeric(df['Jumlah Penjualan'], errors='coerce')
    df.dropna(inplace=True)
    return df

df = load_data()
st.dataframe(df.head())

# Preprocessing
X = df[['Harga Satuan', 'Stok Tersedia']]
y = df['Jumlah Penjualan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Residual
residuals = y_test - y_pred

# Metrics
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

# Statsmodels model for assumption tests
X_train_sm = sm.add_constant(X_train)
X_train_sm = X_train_sm.astype(float).dropna()
y_train_sm = y_train[X_train_sm.index].astype(float)
model_sm = sm.OLS(y_train_sm, X_train_sm).fit()

# Uji Normalitas (Shapiro-Wilk)
shapiro_stat, shapiro_p = shapiro(model_sm.resid)

# Uji Homoskedastisitas
bp_stat, bp_pval, _, _ = het_breuschpagan(model_sm.resid, model_sm.model.exog)

# Uji Multikolinearitas
X_train_vif = X_train.copy()
X_train_vif = X_train_vif.astype(float).dropna()
vif_df = pd.DataFrame()
vif_df["Fitur"] = X_train_vif.columns
vif_df["VIF"] = [variance_inflation_factor(X_train_vif.values, i) for i in range(X_train_vif.shape[1])]

# Sidebar input prediksi manual
st.sidebar.header("Input Prediksi Manual")
kategori = st.sidebar.selectbox("Kategori Produk", df['Kategori Produk'].unique())
harga_input = st.sidebar.number_input("Harga Satuan", min_value=0, value=int(df['Harga Satuan'].mean()))
stok_input = st.sidebar.slider("Stok Tersedia", int(df['Stok Tersedia'].min()), int(df['Stok Tersedia'].max()), int(df['Stok Tersedia'].mean()))

# Prediksi dari input manual
X_input = pd.DataFrame({
    'Harga Satuan': [harga_input],
    'Stok Tersedia': [stok_input]
})
pred_manual = model.predict(X_input)[0]

st.sidebar.metric("📦 Prediksi Penjualan", f"{pred_manual:.0f} unit")

# Tabs
menu = st.selectbox("Pilih Menu", ["Evaluasi Model", "Uji Asumsi", "Visualisasi"])

if menu == "Evaluasi Model":
    st.subheader("📊 Evaluasi Model")
    col1, col2, col3 = st.columns(3)
    col1.metric("R-squared", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")

elif menu == "Uji Asumsi":
    st.subheader("📋 Uji Asumsi Klasik")
    st.write("### 1. Uji Normalitas Residual (Shapiro-Wilk)")
    st.write(f"P-value: `{shapiro_p:.4f}`")
    st.success("✅ Lolos" if shapiro_p > 0.05 else "❌ Tidak Lolos")

    st.write("### 2. Uji Homoskedastisitas (Breusch-Pagan)")
    st.write(f"P-value: `{bp_pval:.4f}`")
    st.success("✅ Lolos" if bp_pval > 0.05 else "❌ Tidak Lolos")

    st.write("### 3. Uji Multikolinearitas (VIF)")
    st.dataframe(vif_df)

elif menu == "Visualisasi":
    st.subheader("📈 Visualisasi")
    fig1, ax1 = plt.subplots(figsize=(6, 3))
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Distribusi Residual")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.scatter(y_test, y_pred, alpha=0.7)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Aktual")
    ax2.set_ylabel("Prediksi")
    ax2.set_title("Aktual vs Prediksi")
    st.pyplot(fig2)

st.caption("© 2025 - Dashboard Prediksi Penjualan Produk Retail dengan Regresi Linear")
