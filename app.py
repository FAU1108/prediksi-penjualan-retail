import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# --- Config ---
st.set_page_config(layout="wide", page_title="Dashboard Prediksi Permintaan Retail")

# --- Load data ---
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
df.columns = df.columns.str.strip()
df.rename(columns={"Penjualan (Unit)": "Jumlah Penjualan"}, inplace=True)

# --- Preprocessing ---
df["Harga Satuan"] = pd.to_numeric(df["Harga Satuan"], errors="coerce")
df["Stok Tersedia"] = pd.to_numeric(df["Stok Tersedia"], errors="coerce")
df["Jumlah Penjualan"] = pd.to_numeric(df["Jumlah Penjualan"], errors="coerce")
df.dropna(inplace=True)

df_encoded = pd.get_dummies(df, columns=["Kategori Produk"], drop_first=True)

X = df_encoded[["Harga Satuan", "Stok Tersedia"] + [col for col in df_encoded.columns if "Kategori Produk_" in col]]
y = df_encoded["Jumlah Penjualan"]

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- Statsmodels for classical assumptions ---
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train.astype(float), X_train_sm.astype(float)).fit()
residuals = model_sm.resid

# --- Evaluation Metrics ---
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# --- Uji Shapiro-Wilk (Normalitas Residual) ---
shapiro_stat, shapiro_p = shapiro(residuals)

# --- Uji Breusch-Pagan (Homoskedastisitas) ---
bp_test = het_breuschpagan(residuals, model_sm.model.exog)
bp_pvalue = bp_test[1]

# --- VIF ---
vif_df = pd.DataFrame()
vif_df["Fitur"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# --- Sidebar Prediksi Manual ---
st.sidebar.header("📥 Prediksi Manual")
harga_input = st.sidebar.number_input("Harga Satuan", value=25000, step=1000)
stok_input = st.sidebar.slider("Stok Tersedia", min_value=0, max_value=300, value=150)
kategori_input = st.sidebar.selectbox("Kategori Produk", df["Kategori Produk"].unique())

# --- Build input row ---
input_dict = {
    "Harga Satuan": [harga_input],
    "Stok Tersedia": [stok_input],
}
for kat in [col for col in X.columns if "Kategori Produk_" in col]:
    input_dict[kat] = [1 if kat.endswith(kategori_input) else 0]

input_df = pd.DataFrame(input_dict)
pred_manual = model.predict(input_df)[0]

st.sidebar.markdown("---")
st.sidebar.metric("📈 Prediksi Penjualan", f"{pred_manual:.0f} unit")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Beranda", "Evaluasi Model", "Uji Statistik", "Visualisasi"])

with tab1:
    st.title("📊 Dashboard Prediksi Permintaan Produk Retail")
    st.markdown("**Judul:** Penerapan Business Intelligence untuk P
