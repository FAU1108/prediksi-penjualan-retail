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
    st.markdown("**Judul:** Penerapan Business Intelligence untuk Prediksi Permintaan Produk Retail di Jakarta Timur Menggunakan Algoritma Regresi Linear")

    st.subheader("Contoh Data")
    st.dataframe(df.head())

    st.subheader("Input Prediksi")
    st.write("Simulasikan permintaan berdasarkan input harga, stok, dan kategori produk.")

with tab2:
    st.header("🎯 Evaluasi Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:.2f}")
    col3.metric("MAE", f"{mae:.2f}")
    col4.metric("MSE", f"{mse:.2f}")

with tab3:
    st.header("📌 Uji Asumsi Klasik & Signifikansi")

    st.subheader("1. Uji Normalitas (Shapiro-Wilk)")
    st.write(f"P-value: `{shapiro_p:.4f}`")
    st.success("Lolos uji normalitas" if shapiro_p > 0.05 else "Tidak lolos uji normalitas")

    st.subheader("2. Uji Homoskedastisitas (Breusch-Pagan)")
    st.write(f"P-value: `{bp_pvalue:.4f}`")
    st.success("Lolos homoskedastisitas" if bp_pvalue > 0.05 else "Tidak lolos homoskedastisitas")

    st.subheader("3. Uji Multikolinearitas (VIF)")
    st.dataframe(vif_df)

    st.subheader("4. Uji Signifikansi Koefisien (T-Test)")
    st.dataframe(model_sm.summary2().tables[1][["Coef.", "P>|t|"]])

    st.subheader("5. Uji Signifikansi Model (F-Test)")
    st.write(f"F-statistic: `{model_sm.fvalue:.2f}`, p-value: `{model_sm.f_pvalue:.4f}`")
    st.success("Model signifikan" if model_sm.f_pvalue < 0.05 else "Model tidak signifikan")

with tab4:
    st.header("📈 Visualisasi")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Histogram Residual")
        fig1, ax1 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.subheader("Scatter Aktual vs Prediksi")
        fig2, ax2 = plt.subplots()
        ax2.scatter(y_test, y_pred, alpha=0.7)
        ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
        ax2.set_xlabel("Aktual")
        ax2.set_ylabel("Prediksi")
        st.pyplot(fig2)

st.caption("📌 Dibuat untuk kebutuhan sidang tugas akhir.")
