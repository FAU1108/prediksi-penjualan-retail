import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import matplotlib.pyplot as plt
import seaborn as sns

# Setup layout
st.set_page_config(layout="wide", page_title="Dashboard Prediksi Permintaan Retail")
st.title("📊 Dashboard Prediksi Permintaan Produk Retail")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df.columns = df.columns.str.strip()
    df.rename(columns={'Penjualan (Unit)': 'Jumlah Penjualan'}, inplace=True)
    return df

df = load_data()

# Preprocessing
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df['Harga Satuan'] = pd.to_numeric(df['Harga Satuan'], errors='coerce')
df['Stok Tersedia'] = pd.to_numeric(df['Stok Tersedia'], errors='coerce')
df.dropna(inplace=True)

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Kategori Produk"], drop_first=True)

# Define X and y
X = df_encoded[['Harga Satuan', 'Stok Tersedia'] + [col for col in df_encoded.columns if "Kategori Produk_" in col]]
y = df_encoded["Jumlah Penjualan"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Classical Assumption Testing
# 1. Normalitas Residual (Shapiro-Wilk)
residuals = y_test - y_pred
shapiro_stat, shapiro_p = shapiro(residuals)

# 2. VIF
vif_df = pd.DataFrame()
vif_df["Fitur"] = X.columns
vif_df["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

# 3. Homoskedastisitas (Breusch-Pagan)
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()
bp_test = het_breuschpagan(model_sm.resid, model_sm.model.exog)
bp_pvalue = bp_test[1]

# 4. Signifikansi model
f_pvalue = model_sm.f_pvalue

# UI Layout
tab1, tab2, tab3 = st.tabs(["📈 Prediksi Manual", "📊 Evaluasi Model", "🧪 Uji Asumsi Klasik"])

with tab1:
    st.header("📈 Prediksi Manual Permintaan Produk")

    kategori = st.selectbox("Pilih Kategori Produk", df['Kategori Produk'].unique())
    harga = st.number_input("Harga Satuan (Rp)", min_value=1000, step=500)
    stok = st.slider("Stok Tersedia", int(df['Stok Tersedia'].min()), int(df['Stok Tersedia'].max()), step=1)

    # Buat input jadi sama formatnya dengan X
    input_data = pd.DataFrame({
        "Harga Satuan": [harga],
        "Stok Tersedia": [stok]
    })

    for col in [col for col in X.columns if "Kategori Produk_" in col]:
        input_data[col] = 1 if col == f'Kategori Produk_{kategori}' else 0

    prediksi_manual = model.predict(input_data)[0]
    st.metric("📊 Prediksi Penjualan", f"{int(prediksi_manual)} unit")

with tab2:
    st.header("📊 Evaluasi Model")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("R-squared", f"{r2:.3f}")
    col2.metric("MSE", f"{mse:.2f}")
    col3.metric("RMSE", f"{rmse:.2f}")
    col4.metric("MAE", f"{mae:.2f}")

    st.subheader("📉 Plot Aktual vs Prediksi")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax1.set_xlabel("Aktual")
    ax1.set_ylabel("Prediksi")
    st.pyplot(fig1)

with tab3:
    st.header("🧪 Hasil Uji Asumsi Klasik")

    st.subheader("1. Uji Normalitas Residual (Shapiro-Wilk)")
    st.write(f"**P-value:** {shapiro_p:.4f}")
    if shapiro_p > 0.05:
        st.success("Residual terdistribusi normal ✅")
    else:
        st.error("Residual tidak normal ❌")

    st.subheader("2. Uji Multikolinearitas (VIF)")
    st.dataframe(vif_df.style.format({"VIF": "{:.2f}"}))

    st.subheader("3. Uji Homoskedastisitas (Breusch-Pagan)")
    st.write(f"**P-value:** {bp_pvalue:.4f}")
    if bp_pvalue > 0.05:
        st.success("Homoskedastisitas terpenuhi ✅")
    else:
        st.error("Terdapat heteroskedastisitas ❌")

    st.subheader("4. Uji Signifikansi Model (F-Test)")
    st.write(f"**P-value:** {f_pvalue:.4f}")
    if f_pvalue < 0.05:
        st.success("Model signifikan secara statistik ✅")
    else:
        st.error("Model tidak signifikan ❌")
