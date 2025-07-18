
import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import shapiro
import seaborn as sns
import matplotlib.pyplot as plt

# Layout
st.set_page_config(layout="wide")
st.title("📊 BI Dashboard – Prediksi Permintaan Produk Retail di Jakarta Timur")

# Load Dataset
df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")

# Pra-pemrosesan
# One-hot kategori
kategori_encoded = pd.get_dummies(df['Kategori Produk'], drop_first=True, dtype=int)
df_encoded = pd.concat([df[['Stok Tersedia']], kategori_encoded], axis=1)
X = df_encoded.astype(float)
y = df['Jumlah Penjualan'].astype(float)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# Evaluasi Model
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# Uji Asumsi
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

# Uji Normalitas: Shapiro-Wilk
shapiro_stat, shapiro_p = shapiro(model_sm.resid)

# Uji Homoskedastisitas: Breusch-Pagan
bp_test = het_breuschpagan(model_sm.resid, model_sm.model.exog)
bp_pvalue = bp_test[1]

# VIF
vif_df = pd.DataFrame()
vif_df["Fitur"] = X_train.columns
vif_df["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]

# Uji Signifikansi
f_pval = model_sm.f_pvalue
coefs_df = pd.DataFrame({
    "Fitur": X.columns,
    "Koefisien": model.coef_,
    "P-value (t-test)": model_sm.pvalues.drop("const")
})

# TAB 1 – Evaluasi Model
with st.expander("📌 Evaluasi Model"):
    st.metric("R-squared (R²)", f"{r2:.3f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE", f"{mae:.2f}")
    st.metric("MSE", f"{mse:.2f}")

# TAB 2 – Uji Asumsi Klasik
with st.expander("🧪 Uji Asumsi Klasik"):
    st.subheader("1. Normalitas (Shapiro-Wilk)")
    st.write(f"P-value: `{shapiro_p:.4f}`")
    if shapiro_p > 0.05:
        st.success("Residual terdistribusi normal")
    else:
        st.error("Residual tidak normal")

    st.subheader("2. Homoskedastisitas (Breusch-Pagan)")
    st.write(f"P-value: `{bp_pvalue:.4f}`")
    if bp_pvalue > 0.05:
        st.success("Tidak ada heteroskedastisitas")
    else:
        st.error("Terdapat heteroskedastisitas")

    st.subheader("3. Multikolinearitas (VIF)")
    st.dataframe(vif_df)
    if (vif_df['VIF'] > 5).any():
        st.error("Terdapat multikolinearitas tinggi")
    else:
        st.success("Multikolinearitas aman")

# TAB 3 – Uji Signifikansi Model
with st.expander("📈 Uji Signifikansi"):
    st.subheader("Uji F")
    st.write(f"P-value: `{f_pval:.4f}`")
    if f_pval < 0.05:
        st.success("Model signifikan secara keseluruhan")
    else:
        st.error("Model tidak signifikan")

    st.subheader("Uji t per Variabel")
    st.dataframe(coefs_df)

# TAB 4 – Visualisasi
with st.expander("📊 Visualisasi"):
    fig1, ax1 = plt.subplots()
    sns.histplot(residuals, kde=True, ax=ax1)
    ax1.set_title("Distribusi Residual")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.scatter(y_test, y_pred, alpha=0.5)
    ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax2.set_xlabel("Aktual")
    ax2.set_ylabel("Prediksi")
    ax2.set_title("Prediksi vs Aktual")
    st.pyplot(fig2)

# TAB 5 – Simulasi Prediksi Manual
with st.expander("🧮 Simulasi Prediksi Manual"):
    kategori_list = df['Kategori Produk'].unique().tolist()
    stok_input = st.number_input("Stok Tersedia", value=150)
    kategori_input = st.selectbox("Kategori Produk", kategori_list)

    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0
    input_data.loc[0, 'Stok Tersedia'] = stok_input

    if f"{kategori_input}" in kategori_encoded.columns:
        input_data.loc[0, f"{kategori_input}"] = 1

    pred_manual = model.predict(input_data)[0]
    st.metric("Prediksi Penjualan", f"{pred_manual:.0f} unit")
