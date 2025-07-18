import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import shapiro
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

# === LOAD DATA ===
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])

    # Hapus kolom nama produk, encode kategori
    df = df.drop(columns=['Nama_Produk'])
    df = pd.get_dummies(df, columns=["Kategori"], drop_first=True)

    return df

df = load_data()

# === PERSIAPAN MODEL ===
X = df.drop(["Jumlah_Terjual", "Tanggal"], axis=1)
y = df["Jumlah_Terjual"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
residuals = y_test - y_pred

# === DASHBOARD STREAMLIT ===
st.title("📊 Dashboard Prediksi Permintaan Produk Retail")

menu = st.selectbox("Pilih Analisis:", 
                    ["Evaluasi Model", "Uji Asumsi Klasik", "Uji Signifikansi", "Visualisasi Prediksi"])

if menu == "Evaluasi Model":
    st.subheader("📈 Evaluasi Model Regresi")
    st.metric("R-squared (R²)", round(r2_score(y_test, y_pred), 3))
    st.metric("MAE", round(mean_absolute_error(y_test, y_pred), 2))
    st.metric("MSE", round(mean_squared_error(y_test, y_pred), 2))
    st.metric("RMSE", round(mean_squared_error(y_test, y_pred, squared=False), 2))

elif menu == "Uji Asumsi Klasik":
    st.subheader("🧪 Uji Linearitas")
    fig1, ax1 = plt.subplots()
    ax1.scatter(y_pred, residuals)
    ax1.axhline(0, color='red', linestyle='--')
    ax1.set_xlabel("Prediksi")
    ax1.set_ylabel("Residual")
    st.pyplot(fig1)

    st.subheader("🧪 Uji Normalitas Residual (Shapiro-Wilk)")
    stat, p = shapiro(residuals)
    st.write(f"**P-Value:** {p:.4f}")
    st.write("✅ Residual normal" if p > 0.05 else "❌ Tidak normal")

    st.subheader("🧪 Uji Homoskedastisitas (Breusch-Pagan)")
    _, pval, _, _ = het_breuschpagan(residuals, sm.add_constant(X_test))
    st.write(f"**P-Value:** {pval:.4f}")
    st.write("✅ Homoskedastisitas terpenuhi" if pval > 0.05 else "❌ Heteroskedastisitas terdeteksi")

elif menu == "Uji Signifikansi":
    st.subheader("📊 Uji F dan Uji T")
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2).fit()
    st.text(est.summary())

elif menu == "Visualisasi Prediksi":
    st.subheader("📉 Perbandingan Nilai Aktual vs Prediksi")
    fig2, ax2 = plt.subplots()
    ax2.plot(y_test.values, label="Aktual", marker='o')
    ax2.plot(y_pred, label="Prediksi", marker='x')
    ax2.legend()
    st.pyplot(fig2)
