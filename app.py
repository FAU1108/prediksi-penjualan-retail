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

# Load dataset
df = pd.read_csv('Dataset_Permintaan_Produk_Retail_2024.csv')

# Pra-pemrosesan (sesuaikan dengan notebookmu)
df['Tanggal'] = pd.to_datetime(df['Tanggal'])
df = pd.get_dummies(df, columns=['Kategori_Produk'], drop_first=True)

# Pemodelan
X = df.drop(['Penjualan', 'Tanggal'], axis=1)
y = df['Penjualan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Streamlit Dashboard
st.title("Dashboard Prediksi Permintaan Retail")

menu = st.selectbox("Pilih Analisis:", 
                    ["Evaluasi Model", "Uji Asumsi Klasik", "Uji Signifikansi", "Visualisasi Prediksi"])

if menu == "Evaluasi Model":
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.metric("R-squared (R²)", round(r2, 3))
    st.metric("Mean Absolute Error (MAE)", round(mae, 2))
    st.metric("Mean Squared Error (MSE)", round(mse, 2))
    st.metric("Root MSE (RMSE)", round(rmse, 2))

elif menu == "Uji Asumsi Klasik":
    residuals = y_test - y_pred

    st.subheader("1. Uji Linearitas")
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Residual")
    st.pyplot(fig)

    st.subheader("2. Uji Normalitas Residual (Shapiro-Wilk)")
    stat, p = shapiro(residuals)
    st.write(f"P-Value: {p:.4f}")
    st.write("Normal" if p > 0.05 else "Tidak Normal")

    st.subheader("3. Uji Homoskedastisitas (Breusch-Pagan)")
    _, pval, _, _ = het_breuschpagan(residuals, sm.add_constant(X_test))
    st.write(f"P-Value: {pval:.4f}")
    st.write("Homoskedastisitas terpenuhi" if pval > 0.05 else "Terdapat heteroskedastisitas")

elif menu == "Uji Signifikansi":
    st.subheader("Uji F dan Uji T")
    X2 = sm.add_constant(X)
    est = sm.OLS(y, X2).fit()
    st.text(est.summary())

elif menu == "Visualisasi Prediksi"):
    st.subheader("Visualisasi Hasil Prediksi vs Aktual")
    fig, ax = plt.subplots()
    ax.plot(y_test.values, label="Aktual", marker='o')
    ax.plot(y_pred, label="Prediksi", marker='x')
    ax.legend()
    st.pyplot(fig)
