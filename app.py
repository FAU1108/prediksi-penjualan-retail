import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Dataset_Permintaan_Produk_Retail_2024.csv")
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    return df

df = load_data()

st.title("📊 Dashboard Prediksi Permintaan Produk Retail")

# Diagram Batang Penjualan per Bulan
st.header("Penjualan per Bulan")
df['Bulan'] = df['Tanggal'].dt.to_period('M')
bulanan = df.groupby('Bulan')['Penjualan (Unit)'].sum().reset_index()
bulanan['Bulan'] = bulanan['Bulan'].astype(str)

fig, ax = plt.subplots()
sns.barplot(data=bulanan, x='Bulan', y='Penjualan (Unit)', ax=ax)
plt.xticks(rotation=45)
st.pyplot(fig)

# Regresi
st.header("Uji dan Evaluasi Model Regresi Linear")
X = df[['Harga Satuan', 'Stok Tersedia']]
y = df['Penjualan (Unit)']
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()
predictions = model.predict(X_const)
residuals = y - predictions

with st.expander("📌 Uji Normalitas (Shapiro-Wilk)"):
    stat, p = shapiro(residuals)
    st.write(f"p-value: {p:.4f}")
    st.success("Data residual berdistribusi normal" if p > 0.05 else "Residual tidak normal")

with st.expander("📌 Uji Multikolinearitas (VIF)"):
    vif_data = pd.DataFrame()
    vif_data['feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    st.dataframe(vif_data)

with st.expander("📌 Uji Homoskedastisitas (Breusch-Pagan)"):
    lm, lm_pvalue, fval, f_pvalue = het_breuschpagan(residuals, X_const)
    st.write(f"p-value: {f_pvalue:.4f}")
    st.success("Tidak terdapat heteroskedastisitas" if f_pvalue > 0.05 else "Ada indikasi heteroskedastisitas")

with st.expander("📌 Uji Linearitas (Visual)"):
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=predictions, y=residuals, ax=ax2)
    ax2.axhline(0, color='red', linestyle='--')
    ax2.set_xlabel("Prediksi")
    ax2.set_ylabel("Residual")
    ax2.set_title("Scatterplot Prediksi vs Residual")
    st.pyplot(fig2)

with st.expander("📌 Uji Signifikansi (F dan t-Test)"):
    st.write("### Ringkasan Model:")
    st.text(model.summary())
    st.markdown("- **Uji F** menguji signifikansi model secara keseluruhan.")
    st.markdown("- **Uji t** menguji signifikansi masing-masing variabel bebas.")
    st.markdown("- Lihat nilai p-value untuk mengetahui signifikansi (p < 0.05 = signifikan)")

# Evaluasi Model
st.subheader("📈 Evaluasi Kinerja Model")
r2 = model.rsquared
mae = mean_absolute_error(y, predictions)
mse = mean_squared_error(y, predictions)
rmse = np.sqrt(mse)

st.write(f"**R-squared:** {r2:.4f}")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**MSE:** {mse:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")
