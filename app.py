import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import jarque_bera
from statsmodels.stats.outliers_influence import variance_inflation_factor
import numpy as np

# --- 1. Konfigurasi Halaman Streamlit ---
st.set_page_config(layout="wide", page_title="Prediksi Permintaan Retail Komprehensif")

st.title("📊 Dashboard Analisis & Prediksi Permintaan Produk Retail")
st.write("Dashboard ini menampilkan tahapan lengkap dari model prediksi permintaan produk retail menggunakan Regresi Linear, dari data hingga evaluasi model.")

# --- 2. Memuat Data ---
# GANTI PATH FILE CSV INI DENGAN PATH DATA ANDA YANG SEBENARNYA
# Pastikan file CSV memiliki kolom: 'Stok Tersedia', 'Kategori Produk', 'Jumlah Penjualan'
# Data dummy jika tidak ada file CSV:
@st.cache_data
def load_data():
    try:
        # Coba muat dari file CSV
        df = pd.read_csv('Dataset_Permintaan_Produk_Retail_2024') # <--- GANTI NAMA FILE ANDA DI SINI
        # Pastikan kolom-kolom yang diharapkan ada
        if 'Stok Tersedia' not in df.columns or 'Kategori Produk' not in df.columns or 'Jumlah Penjualan' not in df.columns:
            st.error("File CSV tidak memiliki kolom yang diperlukan: 'Stok Tersedia', 'Kategori Produk', 'Jumlah Penjualan'.")
            return pd.DataFrame() # Kembalikan DataFrame kosong jika kolom tidak cocok
        return df
    except FileNotFoundError:
        st.warning("File 'your_data.csv' tidak ditemukan. Menggunakan data dummy untuk demonstrasi.")
        data = {
            'Stok Tersedia': np.random.randint(50, 250, 366),
            'Kategori Produk': np.random.choice(['Makanan', 'Minuman', 'Elektronik', 'Pakaian', 'Alat Tulis'], 366),
            'Jumlah Penjualan': np.random.randint(10, 200, 366) + np.random.randint(1, 5, 366) * np.random.randint(5, 10, 366)
        }
        df_dummy = pd.DataFrame(data)
        # Tambahkan sedikit korelasi positif antara Stok Tersedia dan Jumlah Penjualan
        df_dummy['Jumlah Penjualan'] = df_dummy['Jumlah Penjualan'] + df_dummy['Stok Tersedia'] * 0.5 + np.random.normal(0, 10, 366)
        df_dummy['Jumlah Penjualan'] = df_dummy['Jumlah Penjualan'].apply(lambda x: max(0, round(x))).astype(int)
        return df_dummy

df = load_data()

if df.empty:
    st.stop() # Hentikan eksekusi jika data tidak berhasil dimuat

# --- 3. Pra-pemrosesan Data ---
# One-Hot Encoding untuk 'Kategori Produk'
df_encoded = pd.get_dummies(df, columns=['Kategori Produk'], drop_first=True)

# Mendefinisikan Variabel Independen (X) dan Dependen (y)
# Pastikan ini sesuai dengan variabel yang ingin Anda gunakan
X = df_encoded.drop(columns=['Jumlah Penjualan'])
y = df_encoded['Jumlah Penjualan']

# Membagi Data Latih dan Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 4. Pelatihan Model Regresi Linear ---
model = LinearRegression()
model.fit(X_train, y_train)

# Prediksi pada Data Uji
y_pred = model.predict(X_test)

# Menghitung Residual
residuals = y_test - y_pred

# --- 5. Menghitung Metrik Evaluasi Model ---
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# --- 6. Uji Asumsi Klasik (Menggunakan statsmodels) ---
# Tambahkan konstanta ke X_train untuk uji asumsi klasik di statsmodels
X_train_sm = sm.add_constant(X_train)
model_sm = sm.OLS(y_train, X_train_sm).fit()

# a. Uji Normalitas Residual (Jarque-Bera Test)
jb_test = jarque_bera(model_sm.resid)
jb_pvalue = jb_test[1] # p-value

# b. Uji Homoskedastisitas (Breusch-Pagan Test)
bp_test = het_breuschpagan(model_sm.resid, model_sm.model.exog)
bp_pvalue = bp_test[1] # p-value

# c. Uji Multikolinearitas (VIF)
# Pastikan X_train digunakan untuk VIF agar konsisten dengan model yang dilatih
vif_data = pd.DataFrame()
vif_data["Fitur"] = X_train.columns
# Handle case where VIF calculation might fail for constant or singular matrix
try:
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
except np.linalg.LinAlgError:
    st.error("Gagal menghitung VIF. Mungkin ada masalah dengan matriks fitur.")
    vif_data["VIF"] = np.inf # Set to infinity if calculation fails

# d. Uji F (Signifikansi Model Keseluruhan)
f_pvalue = model_sm.f_pvalue

# e. Uji T (Signifikansi Koefisien Individu) - Ambil dari summary
p_values_t_test = model_sm.pvalues.drop('const', errors='ignore') # Exclude constant's p-value

# --- Tampilan Dashboard ---

# Tabulasi Tahapan Analisis
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Overview & Prediksi", "Evaluasi Model", "Uji Asumsi Klasik", "Koefisien Model", "Visualisasi"])

with tab1:
    st.header("1. Gambaran Umum Data & Antarmuka Prediksi")
    st.subheader("Data Overview")
    st.write("Cuplikan 5 baris pertama dari dataset Anda:")
    st.dataframe(df.head())
    st.write(f"Dimensi Dataset: {df.shape[0]} baris, {df.shape[1]} kolom")

    st.subheader("Pra-pemrosesan Data")
    st.write("Variabel kategorikal 'Kategori Produk' diubah menjadi format numerik menggunakan **One-Hot Encoding** agar dapat diproses oleh model regresi linear. Variabel yang digunakan untuk pemodelan adalah 'Stok Tersedia' dan kolom-kolom hasil one-hot encoding dari 'Kategori Produk'.")
    st.write(f"Jumlah Fitur Setelah One-Hot Encoding: {X.shape[1]}")

    st.subheader("Antarmuka Prediksi Permintaan")
    st.write("Gunakan slider dan dropdown di sidebar untuk mendapatkan prediksi jumlah penjualan.")

    st.sidebar.header("⚙️ Parameter Prediksi")
    stok_input = st.sidebar.slider("Stok Tersedia", min_value=int(df['Stok Tersedia'].min()), max_value=int(df['Stok Tersedia'].max()), value=int(df['Stok Tersedia'].mean()))
    kategori_input = st.sidebar.selectbox("Kategori Produk", df['Kategori Produk'].unique())

    # Pra-pemrosesan input pengguna untuk prediksi
    input_data_for_pred = pd.DataFrame([[stok_input, kategori_input]], columns=['Stok Tersedia', 'Kategori Produk'])
    input_encoded_for_pred = pd.get_dummies(input_data_for_pred, columns=['Kategori Produk'])

    # Pastikan kolom input sesuai dengan kolom yang digunakan saat pelatihan model
    # Buat DataFrame kosong dengan semua kolom X.columns, lalu isi nilai input
    final_input_features = pd.DataFrame(0, index=[0], columns=X.columns)
    final_input_features['Stok Tersedia'] = stok_input
    for col in input_encoded_for_pred.columns:
        if col in final_input_features.columns: # Hanya tambahkan kolom yang ada di model
            final_input_features[col] = input_encoded_for_pred[col].iloc[0]

    # Prediksi
    try:
        predicted_demand = model.predict(final_input_features)[0]
        st.sidebar.metric("Prediksi Jumlah Penjualan", f"{predicted_demand:.0f} unit")
    except Exception as e:
        st.sidebar.error(f"Error saat memprediksi: {e}. Pastikan semua fitur sesuai.")


with tab2:
    st.header("2. Hasil Evaluasi Model Regresi Linear")
    st.markdown("Berikut adalah metrik yang digunakan untuk menilai performa dan akurasi model dalam memprediksi permintaan produk.")

    col_m1, col_m2, col_m3, col_m4 = st.columns(4)

    with col_m1:
        st.metric("R-squared (R²)", f"{r2:.3f}")
        st.write("Proporsi variasi penjualan yang dapat dijelaskan oleh model. Mendekati 1 lebih baik.")

    with col_m2:
        st.metric("Mean Squared Error (MSE)", f"{mse:.2f}")
        st.write("Rata-rata kuadrat kesalahan prediksi. Nilai lebih rendah lebih baik.")

    with col_m3:
        st.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")
        st.write("Akar kuadrat dari MSE. Skala sama dengan data asli. Nilai lebih rendah lebih baik.")

    with col_m4:
        st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
        st.write("Rata-rata selisih absolut antara prediksi dan aktual. Lebih tahan terhadap outlier.")


with tab3:
    st.header("3. Uji Asumsi Klasik & Signifikansi Statistik")
    st.markdown("Uji ini dilakukan untuk memastikan bahwa model regresi linear Anda valid dan memenuhi asumsi-asumsi dasar.")

    col_uji_1, col_uji_2 = st.columns(2)

    with col_uji_1:
        st.subheader("3.1 Uji Normalitas Residual (Jarque-Bera)")
        st.write(f"**P-value:** `{jb_pvalue:.4f}`")
        if jb_pvalue > 0.05:
            st.success("✅ **Lolos!** Residual terdistribusi normal.")
            st.write("Ini menunjukkan bahwa kesalahan prediksi model terdistribusi secara seimbang dan acak, yang ideal untuk inferensi statistik.")
        else:
            st.error("❌ **Tidak Lolos.** Residual tidak terdistribusi normal.")
            st.write("Ini bisa memengaruhi validitas uji T dan F.")

        st.subheader("3.2 Uji Multikolinearitas (VIF)")
        st.write("Mengukur seberapa besar varians estimasi koefisien diperbesar karena korelasi antar variabel independen. VIF < 5 umumnya baik.")
        st.dataframe(vif_data)
        if (vif_data['VIF'] > 5).any():
            st.error("❌ **Tidak Lolos.** Terdapat multikolinearitas signifikan.")
            st.write("Ini berarti beberapa variabel independen sangat berkorelasi, yang dapat menyulitkan interpretasi koefisien.")
        else:
            st.success("✅ **Lolos!** Tidak ada multikolinearitas signifikan.")
            st.write("Variabel independen tidak terlalu berkorelasi satu sama lain, sehingga pengaruh masing-masing dapat diukur dengan jelas.")

    with col_uji_2:
        st.subheader("3.3 Uji Homoskedastisitas (Breusch-Pagan)")
        st.write(f"**P-value:** `{bp_pvalue:.4f}`")
        if bp_pvalue > 0.05:
            st.success("✅ **Lolos!** Varians residual konstan.")
            st.write("Artinya, sebaran kesalahan prediksi konsisten di seluruh rentang nilai prediksi, menunjukkan model stabil.")
        else:
            st.error("❌ **Tidak Lolos.** Terdapat heteroskedastisitas.")
            st.write("Varians residual tidak konstan, yang bisa membuat estimasi koefisien menjadi kurang efisien dan inferensi tidak valid.")

        st.subheader("3.4 Uji Signifikansi Model Keseluruhan (Uji F)")
        st.write(f"**P-value:** `{f_pvalue:.4f}`")
        if f_pvalue < 0.05:
            st.success("✅ **Lolos!** Model secara keseluruhan signifikan.")
            st.write("Ini berarti setidaknya ada satu variabel independen secara signifikan memengaruhi variabel dependen (jumlah penjualan).")
        else:
            st.error("❌ **Tidak Lolos.** Model secara keseluruhan tidak signifikan.")

with tab4:
    st.header("4. Koefisien Model Regresi Linear")
    st.write("Koefisien ini menunjukkan seberapa besar perubahan pada variabel dependen (Jumlah Penjualan) untuk setiap satu unit perubahan pada variabel independen, dengan variabel lain dianggap konstan.")

    # Membuat DataFrame untuk koefisien dan p-value uji T
    coefs_df = pd.DataFrame({
        "Fitur": X.columns,
        "Koefisien": model.coef_,
        "P-value (Uji T)": p_values_t_test # Menggunakan p-values dari statsmodels
    })

    # Tambahkan kolom untuk status signifikansi
    coefs_df['Signifikan (p < 0.05)'] = coefs_df['P-value (Uji T)'] < 0.05

    st.dataframe(coefs_df)
    st.write("Nilai p-value di bawah 0.05 (atau tingkat signifikansi yang Anda tetapkan) mengindikasikan bahwa fitur tersebut berpengaruh signifikan.")
    st.write("Berdasarkan informasi Anda, 'Stok Tersedia' adalah fitur yang signifikan.")

with tab5:
    st.header("5. Visualisasi Hasil Model")

    col_vis1, col_vis2 = st.columns(2)

    with col_vis1:
        st.subheader("5.1 Distribusi Residual Model")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
        sns.histplot(residuals, kde=True, ax=ax_hist, color='skyblue')
        ax_hist.set_title('Histogram Distribusi Residual')
        ax_hist.set_xlabel('Nilai Residual')
        ax_hist.set_ylabel('Frekuensi')
        st.pyplot(fig_hist)
        st.write("Histogram yang menyerupai bentuk lonceng menunjukkan bahwa residual model terdistribusi secara normal.")

        st.subheader("5.2 Prediksi Aktual vs. Prediksi Model")
        fig_actual_pred, ax_actual_pred = plt.subplots(figsize=(8, 5))
        ax_actual_pred.scatter(y_test, y_pred, alpha=0.6, color='green')
        ax_actual_pred.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax_actual_pred.set_xlabel('Penjualan Aktual')
        ax_actual_pred.set_ylabel('Penjualan Prediksi')
        ax_actual_pred.set_title('Perbandingan Penjualan Aktual vs. Prediksi')
        st.pyplot(fig_actual_pred)
        st.write("Titik-titik yang mendekati garis merah (garis ideal) menunjukkan akurasi prediksi model.")


    with col_vis2:
        st.subheader("5.3 Residual vs. Nilai Prediksi")
        fig_scatter, ax_scatter = plt.subplots(figsize=(8, 5))
        ax_scatter.scatter(y_pred, residuals, alpha=0.6, color='coral')
        ax_scatter.axhline(y=0, color='blue', linestyle='--')
        ax_scatter.set_title('Residual vs. Nilai Prediksi')
        ax_scatter.set_xlabel('Nilai Prediksi')
        ax_scatter.set_ylabel('Residual')
        st.pyplot(fig_scatter)
        st.write("Sebaran titik residual yang acak di sekitar garis nol tanpa pola tertentu menunjukkan homoskedastisitas (varians konstan).")


st.markdown("---")
st.caption("Dashboard ini dikembangkan menggunakan Streamlit.")
