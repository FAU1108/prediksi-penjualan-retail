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
# PASTIKAN FILE 'Dataset_Permintaan_Produk_Retail_2024.csv' ADA DI FOLDER YANG SAMA DENGAN app_full_stages.py
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Dataset_Permintaan_Produk_Retail_2024.csv')
        
        # Pastikan kolom-kolom yang diharapkan ada
        required_cols = ['Stok Tersedia', 'Kategori Produk', 'Jumlah Penjualan']
        if not all(col in df.columns for col in required_cols):
            st.error(f"File CSV tidak memiliki kolom yang diperlukan: {', '.join(required_cols)}. Harap periksa nama kolom.")
            return pd.DataFrame()

        # --- Pastikan kolom numerik benar-benar numerik dan tangani missing values ---
        # 'errors='coerce'' akan mengubah nilai non-numerik menjadi NaN (Not a Number)
        # Ini penting untuk membersihkan data dari entri yang tidak valid
        df['Stok Tersedia'] = pd.to_numeric(df['Stok Tersedia'], errors='coerce')
        df['Jumlah Penjualan'] = pd.to_numeric(df['Jumlah Penjualan'], errors='coerce')

        # Penanganan Missing Values setelah konversi (opsi: isi dengan median atau rata-rata)
        # Mengisi NaN dengan median adalah pilihan yang baik agar tidak terpengaruh outlier
        df['Stok Tersedia'].fillna(df['Stok Tersedia'].median(), inplace=True)
        df['Jumlah Penjualan'].fillna(df['Jumlah Penjualan'].median(), inplace=True)
        
        # Jika ada baris yang masih memiliki NaN setelah penanganan di atas (misalnya dari kategori produk)
        # atau jika Anda ingin lebih ketat, Anda bisa menghapus baris dengan NaN:
        # df.dropna(inplace=True) 

        return df
    except FileNotFoundError:
        st.warning("File 'Dataset_Permintaan_Produk_Retail_2024.csv' tidak ditemukan. Menggunakan data dummy untuk demonstrasi.")
        # Data dummy untuk demonstrasi jika file tidak ditemukan
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
    st.error("Gagal memuat data. Mohon periksa file CSV dan pastikan kolom-kolomnya sesuai.")
    st.stop() # Hentikan eksekusi jika data tidak berhasil dimuat

# --- 3. Pra-pemrosesan Data ---
# One-Hot Encoding untuk 'Kategori Produk'
df_encoded = pd.get_dummies(df, columns=['Kategori Produk'], drop_first=True, dtype=int) # dtype=int memastikan output 0/1

# Mendefinisikan Variabel Independen (X) dan Dependen (y)
X = df_encoded.drop(columns=['Jumlah Penjualan'])
y = df_encoded['Jumlah Penjualan']

# Membagi Data Latih dan Uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Tambahan untuk Debugging Tipe Data (akan muncul di dashboard) ---
st.sidebar.subheader("Debugging Tipe Data")
st.sidebar.write("Tipe data X_train:")
st.sidebar.write(X_train.dtypes)
st.sidebar.write("Tipe data y_train:")
st.sidebar.write(y_train.dtypes)
st.sidebar.markdown("---")
# --- Akhir Tambahan Debugging ---

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
# Pastikan semua kolom X_train adalah numerik sebelum menambahkan konstanta
X_train_sm = sm.add_constant(X_train.astype(float), has_constant='add') # Memastikan X_train adalah float

# Handle cases where y_train might still have non-numeric (if error handling in load_data isn't perfect)
y_train_sm = y_train.astype(float)


# Pastikan tidak ada NaN atau inf setelah konversi sebelum fitting model_sm
if X_train_sm.isnull().values.any() or np.isinf(X_train_sm).values.any() or y_train_sm.isnull().values.any() or np.isinf(y_train_sm).values.any():
    st.error("Terdapat nilai NaN atau tak hingga di data setelah pra-pemrosesan untuk model statsmodels. Harap periksa data Anda.")
    st.stop()


try:
    model_sm = sm.OLS(y_train_sm, X_train_sm).fit()
except Exception as e:
    st.error(f"Gagal melatih model statsmodels: {e}. Kemungkinan ada masalah dengan data atau variabel.")
    st.stop()


# a. Uji Normalitas Residual (Jarque-Bera Test)
jb_test = jarque_bera(model_sm.resid)
jb_pvalue = jb_test[1] # p-value

# b. Uji Homoskedastisitas (Breusch-Pagan Test)
# Pastikan model.model.exog digunakan yang merupakan array numpy
bp_test = het_breuschpagan(model_sm.resid, model_sm.model.exog)
bp_pvalue = bp_test[1] # p-value

# c. Uji Multikolinearitas (VIF)
vif_data = pd.DataFrame()
vif_data["Fitur"] = X_train.columns
# Handle case where VIF calculation might fail for constant or singular matrix
try:
    vif_data["VIF"] = [variance_inflation_factor(X_train.values, i) for i in range(len(X_train.columns))]
except np.linalg.LinAlgError:
    st.error("Gagal menghitung VIF. Ini bisa terjadi jika ada fitur dengan varians nol atau korelasi sempurna.")
    vif_data["VIF"] = np.inf # Set to infinity if calculation fails
except Exception as e:
    st.error(f"Kesalahan saat menghitung VIF: {e}")
    vif_data["VIF"] = np.nan # Set to NaN if other error

# d. Uji F (Signifikansi Model Keseluruhan)
f_pvalue = model_sm.f_pvalue

# e. Uji T (Signifikansi Koefisien Individu) - Ambil dari summary
# drop('const') karena konstanta tidak dianggap sebagai fitur independen
p_values_t_test = model_sm.pvalues.drop('const', errors='ignore') 

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
    
    # Pastikan min/max/value untuk slider berasal dari data yang dimuat
    if not df.empty:
        min_stok = int(df['Stok Tersedia'].min())
        max_stok = int(df['Stok Tersedia'].max())
        mean_stok = int(df['Stok Tersedia'].mean())
        unique_kategori = df['Kategori Produk'].unique().tolist()
    else: # Fallback for dummy data or error in loading
        min_stok = 50
        max_stok = 250
        mean_stok = 150
        unique_kategori = ['Makanan', 'Minuman', 'Elektronik', 'Pakaian', 'Alat Tulis']

    stok_input = st.sidebar.slider("Stok Tersedia", min_value=min_stok, max_value=max_stok, value=mean_stok)
    kategori_input = st.sidebar.selectbox("Kategori Produk", unique_kategori)

    # Pra-pemrosesan input pengguna untuk prediksi
    input_data_for_pred = pd.DataFrame([[stok_input, kategori_input]], columns=['Stok Tersedia', 'Kategori Produk'])
    input_encoded_for_pred = pd.get_dummies(input_data_for_pred, columns=['Kategori Produk'], dtype=int)

    # Pastikan kolom input sesuai dengan kolom yang digunakan saat pelatihan model
    final_input_features = pd.DataFrame(0, index=[0], columns=X.columns)
    final_input_features['Stok Tersedia'] = stok_input
    
    # Mengisi kolom one-hot encoded dari input pengguna
    for col in input_encoded_for_pred.columns:
        if col in final_input_features.columns: 
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
            st.write("Ini bisa memengaruhi validitas uji T dan F. Perlu penyelidikan lebih lanjut pada distribusi residual.")

        st.subheader("3.2 Uji Multikolinearitas (VIF)")
        st.write("Mengukur seberapa besar varians estimasi koefisien diperbesar karena korelasi antar variabel independen. VIF < 5 umumnya baik.")
        if not vif_data.empty and not vif_data['VIF'].isnull().all():
            st.dataframe(vif_data)
            if (vif_data['VIF'] > 5).any():
                st.error("❌ **Tidak Lolos.** Terdapat multikolinearitas signifikan.")
                st.write("Ini berarti beberapa variabel independen sangat berkorelasi, yang dapat menyulitkan interpretasi koefisien. Pertimbangkan untuk menghapus atau menggabungkan fitur yang sangat berkorelasi.")
            else:
                st.success("✅ **Lolos!** Tidak ada multikolinearitas signifikan.")
                st.write("Variabel independen tidak terlalu berkorelasi satu sama lain, sehingga pengaruh masing-masing dapat diukur dengan jelas.")
        else:
            st.warning("Tidak dapat menampilkan VIF. Perhitungan gagal atau data kosong.")

    with col_uji_2:
        st.subheader("3.3 Uji Homoskedastisitas (Breusch-Pagan)")
        st.write(f"**P-value:** `{bp_pvalue:.4f}`")
        if bp_pvalue > 0.05:
            st.success("✅ **Lolos!** Varians residual konstan.")
            st.write("Artinya, sebaran kesalahan prediksi konsisten di seluruh rentang nilai prediksi, menunjukkan model stabil.")
        else:
            st.error("❌ **Tidak Lolos.** Terdapat heteroskedastisitas.")
            st.write("Varians residual tidak konstan. Ini bisa memengaruhi efisiensi estimasi koefisien dan validitas inferensi. Metode regresi lain mungkin diperlukan atau transformasi data.")

        st.subheader("3.4 Uji Signifikansi Model Keseluruhan (Uji F)")
        st.write(f"**P-value:** `{f_pvalue:.4f}`")
        if f_pvalue < 0.05:
            st.success("✅ **Lolos!** Model secara keseluruhan signifikan.")
            st.write("Ini berarti setidaknya ada satu variabel independen secara signifikan memengaruhi variabel dependen (jumlah penjualan), dan model Anda berguna.")
        else:
            st.error("❌ **Tidak Lolos.** Model secara keseluruhan tidak signifikan.")
            st.write("Model Anda mungkin tidak efektif dalam menjelaskan variasi variabel dependen.")

with tab4:
    st.header("4. Koefisien Model Regresi Linear")
    st.write("Koefisien ini menunjukkan seberapa besar perubahan pada variabel dependen (Jumlah Penjualan) untuk setiap satu unit perubahan pada variabel independen, dengan variabel lain dianggap konstan.")

    # Membuat DataFrame untuk koefisien dan p-value uji T
    coefs_df = pd.DataFrame({
        "Fitur": X.columns,
        "Koefisien": model.coef_,
        "P-value (Uji T)": p_values_t_test.reindex(X.columns).values # Memastikan urutan p-value sesuai fitur
    })

    # Tambahkan kolom untuk status signifikansi
    coefs_df['Signifikan (p < 0.05)'] = coefs_df['P-value (Uji T)'] < 0.05

    st.dataframe(coefs_df)
    st.write("Nilai p-value di bawah 0.05 (atau tingkat signifikansi yang Anda tetapkan, misalnya 0.01 atau 0.1) mengindikasikan bahwa fitur tersebut berpengaruh signifikan.")
    st.write("Seperti yang Anda sebutkan, 'Stok Tersedia' adalah fitur yang diharapkan signifikan.")

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
        st.write("Histogram yang menyerupai bentuk lonceng menunjukkan bahwa residual model terdistribusi secara normal. Penting untuk validitas inferensi.")

        st.subheader("5.2 Prediksi Aktual vs. Prediksi Model")
        fig_actual_pred, ax_actual_pred = plt.subplots(figsize=(8, 5))
        ax_actual_pred.scatter(y_test, y_pred, alpha=0.6, color='green')
        # Garis ideal di mana prediksi = aktual
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
        st.write("Sebaran titik residual yang acak di sekitar garis nol tanpa pola tertentu menunjukkan homoskedastisitas (varians konstan). Jika ada pola (misalnya bentuk corong), berarti ada heteroskedastisitas.")


st.markdown("---")
st.caption("Dashboard ini dikembangkan menggunakan Streamlit.")
