with st.expander("🧪 Simulasi Prediksi Manual"):
    st.markdown("Masukkan nilai-nilai untuk memprediksi **Unit Terjual**:")

    input_data = {}

    # Input numerik
    for col in X.columns:
        if df[col].dtype in [np.float64, np.int64]:
            val = st.number_input(f"{col}", min_value=0.0, value=float(df[col].mean()))
        else:
            val = st.selectbox(f"{col}", options=[0, 1], index=0)
        input_data[col] = val

    # Buat dataframe input
    input_df = pd.DataFrame([input_data])

    # Prediksi
    if st.button("Prediksi"):
        hasil_prediksi = model.predict(input_df)[0]
        st.success(f"🎯 Prediksi Unit Terjual: **{hasil_prediksi:.2f} unit**")
