# 🚀 FINARY AI MODEL
API berbasis Deep Learning yang dirancang untuk menganalisis profil keuangan pengguna secara cerdas.

**📖 Final Release:** [https://huggingface.co/spaces/raamwhy/finary-model/tree/main](https://huggingface.co/spaces/raamwhy/finary-model/tree/main)

**📍 Dashboard URL:** [https://raamwhy-finary-model.hf.space](https://raamwhy-finary-model.hf.space)  

**📖 API Documentation:** [https://raamwhy-finary-model.hf.space/docs](https://raamwhy-finary-model.hf.space/docs)

## 🧠 Model & Endpoints

## 1️⃣ Financial Condition Classification (POST /classify)

Memprediksi kondisi kesehatan finansial bulanan pengguna berdasarkan kategori.

Jenis ML: Supervised Learning - Multiclass Classification.

Model: Deep Neural Network (MLP) dengan Residual Connection untuk stabilitas gradien yang lebih baik.

Input: 5 parameter (Monthly Income, Monthly Expense, Actual Savings, Budget Goal, Emergency Fund) dalam nominal IDR.

Output: Klasifikasi ke dalam 3 kategori utama: 0 (Survival), 1 (Stable), atau 2 (Growth).

## 2️⃣ Insight Profile Model (POST /predict)
Memberikan analisis insight profil user dan rekomendasi.

Jenis ML: Supervised Learning - Multi-task Learning (Gabungan Regresi & Klasifikasi).

Model: Deep Neural Network dengan dua "kepala" output yang berjalan bersamaan.

Input: 6 parameter (Monthly Income, Monthly Expense, Actual Savings, Budget Goal, Loan Payment, Emergency Fund) dalam nominal IDR.

Output 1 (Regression): Prediksi nominal saldo bulan depan (Forecasting).

Output 2 (Classification): Deteksi probabilitas risiko finansial (Financial Warning) untuk mendeteksi bahaya keuangan.

Output 3: Rekomendasi tindakan finansial otomatis yang dihasilkan secara kontekstual dari hasil prediksi model.

## 3️⃣ Side-Hustle Recommendation (POST /recommend-side-hustle)
Sistem pemberi rekomendasi pekerjaan sampingan yang dioptimasi.

Jenis ML: Supervised Learning - Multi-task Learning.

Model: Deep Neural Network dengan Custom Dense Block untuk ekstraksi fitur yang lebih presisi.

Input: 3 parameter (Experience Level, Available hours per week, Interest Category).

Output 1 (Regression): Prediksi estimasi potensi pendapatan (Earnings Estimation) dalam nominal IDR.

Output 2 (Classification): Prediksi 7 rekomendasi terbaik melalui kombinasi platform dan tipe proyek, yang diurutkan berdasarkan probabilitas keberhasilan (Success Probability) tertinggi sesuai level pengalaman dan minat pengguna.
