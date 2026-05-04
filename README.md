# 🚀 FINARY - AI Insight Profile Service (Deployed)

**FINARY AI** adalah REST API berbasis Deep Learning yang dirancang untuk menganalisis profil keuangan pengguna secara cerdas. Layanan ini merupakan bagian inti dari **Capstone Project DBS Coding Camp 2026 (AI Track)**.

**📍 Production URL:** [https://raamwhy-finary-model.hf.space](https://raamwhy-finary-model.hf.space)  
**📖 API Documentation:** [https://raamwhy-finary-model.hf.space/docs](https://raamwhy-finary-model.hf.space/docs)

## 🧠 Model & Endpoints

Layanan ini mengintegrasikan tiga model Deep Learning utama yang bekerja secara independen:

### 1️⃣ Financial Condition Classification (`POST /classify`)
Memprediksi kondisi keuangan bulanan pengguna ke dalam 3 kategori: `survival`, `stable`, atau `growth` dengan input 5 parameter utama (IDR) seperti Income, Expense, Savings, Budget Goal, dan Emergency Fund.

### 2️⃣ Multi-Task Insight Model (`POST /predict`)
Memberikan analisis mendalam menggunakan model multi-output:
- **Output 1 (Regression):** Prediksi saldo bulan depan (*Balance Forecasting*).
- **Output 2 (Classification):** Deteksi probabilitas risiko finansial (*Financial Warning*).
- **Output 3:** Rekomendasi tindakan finansial yang dipersonalisasi.

### 3️⃣ Side-Hustle Recommendation (`POST /recommend-side-hustle`)
Memberikan 7 saran pekerjaan sampingan terbaik berdasarkan level pengalaman dan kategori minat pengguna, lengkap dengan estimasi pendapatan bulanan dalam IDR.
