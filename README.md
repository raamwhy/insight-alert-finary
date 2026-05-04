# 🚀 FINARY - AI Insight Profile Service (Deployed)

**FINARY AI** adalah REST API berbasis Deep Learning yang dirancang untuk menganalisis profil keuangan pengguna secara cerdas. Layanan ini merupakan bagian inti dari **Capstone Project DBS Coding Camp 2026 (AI Track)**.

**📍 Production URL:** [https://raamwhy-finary-model.hf.space](https://raamwhy-finary-model.hf.space)  
**📖 API Documentation:** [https://raamwhy-finary-model.hf.space/docs](https://raamwhy-finary-model.hf.space/docs)

## 🛠️ Tech Stack & Deployment
- **Framework:** FastAPI (Python 3.10+)
- **ML Engine:** TensorFlow 2.x (Functional API)
- **Infrastructure:** Hugging Face Spaces (Gradio/Python SDK)
- **Data Processing:** Scikit-Learn (RobustScaler), Pandas, Joblib

## 🧠 Model & Endpoints

Layanan ini mengintegrasikan tiga model Deep Learning utama yang bekerja secara independen:

### 1️⃣ Financial Condition Classification (`POST /classify`)
Memprediksi kondisi keuangan bulanan pengguna ke dalam 3 kategori: `survival`, `stable`, atau `growth`.
- **Input:** 5 parameter utama (IDR) seperti Income, Expense, Savings, Budget Goal, dan Emergency Fund.
- **Backend Logic:** Menghitung secara otomatis fitur turunan seperti *Expense Ratio* dan *Savings Rate*.
- **Scaling:** Menggunakan `UNIT_SCALE = 2500` untuk sinkronisasi data IDR ke model.

### 2️⃣ Multi-Task Insight Model (`POST /predict`)
Memberikan analisis mendalam menggunakan model multi-output:
- **Output 1 (Regression):** Prediksi saldo bulan depan (*Balance Forecasting*).
- **Output 2 (Classification):** Deteksi probabilitas risiko finansial (*Financial Warning*).
- **Output 3:** Rekomendasi tindakan finansial yang dipersonalisasi.

### 3️⃣ Side-Hustle Recommendation (`POST /recommend-side-hustle`)
Memberikan 7 saran pekerjaan sampingan terbaik berdasarkan level pengalaman dan kategori minat pengguna, lengkap dengan estimasi pendapatan bulanan dalam IDR.

## 🔁 Alur Data (Inference Workflow)

1. **Request:** Client mengirimkan data finansial mentah dalam mata uang **IDR**.
2. **Preprocessing:** Backend mengonversi nilai IDR ke skala training (IDR / 2500) dan menghitung fitur engineered secara deterministik.
3. **AI Inference:** Model TensorFlow melakukan prediksi (Klasifikasi/Regresi).
4. **Post-processing:** Hasil normalisasi model dikonversi kembali ke nilai IDR yang mudah dipahami manusia.
5. **Response:** API mengembalikan struktur JSON yang siap dikonsumsi oleh aplikasi mobile/frontend.

## 📦 Contoh Penggunaan (CURL)
```bash
curl -X 'POST' \
  '[https://raamwhy-finary-model.hf.space/classify](https://raamwhy-finary-model.hf.space/classify)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "monthly_income": 10000000,
  "monthly_expense_total": 6000000,
  "actual_savings": 2000000,
  "budget_goal": 1500000,
  "emergency_fund": 5000000
}'
