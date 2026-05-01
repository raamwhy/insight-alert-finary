# 🚀 FINARY - AI Insight Profile Service

REST API berbasis Machine Learning untuk menganalisis profil keuangan pengguna, memprediksi saldo bulan depan, serta memberikan *warning* dan rekomendasi finansial secara otomatis.

Proyek ini merupakan bagian dari **Finary**, platform manajemen keuangan pribadi untuk **Capstone Project DBS Coding Camp 2026 (AI Track)**.

---

## ✨ Key Features

Sistem menggunakan pendekatan **Hybrid Intelligence (Deep Learning + Rule-Based System)** untuk memastikan hasil yang tidak hanya akurat, tetapi juga actionable.

### 1. 📊 Balance Forecasting
Model regresi berbasis **TensorFlow Functional API** untuk memprediksi saldo bulan berikutnya berdasarkan pola historis transaksi.

### 2. ⚠️ Financial Risk Warning
Model klasifikasi (sigmoid output) untuk mendeteksi probabilitas risiko:
- **Aman**
- **Waspada**
- **Bahaya**

### 3. 💡 Smart Recommendation Engine
Rule-based logic yang menghasilkan rekomendasi finansial kontekstual berbasis hasil prediksi dan profil pengguna.

---

## 🧠 System Architecture

Arsitektur dirancang untuk **low-latency inference** dan modularitas tinggi:

- **ML Layer** → Multi-output deep learning model (regression + classification)
- **Serving Layer** → FastAPI (ASGI-based, high-performance)
- **Logic Layer** → Rule-based recommendation engine
- **Validation Layer** → Pydantic schema validation

---

## 🛠️ Tech Stack

### Machine Learning
- TensorFlow 2.x
- Scikit-Learn
- Pandas
- NumPy

### Backend & API
- FastAPI
- Uvicorn
- Pydantic

---

## ⚡ FastAPI (Serving Layer)

![FastAPI Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/FastAPI_logo.svg/512px-FastAPI_logo.svg.png)

FastAPI digunakan sebagai **ML Inference API** karena:

- **High Performance (ASGI)** → cocok untuk real-time prediction
- **Async-first design** → scalable untuk concurrent request
- **Automatic validation** → via Pydantic (type-safe request/response)
- **Auto-generated docs** → Swagger & OpenAPI built-in

---

## 🔁 Data Flow (High-Level)

1. User mengirim data finansial (income, expenses, behavior)
2. API melakukan validasi schema (Pydantic)
3. Model ML melakukan:
   - Prediksi saldo (regression)
   - Prediksi risiko (classification)
4. Rule engine generate rekomendasi
5. API mengembalikan structured JSON response

---

## 📦 Example Output

```json
{
  "predicted_balance": 1250000,
  "risk_level": "Waspada",
  "recommendation": "Kurangi pengeluaran hiburan sebesar 20% untuk menjaga stabilitas saldo."
}
```

---

## ▶️ Menjalankan API

```bash
uvicorn api_service:app --reload
```

## 🔌 Endpoints (urut konsisten)

### 1) `POST /classify`
Klasifikasi kondisi keuangan bulanan: `survival` / `stable` / `growth`.

- **Kontrak input (ringkas)**:
  - **Wajib (IDR, 5 field)**: `monthly_income`, `monthly_expense_total`, `actual_savings`, `budget_goal`, `emergency_fund`
- **Catatan penting**:
  - Input uang di API selalu **IDR**.
  - Sebelum masuk scaler + model, nilai uang dikonversi ke skala training dengan \( \text{nilai\_training} = \text{IDR} / 2500 \).
  - User tidak perlu mengirim fitur turunan seperti `net_cash_flow`, `expense_ratio`, `savings_rate`, `savings_goal_met`, `spending_efficiency` (backend menghitung otomatis).

**Example request**:

```json
{
  "monthly_income": 9000000,
  "monthly_expense_total": 5500000,
  "actual_savings": 2500000,
  "budget_goal": 2000000,
  "emergency_fund": 15000000
}
```

**Expected response (contoh)**:

```json
{
  "classification": "growth",
  "score": 0.94,
  "probabilities": {
    "survival": 0.01,
    "stable": 0.05,
    "growth": 0.94
  },
  "financial_indicators": {
    "monthly_income": 9000000,
    "monthly_expense_total": 5500000,
    "actual_savings": 2500000,
    "budget_goal": 2000000,
    "emergency_fund": 15000000,
    "net_cash_flow": 3500000,
    "expense_ratio": 0.6111,
    "savings_rate": 0.2778,
    "spending_efficiency": 0.6364
  },
  "risk_flags": {
    "negative_cash_flow": false,
    "high_expense_ratio": false,
    "low_savings_rate": false,
    "savings_goal_not_met": false,
    "low_spending_efficiency": false
  },
  "recommendation_focus": [
    "maintain_growth_momentum",
    "increase_investment_allocation",
    "optimize_long_term_savings"
  ],
  "explanation": "The user is classified as growth because monthly cash flow and savings indicators are strong. Model confidence: 0.94."
}
```

### 2) `POST /predict`
Prediksi saldo bulan depan + warning + rekomendasi (Insight model).

### 3) `POST /recommend-side-hustle`
Rekomendasi side-hustle (7 rekomendasi).

---

## 📓 Notebook Penting

### `finary_classify_model.ipynb`
Notebook untuk training model klasifikasi final (`survival/stable/growth`) dan ekspor artifact:
- `artifacts/classification_model.keras`
- `artifacts/classification_scaler.joblib`
- `artifacts/classification_feature_columns.json`
- `artifacts/classification_label_mapping.json`

Notebook ini juga mendefinisikan **kontrak inference** yang dipakai produksi:
- Input user **IDR minimal** (sebagian fitur dihitung otomatis/engineered)
- Normalisasi uang menggunakan `UNIT_SCALE = 2500`

### `api_service.py` (bagian `/classify`)
Endpoint `/classify` sudah disesuaikan mengikuti metode inference minimal pada notebook:
- Menghitung fitur turunan secara otomatis (mis. `net_cash_flow`, `expense_ratio`, `financial_buffer`, dll)
- Mapping fitur mengikuti schema 27 fitur hasil seleksi (sesuai `classification_feature_columns.json`)
