from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =========================================
# 1. SETUP PATHS & KONSTANTA
# =========================================
ARTIFACT_DIR = Path("artifacts")
UNIT_SCALE = 2500.0  # Konversi IDR (Untuk Model Insight)
USD_TO_IDR = 17000.0 # Konversi Prediksi Earnings USD ke IDR (Untuk Side Hustle)

# =========================================
# 2. CLASS CUSTOM LAYER (WAJIB ADA DI SINI)
# =========================================
@tf.keras.utils.register_keras_serializable()
class CustomDenseBlock(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.dense(inputs)
        x = self.bn(x)
        return self.relu(x)

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config

# =========================================
# 3. SCHEMA PYDANTIC (PAYLOAD FRONTEND)
# =========================================
class PredictRequest(BaseModel):
    income: float = Field(..., description="Pendapatan bulanan (IDR)")
    expense: float = Field(..., description="Total pengeluaran (IDR)")
    savings: float = Field(..., description="Tabungan saat ini (IDR)")
    target_tabungan: float = Field(..., description="Target tabungan (IDR)")
    loan_payment: float = Field(..., description="Total cicilan utang (IDR)")
    emergency_fund: float = Field(..., description="Dana darurat (IDR)")
    income_type: str = Field("Salary", description="Salary/Mixed")
    main_category: str = Field("Utilities", description="Kategori pengeluaran")

class PredictResponse(BaseModel):
    predicted_next_month_balance: float
    warning_probability: float
    warning_flag: int
    recommendations: List[str]

class SideHustleRequest(BaseModel):
    experience_level: str = Field(..., description="Beginner, Intermediate, Expert")
    available_hours_per_week: int = Field(..., description="Waktu luang per minggu")
    interest_category: str = Field(..., description="Bidang: App Development, SEO, dll")

class SideHustleRecommendation(BaseModel):
    job_category: str
    platform: str
    project_type: str
    predicted_monthly_earnings_idr: float

class SideHustleResponse(BaseModel):
    recommendations: List[SideHustleRecommendation]

# =========================================
# 4. LOAD ARTIFACTS (KEDUA MODEL)
# =========================================
# --- Insight Model ---
INS_MODEL = tf.keras.models.load_model(ARTIFACT_DIR / "finary_multitask_model.keras")
INS_SCALER = joblib.load(ARTIFACT_DIR / "scaler.joblib")
with open(ARTIFACT_DIR / "feature_columns.json", "r") as f: INS_FEAT_COLS = json.load(f)
with open(ARTIFACT_DIR / "target_stats.json", "r") as f: ins_stats = json.load(f)
INS_BAL_MIN, INS_BAL_MAX = float(ins_stats["balance_min"]), float(ins_stats["balance_max"])

# --- Side Hustle Model ---
SH_MODEL = tf.keras.models.load_model(
    ARTIFACT_DIR / "sh_model.keras",
    custom_objects={"CustomDenseBlock": CustomDenseBlock}
)
SH_SCALER = joblib.load(ARTIFACT_DIR / "sh_scaler.joblib")
with open(ARTIFACT_DIR / "sh_feature_columns.json", "r") as f: SH_FEAT_COLS = json.load(f)
with open(ARTIFACT_DIR / "sh_target_stats.json", "r") as f: sh_stats = json.load(f)
SH_EARN_MIN, SH_EARN_MAX = float(sh_stats["earn_min"]), float(sh_stats["earn_max"])

PLATFORMS = sh_stats["platforms"]
PROJECT_TYPES = sh_stats["project_types"]

# =========================================
# 5. APLIKASI FASTAPI
# =========================================
app = FastAPI(title="FINARY AI Microservices", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "message": "Both Insight and Side Hustle models loaded."}

# -----------------------------------------
# ENDPOINT 1: INSIGHT & WARNING
# -----------------------------------------
def build_insight_recs(features_dict: Dict[str, float], warning_prob: float) -> list[str]:
    recs = []
    if features_dict.get("debt_ratio_flag", 0) == 1.0: recs.append("Prioritaskan pelunasan utang berbunga tinggi.")
    if features_dict.get("low_emergency_flag", 0) == 1.0: recs.append("Tingkatkan emergency fund.")
    if warning_prob > 0.7: recs.append("Warning tinggi: batasi transaksi non-esensial selama 2 minggu.")
    if not recs: recs.append("Profil keuangan sehat.")
    return recs

@app.post("/predict", response_model=PredictResponse)
def predict_insight(payload: PredictRequest):
    try:
        inc = payload.income / UNIT_SCALE
        exp = payload.expense / UNIT_SCALE
        sav = payload.savings / UNIT_SCALE
        tgt_sav = payload.target_tabungan / UNIT_SCALE
        loan = payload.loan_payment / UNIT_SCALE
        emg = payload.emergency_fund / UNIT_SCALE
        
        net_cf = inc - exp
        dti = loan / inc if inc > 0 else 0.0
        buffer = emg / exp if exp > 0 else 0.0

        features = {col: 0.0 for col in INS_FEAT_COLS}
        features.update({
            "monthly_income": inc, "monthly_expense_total": exp, "actual_savings": sav,
            "budget_goal": tgt_sav, "loan_payment": loan, "emergency_fund": emg,
            "net_cash_flow": net_cf, "savings_rate": sav / inc if inc > 0 else 0.0,
            "expense_ratio": exp / inc if inc > 0 else 0.0, "debt_to_income_ratio": dti,
            "financial_buffer": buffer, "savings_goal_met": 1.0 if sav >= tgt_sav else 0.0,
            "debt_ratio_flag": 1.0 if dti >= 0.35 else 0.0, "low_emergency_flag": 1.0 if buffer < 1.0 else 0.0
        })
        
        if f"income_type_{payload.income_type}" in features: features[f"income_type_{payload.income_type}"] = 1.0
        if f"category_{payload.main_category}" in features: features[f"category_{payload.main_category}"] = 1.0
        features["cash_flow_status_Positive"] = 1.0 if net_cf > 0 else 0.0
        features["cash_flow_status_Neutral"] = 1.0 if net_cf <= 0 else 0.0

        row_scaled = INS_SCALER.transform(pd.DataFrame([features])[INS_FEAT_COLS].values)
        pred_balance_norm, pred_warning_prob = INS_MODEL.predict(row_scaled, verbose=0)
        
        pred_bal_val = float(np.clip(pred_balance_norm[0][0], 0.0, None))
        pred_warn_val = float(np.clip(pred_warning_prob[0][0], 0.0, 1.0))

        predicted_balance_idr = (pred_bal_val * (INS_BAL_MAX - INS_BAL_MIN) + INS_BAL_MIN) * UNIT_SCALE

        return PredictResponse(
            predicted_next_month_balance=round(predicted_balance_idr, 2),
            warning_probability=round(pred_warn_val, 4),
            warning_flag=int(pred_warn_val >= 0.5),
            recommendations=build_insight_recs(features, pred_warn_val)
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

# -----------------------------------------
# ENDPOINT 2: SIDE HUSTLE RECOMMENDATION (7 Rekomendasi & Variasi Platform)
# -----------------------------------------
@app.post("/recommend-side-hustle", response_model=SideHustleResponse)
def recommend_side_hustle(payload: SideHustleRequest):
    try:
        # 1. Normalisasi Input
        exp_input = payload.experience_level.strip().title()
        interest_input = payload.interest_category.strip().title()
        
        # 2. Penentuan Rate Berdasarkan Level (Sesuai Standar Freelance)
        rate_map = {"Beginner": 10.0, "Intermediate": 15.0, "Expert": 25.0}
        target_hourly_rate_usd = rate_map.get(exp_input, 15.0)

        # 3. Hitung Total Jam & Durasi Kerja
        total_hours_per_month = payload.available_hours_per_week * 4
        duration_days = total_hours_per_month / 8.0 
        
        # Bobot Platform sesuai tren di dataset (Toptal paling tinggi, Fiverr paling rendah)
        plat_weights = {
            "Toptal": 1.25, "Upwork": 1.15, "Freelancer": 1.0, 
            "PeoplePerHour": 0.95, "Fiverr": 0.85
        }
        # Bobot Project Type (Fixed biasanya memiliki sedikit premi harga)
        type_weights = {"Fixed": 1.1, "Hourly": 1.0}

        simulations = []
        sim_metadata = []
        
        for plat in PLATFORMS:
            for ptype in PROJECT_TYPES:
                feat_map = {col: 0.0 for col in SH_FEAT_COLS}
                
                if "Hourly_Rate" in feat_map: feat_map["Hourly_Rate"] = float(target_hourly_rate_usd)
                if "Job_Duration_Days" in feat_map: feat_map["Job_Duration_Days"] = float(duration_days)
                
                if f"Experience_Level_{exp_input}" in feat_map: feat_map[f"Experience_Level_{exp_input}"] = 1.0
                if f"Job_Category_{interest_input}" in feat_map: feat_map[f"Job_Category_{interest_input}"] = 1.0
                if f"Platform_{plat}" in feat_map: feat_map[f"Platform_{plat}"] = 1.0
                if f"Project_Type_{ptype}" in feat_map: feat_map[f"Project_Type_{ptype}"] = 1.0
                    
                simulations.append(feat_map)
                sim_metadata.append({"platform": plat, "project_type": ptype})

        # --- PROTEKSI ERROR INDEX & CONSISTENCY ---
        df_sim = pd.DataFrame(simulations)
        df_sim = df_sim.reindex(columns=SH_FEAT_COLS, fill_value=0.0)
        
        X_sim_scaled = SH_SCALER.transform(df_sim.values).astype(np.float32)
        tensor_input = tf.constant(X_sim_scaled)
        _, pred_succ_prob = SH_MODEL(tensor_input, training=False)
        
        results = []
        for i, meta in enumerate(sim_metadata):
            succ_prob = float(np.clip(pred_succ_prob[i][0], 0.0, 1.0))
            
            # 4. PERHITUNGAN GAJI BERVARIASI (Berdasarkan Platform & Project Type)
            p_mul = plat_weights.get(meta["platform"], 1.0)
            t_mul = type_weights.get(meta["project_type"], 1.0)
            
            earn_usd = total_hours_per_month * target_hourly_rate_usd * p_mul * t_mul
            earn_idr = earn_usd * USD_TO_IDR
            
            results.append({
                "job_category": interest_input,
                "platform": meta["platform"],
                "project_type": meta["project_type"],
                "predicted_monthly_earnings_idr": round(earn_idr, 2),
                "score": succ_prob 
            })
            
        # Urutkan berdasarkan peluang sukses tertinggi (AI Ranking)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Ambil 7 rekomendasi terbaik
        top_7 = results[:7]
        
        for item in top_7: 
            if "score" in item:
                del item["score"]

        return SideHustleResponse(recommendations=top_7)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(exc)}")