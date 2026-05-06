from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.saving import register_keras_serializable
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =========================================
# 1. SETUP PATHS & KONSTANTA
# =========================================
ARTIFACT_DIR = Path("artifacts")
UNIT_SCALE = 2500.0  # Konversi IDR (Konsisten dengan training dataset)
USD_TO_IDR = 17000.0 # Konversi Prediksi Earnings USD ke IDR (Untuk Side Hustle)
INS_DEFAULT_INCOME_TYPE = "Salary"
INS_DEFAULT_MAIN_CATEGORY = "Utilities"

def _safe_div(n: float, d: float) -> float:
    d = float(d)
    return float(n) / d if d > 0 else 0.0

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


# --- Classification custom layer (required to load classification_model.keras) ---
# Penting: model disimpan dengan registered_name `finary>ResidualDenseBlock`,
# jadi package harus sama agar `load_model()` bisa menemukan kelasnya.
@register_keras_serializable(package="finary")
class ResidualDenseBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        units: int,
        dropout: float,
        l2: float,
        activation: str = "gelu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        # Implementasi ini harus SELARAS dengan training notebook `finary_classify_model.ipynb`
        # agar bobot pada file `.keras` bisa dimuat tanpa mismatch.
        reg = tf.keras.regularizers.l2(float(l2))
        self.units = int(units)
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.activation = str(activation)

        self.dense1 = tf.keras.layers.Dense(self.units, kernel_regularizer=reg)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.drop1 = tf.keras.layers.Dropout(self.dropout)

        self.dense2 = tf.keras.layers.Dense(self.units, kernel_regularizer=reg)
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.drop2 = tf.keras.layers.Dropout(self.dropout)

        self.proj: tf.keras.layers.Layer | None = None

    def _act(self, x):
        return tf.keras.activations.gelu(x) if self.activation.lower() == "gelu" else tf.nn.relu(x)

    def build(self, input_shape):
        in_units = int(input_shape[-1])
        if in_units != self.units:
            self.proj = tf.keras.layers.Dense(self.units)
        super().build(input_shape)

    def call(self, x, training=False):
        skip = self.proj(x) if self.proj is not None else x

        y = self.dense1(x)
        y = self.bn1(y, training=training)
        y = self._act(y)
        y = self.drop1(y, training=training)

        y = self.dense2(y)
        y = self.bn2(y, training=training)
        y = self._act(y)
        y = self.drop2(y, training=training)

        return skip + y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "units": self.units,
                "dropout": self.dropout,
                "l2": self.l2,
                "activation": self.activation,
            }
        )
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

# --- Classification (Financial Scenario) ---
class ClassifyRequest(BaseModel):
    """Payload minimal untuk klasifikasi kondisi keuangan bulanan.

    Catatan:
    - Semua input uang dalam **IDR asli**.
    - Backend akan melakukan scaling internal (IDR -> training space) dengan `UNIT_SCALE`
      dan menghitung fitur turunan secara deterministik.
    - Frontend **tidak perlu** mengirim fitur turunan seperti `net_cash_flow`, `expense_ratio`, dll.
    """

    monthly_income: float = Field(..., gt=0, description="Pendapatan bulanan (IDR). Wajib > 0.")
    monthly_expense_total: float = Field(..., ge=0, description="Total pengeluaran bulanan (IDR). Wajib >= 0.")
    actual_savings: float = Field(..., ge=0, description="Tabungan aktual bulan ini (IDR). Wajib >= 0.")
    budget_goal: float = Field(..., ge=0, description="Target tabungan/budget goal bulanan (IDR). Wajib >= 0.")
    emergency_fund: float = Field(..., ge=0, description="Dana darurat saat ini (IDR). Wajib >= 0.")

    # OpenAPI examples (Pydantic v2). Safe to keep even if ignored.
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "monthly_income": 9000000,
                    "monthly_expense_total": 5500000,
                    "actual_savings": 2500000,
                    "budget_goal": 2000000,
                    "emergency_fund": 15000000,
                }
            ]
        }
    }

class ClassificationProbabilities(BaseModel):
    survival: float
    stable: float
    growth: float


class ClassificationFinancialIndicators(BaseModel):
    monthly_income: float
    monthly_expense_total: float
    actual_savings: float
    budget_goal: float
    emergency_fund: float
    net_cash_flow: float
    expense_ratio: float
    savings_rate: float
    spending_efficiency: float


class ClassificationRiskFlags(BaseModel):
    negative_cash_flow: bool
    high_expense_ratio: bool
    low_savings_rate: bool
    savings_goal_not_met: bool
    low_spending_efficiency: bool


class ClassifyResponse(BaseModel):
    classification: str
    score: float
    probabilities: ClassificationProbabilities
    financial_indicators: ClassificationFinancialIndicators
    risk_flags: ClassificationRiskFlags
    recommendation_focus: List[str]
    explanation: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "classification": "growth",
                    "score": 0.94,
                    "probabilities": {"survival": 0.01, "stable": 0.05, "growth": 0.94},
                    "financial_indicators": {
                        "monthly_income": 9000000,
                        "monthly_expense_total": 5500000,
                        "actual_savings": 2500000,
                        "budget_goal": 2000000,
                        "emergency_fund": 15000000,
                        "net_cash_flow": 3500000,
                        "expense_ratio": 0.6111,
                        "savings_rate": 0.2778,
                        "spending_efficiency": 0.6364,
                    },
                    "risk_flags": {
                        "negative_cash_flow": False,
                        "high_expense_ratio": False,
                        "low_savings_rate": False,
                        "savings_goal_not_met": False,
                        "low_spending_efficiency": False,
                    },
                    "recommendation_focus": [
                        "maintain_growth_momentum",
                        "increase_investment_allocation",
                        "optimize_long_term_savings",
                    ],
                    "explanation": "The user is classified as growth because monthly cash flow and savings indicators are strong. Model confidence: 0.94.",
                }
            ]
        }
    }

# =========================================
# 4. LOAD ARTIFACTS (MODEL PRODUCTION)
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

# --- Classification Model ---
CLS_MODEL = None
CLS_SCALER = None
CLS_FEAT_COLS: list[str] = []
CLS_LABEL_MAPPING: dict[str, str] = {}
try:
    CLS_MODEL = tf.keras.models.load_model(
        ARTIFACT_DIR / "classification_model.keras",
        # Keras 3 menyimpan registered_name `finary>ResidualDenseBlock`, jadi kita map keduanya.
        custom_objects={
            "ResidualDenseBlock": ResidualDenseBlock,
            "finary>ResidualDenseBlock": ResidualDenseBlock,
        },
        compile=False,
    )
    CLS_SCALER = joblib.load(ARTIFACT_DIR / "classification_scaler.joblib")
    with open(ARTIFACT_DIR / "classification_feature_columns.json", "r", encoding="utf-8") as f:
        CLS_FEAT_COLS = json.load(f)
    with open(ARTIFACT_DIR / "classification_label_mapping.json", "r", encoding="utf-8") as f:
        CLS_LABEL_MAPPING = json.load(f)
except Exception:
    # Jangan mematikan service untuk endpoint lain.
    CLS_MODEL = None
    CLS_SCALER = None
    CLS_FEAT_COLS = []
    CLS_LABEL_MAPPING = {}

# =========================================
# 5. APLIKASI FASTAPI
# =========================================
app = FastAPI(title="FINARY AI Microservices", version="2.0.0")

@app.get("/health")
def health():
    return {"status": "ok", "message": "Classification, Insight, and Side Hustle models loaded."}

# -----------------------------------------
# ENDPOINT 1: CLASSIFICATION (MONTHLY FINANCIAL CONDITION)
# -----------------------------------------
def build_classification_features(payload: ClassifyRequest) -> tuple[Dict[str, float], Dict[str, float], Dict[str, bool]]:
    # 1) Raw input (IDR)
    inc_idr = float(payload.monthly_income)
    exp_idr = float(payload.monthly_expense_total)
    sav_idr = float(payload.actual_savings)
    goal_idr = float(payload.budget_goal)
    emg_idr = float(payload.emergency_fund)

    # 2) Derived indicators in IDR (for response)
    net_cf_idr = inc_idr - exp_idr
    expense_ratio = _safe_div(exp_idr, inc_idr)
    savings_rate = _safe_div(sav_idr, inc_idr)
    savings_goal_met = 1.0 if sav_idr >= goal_idr else 0.0
    spending_efficiency = _safe_div(net_cf_idr, exp_idr)

    indicators = {
        "monthly_income": inc_idr,
        "monthly_expense_total": exp_idr,
        "actual_savings": sav_idr,
        "budget_goal": goal_idr,
        "emergency_fund": emg_idr,
        "net_cash_flow": net_cf_idr,
        "expense_ratio": expense_ratio,
        "savings_rate": savings_rate,
        "spending_efficiency": spending_efficiency,
    }

    # 3) Risk flags (threshold sesuai spesifikasi)
    risk_flags = {
        "negative_cash_flow": (net_cf_idr / UNIT_SCALE) < 0,  # sama tanda, pakai scaled space
        "high_expense_ratio": expense_ratio >= 0.85,
        "low_savings_rate": savings_rate < 0.10,
        "savings_goal_not_met": savings_goal_met == 0.0,
        "low_spending_efficiency": spending_efficiency < 0.10,
    }

    # 4) Build model features in training space (scaled)
    inc = inc_idr / UNIT_SCALE
    exp = exp_idr / UNIT_SCALE
    sav = sav_idr / UNIT_SCALE
    goal = goal_idr / UNIT_SCALE
    emg = emg_idr / UNIT_SCALE

    net_cf = inc - exp
    exp_ratio_scaled = _safe_div(exp, inc)
    sav_rate_scaled = _safe_div(sav, inc)
    goal_met_scaled = 1.0 if sav >= goal else 0.0
    spend_eff_scaled = _safe_div(net_cf, exp)

    features = {col: 0.0 for col in CLS_FEAT_COLS}
    update = {
        "monthly_income": inc,
        "monthly_expense_total": exp,
        "actual_savings": sav,
        "net_cash_flow": net_cf,
        "expense_ratio": exp_ratio_scaled,
        "savings_rate": sav_rate_scaled,
        "budget_goal": goal,
        "savings_goal_met": goal_met_scaled,
        "emergency_fund": emg,
        "spending_efficiency": spend_eff_scaled,
    }
    for k, v in update.items():
        if k in features:
            features[k] = float(v)

    return features, indicators, risk_flags

def build_classification_recommendations(
    classification: str,
    indicators: Dict[str, float],
    risk_flags: Dict[str, bool],
) -> List[str]:
    recs: List[str] = []

    if classification == "survival":
        recs.extend(["reduce_non_essential_expenses", "build_cashflow_recovery_plan", "prioritize_emergency_fund"])
    elif classification == "stable":
        recs.extend(["improve_saving_rate", "maintain_budget_discipline", "increase_financial_buffer"])
    elif classification == "growth":
        recs.extend(["maintain_growth_momentum", "increase_investment_allocation", "optimize_long_term_savings"])

    if risk_flags.get("negative_cash_flow"):
        recs.append("fix_negative_cash_flow")
    if risk_flags.get("high_expense_ratio"):
        recs.append("reduce_expense_ratio")
    if risk_flags.get("low_savings_rate"):
        recs.append("improve_saving_rate")
    if risk_flags.get("savings_goal_not_met"):
        recs.append("align_savings_with_goal")
    if risk_flags.get("low_spending_efficiency"):
        recs.append("improve_spending_efficiency")

    # de-dupe while preserving order
    return list(dict.fromkeys(recs))

def build_classification_explanation(
    classification: str,
    score: float,
    indicators: Dict[str, float],
    risk_flags: Dict[str, bool],
) -> str:
    if classification == "survival":
        base = "The user is classified as survival because monthly cash flow is negative or financial margin is very limited."
    elif classification == "stable":
        base = "The user is classified as stable because monthly cash flow is positive but financial margin still needs improvement."
    else:
        base = "The user is classified as growth because monthly cash flow and savings indicators are strong."
    return f"{base} Model confidence: {score:.2f}."


@app.post("/classify", response_model=ClassifyResponse)
def classify_financial_scenario(payload: ClassifyRequest):
    """Classify Monthly Financial Condition.

    Target classes: `survival`, `stable`, `growth`.

    Input uang dalam IDR asli. Backend akan:
    - melakukan scaling IDR -> training space (`UNIT_SCALE`)
    - menghitung fitur turunan deterministik
    - melakukan inference menggunakan artifact klasifikasi
    """
    try:
        if CLS_MODEL is None or CLS_SCALER is None or not CLS_FEAT_COLS or not CLS_LABEL_MAPPING:
            raise RuntimeError("Classification artifacts are not loaded.")

        features, indicators, risk_flags = build_classification_features(payload)

        df_input = pd.DataFrame([features]).reindex(columns=CLS_FEAT_COLS, fill_value=0.0)
        X_scaled = CLS_SCALER.transform(df_input.values)

        pred = CLS_MODEL.predict(X_scaled, verbose=0)[0]
        pred = np.clip(pred, 0.0, 1.0)

        class_id = int(np.argmax(pred))
        score = float(np.max(pred))
        classification = CLS_LABEL_MAPPING[str(class_id)]

        probabilities = {
            CLS_LABEL_MAPPING[str(i)]: round(float(pred[i]), 4)
            for i in range(len(pred))
        }

        recommendations = build_classification_recommendations(
            classification=classification,
            indicators=indicators,
            risk_flags=risk_flags,
        )

        explanation = build_classification_explanation(
            classification=classification,
            score=score,
            indicators=indicators,
            risk_flags=risk_flags,
        )

        return ClassifyResponse(
            classification=classification,
            score=round(score, 4),
            probabilities=probabilities,
            financial_indicators={k: round(float(v), 4) for k, v in indicators.items()},
            risk_flags=risk_flags,
            recommendation_focus=recommendations,
            explanation=explanation,
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Classification inference error: {str(exc)}")

# -----------------------------------------
# ENDPOINT 2: INSIGHT & WARNING
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
        
        default_income_type = INS_DEFAULT_INCOME_TYPE
        default_main_category = INS_DEFAULT_MAIN_CATEGORY
        if f"income_type_{default_income_type}" in features:
            features[f"income_type_{default_income_type}"] = 1.0
        if f"category_{default_main_category}" in features:
            features[f"category_{default_main_category}"] = 1.0
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
# ENDPOINT 3: SIDE HUSTLE RECOMMENDATION (7 Rekomendasi & Variasi Platform)
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
