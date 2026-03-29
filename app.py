import streamlit as st
import joblib
import numpy as np
import pandas as pd
import time

# ─────────────────────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesAI — Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #080c14;
    color: #e8eaf0;
}

.stApp {
    background: radial-gradient(ellipse 120% 80% at 10% 0%, #0d1f3c 0%, #080c14 55%),
                radial-gradient(ellipse 60% 60% at 90% 100%, #0a1a2e 0%, transparent 60%);
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem !important; max-width: 1200px !important; }

/* Hero */
.hero {
    text-align: center;
    padding: 3rem 0 2rem;
    animation: fadeDown 0.8s ease both;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(56,189,248,0.1);
    border: 1px solid rgba(56,189,248,0.25);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1.4rem;
    font-family: 'Syne', sans-serif;
    font-weight: 600;
}
.hero-badge::before {
    content: '';
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #38bdf8;
    animation: pulse-dot 2s ease-in-out infinite;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1.1;
    background: linear-gradient(135deg, #e8eaf0 30%, #38bdf8 70%, #818cf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.8rem;
}
.hero p {
    color: #8892a4;
    font-size: 1.05rem;
    max-width: 520px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 300;
}

/* Stats row */
.stats-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.8rem 0 2.5rem;
    animation: fadeUp 0.7s ease 0.2s both;
}
.stat-chip {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 0.9rem 1.6rem;
    text-align: center;
    min-width: 130px;
}
.stat-chip .val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.stat-chip .lbl {
    font-size: 0.72rem;
    color: #5a6478;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 2px;
}

/* Section label */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #38bdf8;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(56,189,248,0.3) 0%, transparent 100%);
}

/* Glass card */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: 2rem;
    backdrop-filter: blur(12px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
    animation: fadeUp 0.6s ease both;
}
.glass-card:hover {
    border-color: rgba(56,189,248,0.2);
    box-shadow: 0 0 40px rgba(56,189,248,0.05);
}

/* Widget overrides */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stSelectbox"] > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    color: #e8eaf0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
label[data-testid="stWidgetLabel"] > div > p {
    color: #8892a4 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
}

/* Button */
div[data-testid="stButton"] > button {
    width: 100%;
    padding: 1rem 2rem;
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    border: none !important;
    border-radius: 14px !important;
    color: #fff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 30px rgba(14,165,233,0.3) !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 40px rgba(14,165,233,0.45) !important;
}

/* Result box */
.result-box {
    border-radius: 24px;
    padding: 2.5rem;
    text-align: center;
    animation: resultReveal 0.7s cubic-bezier(0.34,1.56,0.64,1) both;
}
.result-box.low    { background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(5,150,105,0.06)); border: 1px solid rgba(16,185,129,0.3); box-shadow: 0 0 60px rgba(16,185,129,0.1); }
.result-box.high   { background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(220,38,38,0.06));   border: 1px solid rgba(239,68,68,0.3);   box-shadow: 0 0 60px rgba(239,68,68,0.1); }
.result-box.medium { background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(217,119,6,0.06)); border: 1px solid rgba(245,158,11,0.3); box-shadow: 0 0 60px rgba(245,158,11,0.1); }
.result-icon  { font-size: 3.5rem; margin-bottom: 0.8rem; animation: iconBounce 0.6s ease 0.3s both; display: block; }
.result-label { font-family: 'Syne', sans-serif; font-size: 0.75rem; letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 0.4rem; opacity: 0.7; }
.result-title { font-family: 'Syne', sans-serif; font-size: 2rem; font-weight: 800; margin-bottom: 0.5rem; }
.result-box.low .result-title    { color: #10b981; }
.result-box.high .result-title   { color: #ef4444; }
.result-box.medium .result-title { color: #f59e0b; }
.result-subtitle { font-size: 0.9rem; color: #8892a4; line-height: 1.5; max-width: 300px; margin: 0 auto; }

/* Confidence bars */
.conf-wrap {
    margin-top: 1.2rem;
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    border: 1px solid rgba(255,255,255,0.06);
    text-align: left;
}
.conf-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: #8892a4;
    margin-bottom: 0.7rem;
    font-family: 'Syne', sans-serif;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.conf-track { background: rgba(255,255,255,0.06); border-radius: 100px; height: 10px; overflow: hidden; }
.conf-fill  { height: 100%; border-radius: 100px; animation: fillBar 1.2s cubic-bezier(0.4,0,0.2,1) 0.4s both; }
.conf-fill.low    { background: linear-gradient(90deg, #059669, #10b981); }
.conf-fill.high   { background: linear-gradient(90deg, #dc2626, #ef4444); }
.conf-fill.medium { background: linear-gradient(90deg, #d97706, #f59e0b); }

/* Feature bars */
.feat-row  { display: flex; align-items: center; gap: 10px; margin-bottom: 0.8rem; }
.feat-name { font-size: 0.78rem; color: #8892a4; width: 150px; flex-shrink: 0; font-family: 'Syne', sans-serif; }
.feat-track { flex: 1; background: rgba(255,255,255,0.05); border-radius: 100px; height: 6px; overflow: hidden; }
.feat-bar  { height: 100%; border-radius: 100px; background: linear-gradient(90deg, #0ea5e9, #818cf8); animation: fillBar 1s ease both; }
.feat-val  { font-size: 0.75rem; color: #38bdf8; width: 40px; text-align: right; font-family: 'Syne', sans-serif; }

/* Misc */
.glowing-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(56,189,248,0.3), transparent); margin: 2.5rem 0; }
.disclaimer {
    background: rgba(245,158,11,0.06);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 12px;
    padding: 0.9rem 1.2rem;
    font-size: 0.78rem;
    color: #fbbf24;
    opacity: 0.8;
    display: flex;
    gap: 10px;
    align-items: flex-start;
    margin-top: 1.5rem;
    line-height: 1.5;
}

/* Animations */
@keyframes fadeDown   { from { opacity:0; transform:translateY(-20px); } to { opacity:1; transform:translateY(0); } }
@keyframes fadeUp     { from { opacity:0; transform:translateY(20px);  } to { opacity:1; transform:translateY(0); } }
@keyframes resultReveal { from { opacity:0; transform:scale(0.9); } to { opacity:1; transform:scale(1); } }
@keyframes iconBounce { from { opacity:0; transform:scale(0.5); } to { opacity:1; transform:scale(1); } }
@keyframes fillBar    { from { width:0 !important; } }
@keyframes pulse-dot  { 0%,100%{ opacity:1; transform:scale(1); } 50%{ opacity:0.5; transform:scale(0.7); } }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Model
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load("model.pkl")

model = load_model()

FEATURE_COLS = [
    'age', 'hypertension', 'heart_disease', 'bmi',
    'HbA1c_level', 'blood_glucose_level',
    'gender_Male',
    'smoking_history_current', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never',
    'smoking_history_not current'
]

# ─────────────────────────────────────────────────────────────────────────────
# Hero
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI-Powered Clinical Tool</div>
    <h1>Diabetes Risk Predictor</h1>
    <p>Enter patient vitals and clinical history to receive an instant,
       AI-powered diabetes risk assessment.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stats-row">
    <div class="stat-chip"><div class="val">GBM</div><div class="lbl">Model</div></div>
    <div class="stat-chip"><div class="val">8</div><div class="lbl">Features</div></div>
    <div class="stat-chip"><div class="val">100K</div><div class="lbl">Training Rows</div></div>
    <div class="stat-chip"><div class="val">Binary</div><div class="lbl">Output</div></div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="glowing-divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Patient Information</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 👤 Demographics")
    gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    age    = st.slider("Age (years)", 0, 80, 35)
    bmi    = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1, format="%.1f")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 🩸 Lab Results")
    hba1c   = st.number_input("HbA1c Level (%)", min_value=3.5, max_value=9.0, value=5.5, step=0.1, format="%.1f")
    glucose = st.number_input("Blood Glucose (mg/dL)", min_value=80, max_value=300, value=130, step=1)
    smoking = st.selectbox("Smoking History", ["No Info", "never", "former", "current", "ever", "not current"])
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("##### 🏥 Medical History")
    hypertension  = st.radio("Hypertension",  ["No", "Yes"], horizontal=True)
    st.write("")
    heart_disease = st.radio("Heart Disease", ["No", "Yes"], horizontal=True)
    st.write("")
    st.write("")
    st.write("")
    predict_btn = st.button("🔍  Analyze Patient", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([[
        age,
        1 if hypertension  == "Yes" else 0,
        1 if heart_disease == "Yes" else 0,
        bmi, hba1c, glucose,
        1 if gender  == "Male"        else 0,
        1 if smoking == "current"     else 0,
        1 if smoking == "ever"        else 0,
        1 if smoking == "former"      else 0,
        1 if smoking == "never"       else 0,
        1 if smoking == "not current" else 0,
    ]], columns=FEATURE_COLS)

    with st.spinner("Analyzing patient data…"):
        time.sleep(0.9)
        prediction    = model.predict(input_data)[0]
        proba         = model.predict_proba(input_data)[0]

    diabetic_prob = proba[1] * 100
    healthy_prob  = proba[0] * 100

    if diabetic_prob >= 60:
        risk_class = "high";   icon = "🔴"; title = "High Risk"
        subtitle = "Strong indicators for diabetes detected. Immediate clinical follow-up is advised."
    elif diabetic_prob >= 35:
        risk_class = "medium"; icon = "🟡"; title = "Moderate Risk"
        subtitle = "Some risk factors present. Lifestyle changes and monitoring are recommended."
    else:
        risk_class = "low";    icon = "🟢"; title = "Low Risk"
        subtitle = "No significant diabetes risk indicators detected at this time."

    st.markdown('<div class="glowing-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">Prediction Result</div>', unsafe_allow_html=True)

    res_col, feat_col = st.columns([1, 1], gap="large")

    # Result card
    with res_col:
        st.markdown(f"""
        <div class="result-box {risk_class}">
            <span class="result-icon">{icon}</span>
            <div class="result-label">Diagnosis Assessment</div>
            <div class="result-title">{title}</div>
            <div class="result-subtitle">{subtitle}</div>
            <div class="conf-wrap">
                <div class="conf-header"><span>Diabetes Probability</span><span>{diabetic_prob:.1f}%</span></div>
                <div class="conf-track"><div class="conf-fill {risk_class}" style="width:{diabetic_prob:.1f}%"></div></div>
            </div>
            <div class="conf-wrap">
                <div class="conf-header"><span>Healthy Probability</span><span>{healthy_prob:.1f}%</span></div>
                <div class="conf-track"><div class="conf-fill low" style="width:{healthy_prob:.1f}%"></div></div>
            </div>
        </div>
        <div class="disclaimer">
            ⚠️ This tool is for informational purposes only and does not constitute
            medical advice. Always consult a licensed healthcare professional.
        </div>
        """, unsafe_allow_html=True)

    # Risk factors + summary
    with feat_col:
        st.markdown('<div class="section-label">Key Risk Factors</div>', unsafe_allow_html=True)

        factors = {
            "Blood Glucose":  min(glucose / 300, 1.0),
            "HbA1c Level":    min((hba1c - 3.5) / 5.5, 1.0),
            "BMI":            min((bmi - 10) / 60, 1.0),
            "Age":            min(age / 80, 1.0),
            "Hypertension":   1.0  if hypertension  == "Yes" else 0.1,
            "Heart Disease":  1.0  if heart_disease == "Yes" else 0.1,
            "Smoking":        0.85 if smoking == "current" else (0.5 if smoking == "former" else 0.1),
        }

        bars_html = "".join(f"""
        <div class="feat-row">
            <span class="feat-name">{name}</span>
            <div class="feat-track"><div class="feat-bar" style="width:{val*100:.0f}%"></div></div>
            <span class="feat-val">{val*100:.0f}%</span>
        </div>""" for name, val in factors.items())

        st.markdown(f'<div class="glass-card">{bars_html}</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1.5rem">Input Summary</div>', unsafe_allow_html=True)
        summary = pd.DataFrame({
            "Parameter": ["Age", "BMI", "HbA1c", "Blood Glucose", "Gender", "Smoking", "Hypertension", "Heart Disease"],
            "Value":     [f"{age} yrs", f"{bmi:.1f}", f"{hba1c:.1f}%", f"{glucose} mg/dL",
                          gender, smoking, hypertension, heart_disease]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)