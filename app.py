import streamlit as st
import numpy as np
import joblib

# ── Load Model ─────────────────
model      = joblib.load('best_model.pkl')
scaler     = joblib.load('scaler.pkl')
le_cut     = joblib.load('le_cut.pkl')
le_color   = joblib.load('le_color.pkl')
le_clarity = joblib.load('le_clarity.pkl')

CUT_OPTIONS     = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
COLOR_OPTIONS   = ['D','E','F','G','H','I','J']
CLARITY_OPTIONS = ['IF','VVS1','VVS2','VS1','VS2','SI1','SI2','I1']

st.set_page_config(
    page_title="Prediksi Harga Berlian",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    color: #1f4e79;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(180deg,#dff4ff,#c7ecff);
}

/* Main card */
.block-container {
    background: white;
    padding: 2rem 2.5rem;
    border-radius: 18px;
    box-shadow: 0 10px 25px rgba(0,0,0,0.08);
}

/* Title */
h1 {
    text-align:center;
    font-size:30px;
    color:#2b6fa6;
    font-weight:650;
    letter-spacing:0.5px;
}

/* Subheader Streamlit */
[data-testid="stSubheader"] {
    font-size:12px !important;
    color:#3b79a8 !important;
    margin-top:6px !important;
    margin-bottom:6px !important;
    font-weight:600 !important;
}

/* Selectbox & Input */
label {
    font-size:14px;
    color:#3b6f9e;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg,#8ed4ff,#6fc7ff);
    color:#13496d;
    font-size:15px;
    font-weight:600;
    border:none;
    border-radius:12px;
    padding:10px 18px;
    transition:0.25s;
}

.stButton > button:hover {
    background: linear-gradient(90deg,#6fc7ff,#52b8ff);
    color:white;
    transform: scale(1.03);
}

/* Result box */
.result-box {
    background:#f2fbff;
    padding:25px;
    border-radius:18px;
    text-align:center;
    margin-top:25px;
    box-shadow:0 8px 18px rgba(0,0,0,0.05);
}

.price {
    font-size:36px;
    font-weight:700;
    color:#2c6aa0;
}

.range {
    font-size:13px;
    color:#6aa2c9;
}

</style>
""", unsafe_allow_html=True)

# ── HEADER ─────────────────
st.title("୨ৎ Diamond Price Predictor ୨ৎ")

st.divider()

# ── INPUT ─────────────────
st.subheader("Kualitas Berlian")

col1,col2,col3 = st.columns(3)

with col1:
    cut = st.selectbox("Cut", CUT_OPTIONS, index=4)

with col2:
    color = st.selectbox("Color", COLOR_OPTIONS, index=1)

with col3:
    clarity = st.selectbox("Clarity", CLARITY_OPTIONS, index=6)

st.subheader("Proporsi Berlian")

col4,col5,col6 = st.columns(3)

with col4:
    carat = st.number_input("Carat",0.2,5.0,0.23)

with col5:
    depth = st.number_input("Depth",43.0,79.0,61.5)

with col6:
    table = st.number_input("Table",49.0,95.0,55.0)

st.subheader("Dimensi (mm)")

col7,col8,col9 = st.columns(3)

with col7:
    x = st.number_input("X (Length)",0.0,10.74,3.95)

with col8:
    y = st.number_input("Y (Width)",0.0,58.9,3.98)

with col9:
    z = st.number_input("Z (Depth)",0.0,31.8,2.43)

# ── PREDICT ─────────────────
if st.button("Prediksi Harga"):

    cut_enc     = le_cut.transform([cut])[0]
    color_enc   = le_color.transform([color])[0]
    clarity_enc = le_clarity.transform([clarity])[0]

    fitur = np.array([[carat,cut_enc,color_enc,clarity_enc,depth,table,x,y,z]])
    fitur_scaled = scaler.transform(fitur)

    harga = model.predict(fitur_scaled)[0]

    st.markdown(f"""
    <div class="result-box">
        <div class="price">${harga:,.0f}</div>
        <div class="range">
        estimasi harga ${harga*0.9:,.0f} — ${harga*1.1:,.0f}
        </div>
    </div>
    """, unsafe_allow_html=True)