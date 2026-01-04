# app.py
import warnings
warnings.filterwarnings('ignore')

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler

# Optional: Twilio for SMS/voice alerts
TWILIO_ENABLED = False
try:
    from twilio.rest import Client
    TWILIO_ENABLED = True
except:
    TWILIO_ENABLED = False

# --------------- Page Setup ---------------
st.set_page_config(page_title="CropBalance AI", page_icon="🌾", layout="wide")
st.markdown("""
<style>
    .big { font-size: 20px; }
    .metric-good { color: #1a7f37; }
    .metric-bad { color: #b60205; }
    .pill { padding: 4px 10px; border-radius: 16px; display: inline-block; font-weight: 600; }
    .pill-green { background: #e6ffed; color: #1a7f37; }
    .pill-yellow { background: #fff5b1; color: #735c0f; }
    .pill-red { background: #ffeef0; color: #b60205; }
    .card { border: 1px solid #e5e7eb; border-radius: 10px; padding: 14px; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 CropBalance AI")
st.markdown("### 🚀 AI-Powered Market Intelligence for Indian Farmers")

# --------------- Data Loading ---------------
def load_or_create_data():
    try:
        df = pd.read_csv('crop_data.csv')
        # Expect columns: Crop_Name, Previous_Year_High_Price, Rainfall_Level, Total_Area_Sown, Yield_per_Hectare(optional)
        if 'Yield_per_Hectare' not in df.columns:
            df['Yield_per_Hectare'] = np.random.randint(15, 35, size=len(df))  # synthetic yield (quintal/ha)
        return df, True
    except FileNotFoundError:
        # Create synthetic demo dataset
        np.random.seed(42)
        crops = ['Onion', 'Wheat', 'Soyabean', 'Maize', 'Rice', 'Bajra', 'Jowar']
        rainfall_levels = ['Low', 'Medium', 'High']
        rows = []
        for crop in crops:
            for year in range(2018, 2025):
                for rl in rainfall_levels:
                    base_price = {
                        'Onion': 3000, 'Wheat': 2200, 'Soyabean': 3500,
                        'Maize': 1800, 'Rice': 2400, 'Bajra': 1700, 'Jowar': 1600
                    }[crop]
                    price_noise = np.random.randint(-600, 800)
                    prev_high_price = max(1000, base_price + price_noise)
                    rain_score = {'Low': 1, 'Medium': 2, 'High': 3}[rl]
                    # Oversupply logic: higher previous price -> more area sown (herd mentality)
                    area_sown = int(500 + 0.9 * prev_high_price / 5 + rain_score * 300 + np.random.randint(-200, 200))
                    yield_ha = int(12 + rain_score * 7 + np.random.randint(-3, 4))  # synthetic yield
                    rows.append({
                        'Year': year,
                        'Crop_Name': crop,
                        'Previous_Year_High_Price': prev_high_price,
                        'Rainfall_Level': rl,
                        'Total_Area_Sown': max(100, area_sown),
                        'Yield_per_Hectare': max(5, yield_ha)
                    })
        df = pd.DataFrame(rows)
        return df, False

df, has_file = load_or_create_data()
graph_df = df.copy()

# --------------- Model Training ---------------
df['Rainfall_Score'] = df['Rainfall_Level'].map({'Low': 1, 'Medium': 2, 'High': 3})
X = df[['Previous_Year_High_Price', 'Rainfall_Score']]
y_area = df['Total_Area_Sown']
area_model = LinearRegression()
area_model.fit(X, y_area)

# Simple price model: price forecast informed by predicted area (inverse elasticity)
# We’ll fit a regression price ~ f(prev_price, rainfall_score, area_sown)
# Create synthetic historical price from prev_high_price with noise to fit the model
hist_price = df['Previous_Year_High_Price'] * (0.75 + np.random.uniform(0.8, 1.2, len(df))) - 0.0006 * df['Total_Area_Sown']
df['Market_Price'] = np.clip(hist_price, 800, None)
X_price = df[['Previous_Year_High_Price', 'Rainfall_Score', 'Total_Area_Sown']]
y_price = df['Market_Price']
price_model = LinearRegression()
price_model.fit(X_price, y_price)

# --------------- Utility Functions ---------------
def rainfall_to_score(level: str) -> int:
    return {'Low': 1, 'Medium': 2, 'High': 3}[level]

def compute_sri(predicted_area: float, crop_name: str) -> int:
    # Normalize against historical max area for that crop
    hist = df[df['Crop_Name'] == crop_name]
    if hist.empty:
        max_area = df['Total_Area_Sown'].max()
    else:
        max_area = max(1000, hist['Total_Area_Sown'].max())
    score = int(np.clip((predicted_area / max_area) * 100, 0, 100))
    return score

def sri_label(score: int) -> str:
    if score < 35:
        return "SAFE"
    elif score < 70:
        return "CAUTION"
    else:
        return "HIGH RISK"

def sri_pill(score: int) -> str:
    lbl = sri_label(score)
    if lbl == "SAFE":
        klass = "pill-green"
    elif lbl == "CAUTION":
        klass = "pill-yellow"
    else:
        klass = "pill-red"
    return f'<span class="pill {klass}">{lbl} ({score}/100)</span>'

def forecast_price(prev_price: float, rain_score: int, predicted_area: float):
    # Price decreases with higher predicted area; add uncertainty band
    base_forecast = price_model.predict([[prev_price, rain_score, predicted_area]])[0]
    # Confidence interval (±10% as illustrative)
    lower = base_forecast * 0.9
    upper = base_forecast * 1.1
    return max(800, base_forecast), lower, upper

def get_best_crop(rain_level: str):
    if rain_level == "High":
        return ["Rice", "Sugarcane"]
    elif rain_level == "Medium":
        return ["Soyabean", "Maize"]
    else:
        return ["Bajra", "Jowar"]

def yield_estimate(crop_name: str, rain_level: str):
    # Use historical yield if available, else heuristic
    subset = df[(df['Crop_Name'] == crop_name) & (df['Rainfall_Level'] == rain_level)]
    if not subset.empty:
        return int(subset['Yield_per_Hectare'].median())
    base = {'Onion': 20, 'Wheat': 18, 'Soyabean': 16, 'Maize': 22, 'Rice': 28, 'Bajra': 14, 'Jowar': 15}
    adjust = {'Low': -2, 'Medium': 0, 'High': 2}[rain_level]
    return max(8, base.get(crop_name, 16) + adjust)

def send_twilio_alert(phone_number: str, message: str):
    if not TWILIO_ENABLED:
        st.warning("Twilio not installed. Install with 'pip install twilio' and set environment variables.")
        return False
    sid = os.getenv("TWILIO_ACCOUNT_SID")
    token = os.getenv("TWILIO_AUTH_TOKEN")
    from_num = os.getenv("TWILIO_FROM_NUMBER")
    if not (sid and token and from_num):
        st.error("Twilio environment variables not set.")
        return False
    client = Client(sid, token)
    try:
        client.messages.create(to=phone_number, from_=from_num, body=message)
        return True
    except Exception as e:
        st.error(f"Twilio error: {e}")
        return False

# In-memory store for community intentions (replace with DB in production)
if 'intentions' not in st.session_state:
    st.session_state.intentions = pd.DataFrame(columns=['Timestamp', 'District', 'Crop_Name', 'Area_Hectares', 'Rainfall_Level'])

# --------------- Sidebar Inputs ---------------
st.sidebar.header("Farmer Input Panel")
crop_name = st.sidebar.selectbox("Select Crop", ["Onion", "Wheat", "Soyabean", "Maize", "Rice", "Bajra", "Jowar"])
prev_price = st.sidebar.number_input("Last Year's High Price (₹/Quintal)", 800, 10000, 5000, step=50)
rainfall = st.sidebar.select_slider("Rainfall Forecast", options=["Low", "Medium", "High"], value="Medium")
land_size = st.sidebar.number_input("Your Land Size (Hectares)", 0.5, 1000.0, 2.0, step=0.5)

cost_seed = st.sidebar.number_input("Cost: Seeds (₹/ha)", 0, 50000, 8000, step=100)
cost_fert = st.sidebar.number_input("Cost: Fertilizers (₹/ha)", 0, 50000, 6000, step=100)
cost_labor = st.sidebar.number_input("Cost: Labor (₹/ha)", 0, 100000, 12000, step=500)
cost_other = st.sidebar.number_input("Cost: Other (₹/ha)", 0, 50000, 4000, step=100)
analyze_btn = st.sidebar.button("🔍 Analyze Market Risk", use_container_width=True)

# --------------- Tabs ---------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "SRI & Price Forecast", "Profitability", "Scenario Simulator",
    "Community Map", "Explainable AI", "Alerts"
])

# --------------- Analysis ---------------
if analyze_btn:
    rain_score = rainfall_to_score(rainfall)
    predicted_area = float(area_model.predict([[prev_price, rain_score]])[0])
    sri = compute_sri(predicted_area, crop_name)
    forecast, lower, upper = forecast_price(prev_price, rain_score, predicted_area)

    # Shared context for all tabs
    st.session_state.analysis = {
        'predicted_area': predicted_area,
        'sri': sri,
        'forecast_price': forecast,
        'forecast_lower': lower,
        'forecast_upper': upper,
        'rain_score': rain_score,
        'rainfall': rainfall,
        'crop_name': crop_name,
        'prev_price': prev_price,
        'land_size': land_size
    }

# Fallback if not analyzed yet
if 'analysis' not in st.session_state:
    rain_score = rainfall_to_score(rainfall)
    predicted_area = float(area_model.predict([[prev_price, rain_score]])[0])
    sri = compute_sri(predicted_area, crop_name)
    forecast, lower, upper = forecast_price(prev_price, rain_score, predicted_area)
    st.session_state.analysis = {
        'predicted_area': predicted_area,
        'sri': sri,
        'forecast_price': forecast,
        'forecast_lower': lower,
        'forecast_upper': upper,
        'rain_score': rain_score,
        'rainfall': rainfall,
        'crop_name': crop_name,
        'prev_price': prev_price,
        'land_size': land_size
    }

analysis = st.session_state.analysis

# --------------- Tab 1: SRI & Price Forecast ---------------
with tab1:
    st.subheader("📊 Saturation Risk Index & Price Forecast")
    colA, colB, colC = st.columns(3)
    with colA:
        st.markdown("**Risk level:**", unsafe_allow_html=True)
        st.markdown(sri_pill(analysis['sri']), unsafe_allow_html=True)
        st.metric("Predicted Total Sowing (ha)", f"{int(analysis['predicted_area']):,}")
    with colB:
        st.metric("Forecasted Market Price (₹/Quintal)", f"{int(analysis['forecast_price']):,}")
        st.caption(f"Confidence range: ₹{int(analysis['forecast_lower']):,} – ₹{int(analysis['forecast_upper']):,}")
    with colC:
        recs = get_best_crop(analysis['rainfall'])
        st.info(f"💡 Recommendation: Consider {recs[0]} or {recs[1]} based on rainfall “{analysis['rainfall']}”.")

    # Price forecast chart (single-season band)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=['Now', 'Harvest'],
        y=[analysis['forecast_price']*0.98, analysis['forecast_price']],
        mode='lines+markers',
        name='Forecast',
        line=dict(color='royalblue')
    ))
    fig.add_trace(go.Scatter(
        x=['Now', 'Harvest'],
        y=[analysis['forecast_lower']*0.98, analysis['forecast_lower']],
        fill=None, mode='lines', line=dict(color='lightblue'), name='Lower'
    ))
    fig.add_trace(go.Scatter(
        x=['Now', 'Harvest'],
        y=[analysis['forecast_upper']*0.98, analysis['forecast_upper']],
        fill='tonexty', mode='lines', line=dict(color='lightblue'), name='Upper'
    ))
    fig.update_layout(title="Price forecast with confidence band", yaxis_title="₹/Quintal", xaxis_title="Timeline")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📈 Historical trend: Price vs Area (selected crop)")
    subset = graph_df[graph_df['Crop_Name'] == analysis['crop_name']]
    if subset.empty:
        st.write("(No historical data available for this crop in prototype)")
    else:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=subset['Previous_Year_High_Price'], y=subset['Total_Area_Sown'],
            mode='markers', name='History', marker=dict(color='darkorange')
        ))
        fig2.update_layout(xaxis_title="Previous Year High Price (₹/Quintal)", yaxis_title="Total Area Sown (ha)")
        st.plotly_chart(fig2, use_container_width=True)

# --------------- Tab 2: Profitability ---------------
with tab2:
    st.subheader("💸 Profitability Calculator")
    yield_ha = yield_estimate(analysis['crop_name'], analysis['rainfall'])
    total_cost_per_ha = cost_seed + cost_fert + cost_labor + cost_other

    # Revenue scenarios
    best_price = analysis['forecast_upper']
    base_price = analysis['forecast_price']
    worst_price = analysis['forecast_lower']

    revenue_best = yield_ha * best_price
    revenue_base = yield_ha * base_price
    revenue_worst = yield_ha * worst_price

    profit_best = revenue_best - total_cost_per_ha
    profit_base = revenue_base - total_cost_per_ha
    profit_worst = revenue_worst - total_cost_per_ha

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Estimated yield (quintal/ha)", f"{yield_ha}")
    with col2:
        st.metric("Cost per hectare (₹)", f"{int(total_cost_per_ha):,}")
    with col3:
        st.metric("Profit per ha (base)", f"₹{int(profit_base):,}")
    with col4:
        total_profit = profit_base * analysis['land_size']
        st.metric("Total profit (base)", f"₹{int(total_profit):,}")

    st.markdown("#### Scenario breakdown")
    fig = go.Figure(data=[
        go.Bar(name='Worst', x=['Revenue', 'Profit'], y=[revenue_worst, profit_worst], marker_color='#ff7f7f'),
        go.Bar(name='Base', x=['Revenue', 'Profit'], y=[revenue_base, profit_base], marker_color='#f1c40f'),
        go.Bar(name='Best', x=['Revenue', 'Profit'], y=[revenue_best, profit_best], marker_color='#2ecc71')
    ])
    fig.update_layout(barmode='group', yaxis_title="₹ per hectare", xaxis_title="Scenario")
    st.plotly_chart(fig, use_container_width=True)

    if analysis['sri'] >= 70:
        st.error("🛑 High saturation risk detected — base scenario may be optimistic. Consider diversification.")
    elif analysis['sri'] >= 35:
        st.warning("⚠️ Moderate risk — monitor market arrivals and be cautious with input costs.")
    else:
        st.success("✅ Low risk — profitability outlook is favorable under base scenario.")

# --------------- Tab 3: Scenario Simulator ---------------
with tab3:
    st.subheader("🎛️ Scenario Simulator")
    sim_prev_price = st.slider("Simulated previous year high price (₹/Quintal)", 800, 10000, int(analysis['prev_price']), step=50)
    sim_rainfall = st.select_slider("Simulated rainfall forecast", options=["Low", "Medium", "High"], value=analysis['rainfall'])

    sim_rain_score = rainfall_to_score(sim_rainfall)
    sim_pred_area = float(area_model.predict([[sim_prev_price, sim_rain_score]])[0])
    sim_sri = compute_sri(sim_pred_area, analysis['crop_name'])
    sim_forecast, sim_lower, sim_upper = forecast_price(sim_prev_price, sim_rain_score, sim_pred_area)

    colS1, colS2, colS3 = st.columns(3)
    with colS1:
        st.metric("Simulated sowing (ha)", f"{int(sim_pred_area):,}")
    with colS2:
        st.markdown(sri_pill(sim_sri), unsafe_allow_html=True)
    with colS3:
        st.metric("Simulated price (₹/Quintal)", f"{int(sim_forecast):,}")

    st.caption(f"Confidence range: ₹{int(sim_lower):,} – ₹{int(sim_upper):,}")

# --------------- Tab 4: Community Map ---------------
with tab4:
    st.subheader("🗺️ Community Sowing Intentions")
    st.markdown("Submit sowing intentions to reduce herd mentality via transparency.")

    # Minimal district lat/long for demo map
    district_coords = {
        'Solapur, Maharashtra': (17.6599, 75.9064),
        'Nashik, Maharashtra': (19.9975, 73.7898),
        'Indore, MP': (22.7196, 75.8577),
        'Bengaluru Rural, KA': (13.1900, 77.4600),
        'Jaipur, Rajasthan': (26.9124, 75.7873)
    }

    colM1, colM2 = st.columns(2)
    with colM1:
        district = st.selectbox("District", list(district_coords.keys()))
        intent_crop = st.selectbox("Crop", ["Onion", "Wheat", "Soyabean", "Maize", "Rice", "Bajra", "Jowar"])
        intent_area = st.number_input("Area to sow (ha)", 0.1, 500.0, 1.0, step=0.1)
        intent_rain = st.select_slider("Rainfall expectation", options=["Low", "Medium", "High"], value=analysis['rainfall'])
        if st.button("📥 Submit intention"):
            new_row = {
                'Timestamp': pd.Timestamp.now(),
                'District': district,
                'Crop_Name': intent_crop,
                'Area_Hectares': intent_area,
                'Rainfall_Level': intent_rain
            }
            st.session_state.intentions = pd.concat([st.session_state.intentions, pd.DataFrame([new_row])], ignore_index=True)
            st.success("Intention submitted successfully.")
    with colM2:
        st.markdown("#### Live intentions summary")
        if st.session_state.intentions.empty:
            st.write("(No intentions yet — be the first to submit!)")
        else:
            agg = st.session_state.intentions.groupby(['District', 'Crop_Name']).agg(
                Total_Area=('Area_Hectares', 'sum'), Submissions=('Crop_Name', 'count')
            ).reset_index()
            st.dataframe(agg)

    st.markdown("---")
    st.markdown("#### Basic sowing intensity map")
    try:
        import folium
        from streamlit.components.v1 import html
        fmap = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center on India
        if not st.session_state.intentions.empty:
            for _, row in st.session_state.intentions.iterrows():
                lat, lon = district_coords.get(row['District'], (20.5937, 78.9629))
                popup = f"{row['District']} | {row['Crop_Name']} | {row['Area_Hectares']} ha"
                folium.CircleMarker(location=(lat, lon), radius=min(12, 4 + int(row['Area_Hectares'])),
                                    color='red', fill=True, fill_color='orange', popup=popup).add_to(fmap)
        else:
            folium.Marker(location=[17.6599, 75.9064], popup="Solapur example").add_to(fmap)
        html(fmap._repr_html_(), height=450)
    except Exception as e:
        st.warning("Install 'folium' for map visualization. Showing a bar chart instead.")
        if not st.session_state.intentions.empty:
            agg2 = st.session_state.intentions.groupby('District')['Area_Hectares'].sum().reset_index()
            fig_map = go.Figure(go.Bar(x=agg2['District'], y=agg2['Area_Hectares'], marker_color='teal'))
            fig_map.update_layout(title="Total intended area by district", yaxis_title="Hectares")
            st.plotly_chart(fig_map, use_container_width=True)

# --------------- Tab 5: Explainable AI ---------------
with tab5:
    st.subheader("🔍 Explainable AI")
    st.markdown("Understand why the model predicts saturation and price.")

    # Feature importance for area model (coefficients)
    coef_area = pd.DataFrame({
        'Feature': ['Previous_Year_High_Price', 'Rainfall_Score'],
        'Coefficient': area_model.coef_
    }).sort_values(by='Coefficient', ascending=False)

    st.markdown("#### Area prediction driver importance")
    st.dataframe(coef_area, use_container_width=True)

    # Contribution breakdown for current inputs
    bias_area = area_model.intercept_
    contrib_prev = area_model.coef_[0] * analysis['prev_price']
    contrib_rain = area_model.coef_[1] * analysis['rain_score']
    total_pred_area = bias_area + contrib_prev + contrib_rain

    st.markdown("#### Contribution breakdown")
    fig_xai = go.Figure(go.Bar(
        x=['Intercept', 'Prev Price', 'Rainfall Score', 'Total'],
        y=[bias_area, contrib_prev, contrib_rain, total_pred_area],
        marker_color=['#95a5a6', '#3498db', '#2ecc71', '#9b59b6']
    ))
    fig_xai.update_layout(yaxis_title="Area (ha)")
    st.plotly_chart(fig_xai, use_container_width=True)

    st.info(f"Explanation: High previous prices increase sowing intentions; higher rainfall score raises expected sowing due to yield optimism.")

# --------------- Tab 6: Alerts ---------------
with tab6:
    st.subheader("📣 Voice & SMS Alerts")
    st.markdown("Send alerts to farmers in local languages for accessibility.")
    phone = st.text_input("Farmer phone number (+91XXXXXXXXXX)")
    language = st.selectbox("Language", ["English", "Hindi", "Marathi"])
    risk_label = sri_label(analysis['sri'])

    # Compose localized message
    base_msg_en = f"Alert: {analysis['crop_name']} {risk_label} in your district. Predicted sowing: {int(analysis['predicted_area'])} ha. Forecast price: ₹{int(analysis['forecast_price'])}/quintal. Consider {', '.join(get_best_crop(analysis['rainfall']))}."
    messages = {
        "English": base_msg_en,
        "Hindi": f"चेतावनी: {analysis['crop_name']} में {risk_label} जोखिम। अनुमानित बोना: {int(analysis['predicted_area'])} हेक्टेयर। अनुमानित कीमत: ₹{int(analysis['forecast_price'])}/क्विंटल। {', '.join(get_best_crop(analysis['rainfall']))} पर विचार करें।",
        "Marathi": f"सूचना: {analysis['crop_name']} साठी {risk_label} धोका. अंदाजे पेरणी: {int(analysis['predicted_area'])} हेक्टर. अंदाजे किंमत: ₹{int(analysis['forecast_price'])}/क्विंटल. {', '.join(get_best_crop(analysis['rainfall']))} विचारात घ्या."
    }
    msg = messages[language]

    colA, colB = st.columns(2)
    with colA:
        if st.button("📲 Send SMS"):
            if phone.strip():
                ok = send_twilio_alert(phone.strip(), msg)
                if ok:
                    st.success("SMS sent successfully.")
            else:
                st.error("Please enter a valid phone number.")
    with colB:
        st.caption("Voice call demo not implemented in this prototype. Use Twilio Voice API similarly if needed.")

# --------------- Footer ---------------
st.markdown("---")

import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import os
import requests

# ---------------- PAGE SETUP ----------------
st.set_page_config(page_title="CropBalance AI", page_icon="🌾", layout="wide")
st.title("🌾 CropBalance AI")
st.markdown("### 🚀 AI-Powered Market Intelligence for Indian Farmers")

# ---------------- LANGUAGE SUPPORT ----------------
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Bengali": "bn",
    "Assamese": "as",
    "Odia": "or",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
    "Konkani": "kok",
    "Sindhi": "sd",
    "Kashmiri": "ks",
    "Dogri": "doi",
    "Manipuri": "mni",
    "Bodo": "brx",
    "Santhali": "sat",
    "Maithili": "mai",
    "Nepali": "ne",
    "Sanskrit": "sa"
}

language = st.sidebar.selectbox("🌐 Choose Language", list(LANGUAGES.keys()))

# Manual dictionary for key UI phrases
translations = {
    "Analyze Market Risk": {
        "English": "Analyze Market Risk",
        "Hindi": "बाजार जोखिम का विश्लेषण करें",
        "Marathi": "बाजार जोखमीचे विश्लेषण करा",
        "Tamil": "சந்தை அபாயத்தை பகுப்பாய்வு செய்யவும்",
        "Telugu": "మార్కెట్ ప్రమాదాన్ని విశ్లేషించండి",
        "Gujarati": "બજાર જોખમનું વિશ્લેષણ કરો",
        "Punjabi": "ਬਾਜ਼ਾਰ ਜੋਖਮ ਦਾ ਵਿਸ਼ਲੇਸ਼ਣ ਕਰੋ",
        "Bengali": "বাজার ঝুঁকি বিশ্লেষণ করুন",
        "Urdu": "بازار کے خطرے کا تجزیہ کریں"
    }
}

def t(key):
    """Translate static UI phrases"""
    return translations.get(key, {}).get(language, key)

# Azure Translator API (dynamic text)
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")

def translate_text(text, target_lang):
    if not AZURE_TRANSLATOR_KEY or not AZURE_TRANSLATOR_ENDPOINT:
        return text  # fallback: English only
    path = '/translate?api-version=3.0'
    params = f"&to={target_lang}"
    constructed_url = AZURE_TRANSLATOR_ENDPOINT + path + params
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': 'centralindia',
        'Content-type': 'application/json'
    }
    body = [{"text": text}]
    request = requests.post(constructed_url, headers=headers, json=body)
    response = request.json()
    try:
        return response[0]['translations'][0]['text']
    except:
        return text

# ---------------- DEMO MODEL ----------------
# Fake dataset for demo
df = pd.DataFrame({
    "Previous_Year_High_Price": [3000, 5000, 7000],
    "Rainfall_Score": [1, 2, 3],
    "Total_Area_Sown": [1000, 2000, 3000]
})
X = df[["Previous_Year_High_Price", "Rainfall_Score"]]
y = df["Total_Area_Sown"]
model = LinearRegression().fit(X, y)

# ---------------- UI ----------------
st.sidebar.markdown("📝 **Farmer Input Panel**")
crop_name = st.sidebar.selectbox("Select Crop", ["Onion", "Wheat", "Soyabean"])
prev_price = st.sidebar.number_input("Last Year's High Price (₹/Quintal)", 1000, 10000, 5000)
rainfall = st.sidebar.select_slider("Rainfall Forecast", options=["Low", "Medium", "High"])
analyze_btn = st.sidebar.button(t("Analyze Market Risk"))

if analyze_btn:
    rain_mapping = {"High": 3, "Medium": 2, "Low": 1}
    rain_score = rain_mapping[rainfall]
    prediction = int(model.predict([[prev_price, rain_score]])[0])

    st.write(f"### 📊 {translate_text('Analysis Report for', LANGUAGES[language])} {crop_name}")
    st.metric(translate_text("Predicted Total Sowing", LANGUAGES[language]), f"{prediction} Hectares")

    if prediction > 2000:
        st.error(translate_text("🛑 Danger: Market Saturation Detected!", LANGUAGES[language]))
        st.write(translate_text("Oversupply will crash prices.", LANGUAGES[language]))
    else:
        st.success(translate_text("✅ Safe: Market looks stable.", LANGUAGES[language]))

import streamlit as st

st.set_page_config(
    page_title="CropBalance AI",
    page_icon="🚀",
    layout="wide"
)

import streamlit as st
import pandas as pd

# This MUST be the first Streamlit command
st.set_page_config(page_title="CropBalance AI", layout="wide")

# Now you can use other commands
st.title("Hello World")

try:
    st.set_page_config(layout="wide")
except st.errors.StreamlitAPIException:
    pass  # Already set elsewhere

# Language selection example
import streamlit as st
import requests
import os

# --- MUST BE FIRST ---
st.set_page_config(page_title="CropBalance AI", page_icon="🌾", layout="wide")

# --- Language Setup ---
LANGUAGES = {
    "English": "en", "Hindi": "hi", "Marathi": "mr", "Gujarati": "gu", "Punjabi": "pa",
    "Bengali": "bn", "Assamese": "as", "Odia": "or", "Kannada": "kn", "Malayalam": "ml",
    "Tamil": "ta", "Telugu": "te", "Urdu": "ur", "Konkani": "kok", "Sindhi": "sd",
    "Kashmiri": "ks", "Dogri": "doi", "Manipuri": "mni", "Bodo": "brx", "Santhali": "sat",
    "Maithili": "mai", "Nepali": "ne", "Sanskrit": "sa"
}
language = st.sidebar.selectbox("🌐 Choose Language", list(LANGUAGES.keys()))
lang_code = LANGUAGES[language]

# --- Azure Translator API (optional) ---
AZURE_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT")

def translate_text(text):
    """Translate any text into selected language"""
    if language == "English":  # no translation needed
        return text
    if not AZURE_KEY or not AZURE_ENDPOINT:
        return text  # fallback if API not set
    url = f"{AZURE_ENDPOINT}/translate?api-version=3.0&to={lang_code}"
    headers = {
        "Ocp-Apim-Subscription-Key": AZURE_KEY,
        "Ocp-Apim-Subscription-Region": "centralindia",
        "Content-Type": "application/json"
    }
    body = [{"text": text}]
    try:
        response = requests.post(url, headers=headers, json=body).json()
        return response[0]["translations"][0]["text"]
    except:
        return text

# --- Helper for translating entire UI ---
def T(text):
    return translate_text(text)

# --- Website Content ---
st.title(T("🌾 CropBalance AI"))
st.markdown(T("### 🚀 AI-Powered Market Intelligence for Indian Farmers"))

st.sidebar.header(T("🧑‍🌾 Farmer Input Panel"))
crop_name = st.sidebar.selectbox(T("Select Crop"), ["Onion", "Wheat", "Soyabean"])
prev_price = st.sidebar.number_input(T("Last Year's High Price (₹/Quintal)"), 1000, 10000, 5000)
rainfall = st.sidebar.select_slider(T("Rainfall Forecast"), options=[T("Low"), T("Medium"), T("High")])
analyze_btn = st.sidebar.button(T("Analyze Market Risk"))

if analyze_btn:
    st.subheader(T("📊 Analysis Report"))
    st.metric(T("Predicted Total Sowing"), f"{2000} Hectares")
    st.error(T("🛑 Danger: Market Saturation Detected!"))
    st.write(T("Oversupply will crash prices. Consider switching crops."))
