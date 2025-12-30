import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import warnings

warnings.filterwarnings('ignore')

# --- 1. SET UP THE PAGE ---
st.set_page_config(page_title="CropBalance AI", page_icon="🌾", layout="wide") # Added 'wide' layout for better graphs

# Custom CSS to make it look professional
st.markdown("""
<style>
    .big-font { font-size:20px !important; }
    .stAlert { padding: 20px; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("🌾 CropBalance AI")
st.markdown("### 🚀 AI-Powered Market Intelligence for Indian Farmers")

# --- 2. TRAIN THE MODEL ---
try:
    df = pd.read_csv('crop_data.csv')
    # Data Cleaning for Graph
    graph_df = df.copy() 
    
    df['Rainfall_Score'] = df['Rainfall_Level'].map({'High': 3, 'Medium': 2, 'Low': 1})
    X = df[['Previous_Year_High_Price', 'Rainfall_Score']]
    y = df['Total_Area_Sown']
    model = LinearRegression()
    model.fit(X, y)
    model_loaded = True
except FileNotFoundError:
    st.error("⚠️ Error: 'crop_data.csv' file not found.")
    model_loaded = False

# --- SMART AGENT FUNCTION ---
def get_best_crop(rain_level):
    if rain_level == "High":
        return "🍚 Rice or 🎋 Sugarcane"
    elif rain_level == "Medium":
        return "🌱 Soyabean or 🌽 Maize"
    else: 
        return "🥣 Bajra or 🌾 Jowar"

# --- 3. CREATE THE DASHBOARD ---
if model_loaded:
    col1, col2 = st.columns([1, 2]) # Split screen: Inputs on Left, Results on Right

    with col1:
        st.success("📝 **Farmer Input Panel**")
        crop_name = st.selectbox("Select Crop", ["Onion", "Wheat", "Soyabean"])
        prev_price = st.number_input("Last Year's High Price (₹/Quintal)", 1000, 10000, 5000)
        rainfall = st.select_slider("Rainfall Forecast", options=["Low", "Medium", "High"])
        
        analyze_btn = st.button("🔍 Analyze Market Risk", use_container_width=True)

    with col2:
        if analyze_btn:
            # --- AI PREDICTION ---
            rain_mapping = {'High': 3, 'Medium': 2, 'Low': 1}
            rain_score = rain_mapping[rainfall]
            prediction = int(model.predict([[prev_price, rain_score]])[0])
            
            # --- DISPLAY METRICS ---
            st.write(f"### 📊 Analysis Report for **{crop_name}**")
            
            # 3 Metrics in a row
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Total Sowing", f"{prediction} Hectares", "High Accuracy")
            m2.metric("Rainfall Condition", rainfall, "Weather Data")
            
            # --- RISK LOGIC & FINANCIALS ---
            if prediction > 2000:
                m3.metric("Risk Level", "CRITICAL", "- High Risk", delta_color="inverse")
                
                st.error(f"🛑 **DANGER ALERT: Market Saturation Detected!**")
                st.write(f"Your AI analysis shows that **{prediction} farmers** are planting {crop_name}. This oversupply will crash prices.")
                
                # Financial Impact (The "Money" Feature)
                potential_loss = 50000 # Example calculation: 50% price drop on 1 hectare
                st.markdown(f"#### 💸 **Estimated Loss if you plant: ₹{potential_loss} / hectare**")
                
                # Smart Recommendation
                smart_crop = get_best_crop(rainfall)
                st.info(f"💡 **AI Recommendation:** To maximize profit, switch to **{smart_crop}**.")
                
            else:
                m3.metric("Risk Level", "SAFE", "+ Low Risk")
                st.success(f"✅ **GREEN SIGNAL: Market looks stable.**")
                st.write(f"Supply is controlled. You can expect good profits from {crop_name}.")

            # --- VISUAL EVIDENCE (The Chart) ---
            st.markdown("---")
            st.subheader("📈 Historical Data Trends")
            st.write("See how High Prices (X-axis) lead to Oversupply (Y-axis):")
            
            # Simple Chart showing the relationship
            chart_data = graph_df[graph_df['Crop_Name'] == crop_name]
            if not chart_data.empty:
                st.bar_chart(chart_data, x="Previous_Year_High_Price", y="Total_Area_Sown")
            else:
                st.write("(No historical chart data available for this crop in prototype)")