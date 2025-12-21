import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from huggingface_hub import hf_hub_download

# --- CONFIGURATION ---
REPO_ID = "divyabuni/tourism-package-model"

@st.cache_resource
def load_assets():
    # Download the trained model
    model_path = hf_hub_download(repo_id=REPO_ID, filename="model.pkl")
    model = joblib.load(model_path)
    # Download the feature list for column alignment
    feat_path = hf_hub_download(repo_id=REPO_ID, filename="features.pkl")
    features = joblib.load(feat_path)
    return model, features

model, model_features = load_assets()

# --- UI SETUP ---
st.set_page_config(page_title="Wellness Tourism Predictor", layout="wide", page_icon="🌴")
st.title("🌴 Visit with Us: Wellness Package Predictor")
st.markdown("### Decision Support System for Holiday Package Sales")

# --- MAPPINGS ---
gender_map = {"Female": 0, "Male": 1}
contact_map = {"Company Invited": 0, "Self Enquiry": 1}
marital_map = {"Divorced": 0, "Married": 1, "Single": 2}
occ_map = {"Free Lancer": 0, "Large Business": 1, "Salaried": 2, "Small Business": 3}
dest_map = {"Associate": 0, "Executive": 1, "Manager": 2, "Senior Manager": 3, "VP": 4}

# --- INPUT FORM ---
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", 18, 70, 30)
    gender_label = st.selectbox("Gender", list(gender_map.keys()))
    marital_label = st.selectbox("Marital Status", list(marital_map.keys()))
    income = st.number_input("Monthly Income", 1000, 100000, 25000)
with col2:
    st.subheader("Engagement")
    contact_label = st.selectbox("Type of Contact", list(contact_map.keys()))
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    pitch_dur = st.number_input("Duration of Pitch (min)", 5, 120, 15)
    trips = st.number_input("Number of Past Trips", 1, 20, 3)
with col3:
    st.subheader("Professional Info")
    occ_label = st.selectbox("Occupation", list(occ_map.keys()))
    dest_label = st.selectbox("Designation", list(dest_map.keys()))
    passport = st.selectbox("Has Passport?", ["No", "Yes"])
    own_car = st.selectbox("Owns Car?", ["No", "Yes"])

# --- DATA PROCESSING ---
input_data = {
    'Age': age, 'TypeofContact': contact_map[contact_label], 'CityTier': city_tier,
    'DurationOfPitch': pitch_dur, 'Occupation': occ_map[occ_label], 'Gender': gender_map[gender_label],
    'NumberOfPersonVisiting': 2, 'NumberOfFollowups': 3, 'ProductPitched': 1,
    'PreferredPropertyStar': 3, 'MaritalStatus': marital_map[marital_label], 'NumberOfTrips': trips,
    'Passport': 1 if passport == "Yes" else 0, 'PitchSatisfactionScore': 3,
    'OwnCar': 1 if own_car == "Yes" else 0, 'NumberOfChildrenVisiting': 1,
    'Designation': dest_map[dest_label], 'MonthlyIncome': income
}
# Reorder DataFrame to match model_features exactly
input_df = pd.DataFrame([input_data])[model_features]

# --- PREDICTION & VISUALIZATION ---
st.divider()
if st.button("Generate Prediction", type="primary"):
    # Get Probability
    prob = model.predict_proba(input_df)[0][1]
    
    # 1. VISUAL GAUGE CHART
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = prob * 100,
        title = {'text': "Purchase Probability (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps' : [
                {'range': [0, 50], 'color': "#FFCCCB"}, # Light Red
                {'range': [50, 100], 'color': "#90EE90"} # Light Green
            ],
            'threshold' : {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50
            }
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # 2. STATUS MESSAGE
    if prob > 0.5:
        st.success(f"📈 **High Potential:** Likely to purchase. (Confidence: {prob:.2%})")
    else:
        st.warning(f"📉 **Low Potential:** Unlikely to purchase. (Confidence: {1-prob:.2%})")

    # 3. FEATURE IMPORTANCE CHART
    st.subheader("Why this prediction?")
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': model_features, 'Importance': importances})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=True).tail(10)
    
    fig_bar = px.bar(feat_imp_df, x='Importance', y='Feature', orientation='h', 
                     title="Top 10 Factors Influencing This Prediction",
                     color_continuous_scale='Viridis', color='Importance')
    st.plotly_chart(fig_bar, use_container_width=True)
