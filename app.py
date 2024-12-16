# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Custom HTML and CSS for Styling
custom_html = """
<div class="navbar">
    <div class="nav-item" onclick="navigate('Dashboard')">Dashboard</div>
    <div class="nav-item" onclick="navigate('Data Upload')">Data Upload</div>
    <div class="nav-item" onclick="navigate('Analysis')">Analysis</div>
    <div class="nav-item" onclick="navigate('Recommendations')">Recommendations</div>
</div>
<script>
    function navigate(page) {
        document.querySelector('select').value = page;
        const event = new Event('change');
        document.querySelector('select').dispatchEvent(event);
    }
</script>
<style>
    body {
        background-color: #f4f6f9;
        font-family: Arial, sans-serif;
    }
    .navbar {
        display: flex;
        justify-content: center;
        padding: 10px;
        background-color: #0073e6;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
    }
    .nav-item {
        margin: 0 20px;
        color: white;
        font-weight: bold;
        cursor: pointer;
        padding: 10px;
        border-radius: 5px;
    }
    .nav-item:hover {
        background-color: #005bb5;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        background-color: #0073e6;
        color: white;
        font-size: 16px;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
</style>
"""

st.markdown(custom_html, unsafe_allow_html=True)

# Initialize session state
if 'health_data' not in st.session_state:
    st.session_state.health_data = pd.DataFrame()

class HealthAnalytics:
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        
    def preprocess_data(self, data):
        scaler = StandardScaler()
        return scaler.fit_transform(data)
    
    def detect_anomalies(self, data):
        preprocessed_data = self.preprocess_data(data)
        predictions = self.anomaly_detector.fit_predict(preprocessed_data)
        return predictions

def generate_sample_data():
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = {
        'Date': dates,
        'Steps': np.random.randint(2000, 15000, 30),
        'HeartRate': np.random.randint(60, 100, 30),
        'SleepHours': np.random.uniform(5, 9, 30).round(2),
        'StressLevel': np.random.randint(1, 10, 30),
        'CaloriesBurned': np.random.randint(1500, 3000, 30),
        'WaterIntake': np.random.uniform(1, 4, 30).round(2)
    }
    return pd.DataFrame(data)

def main():
    st.title("🏥 AI-Powered Health Analytics Dashboard")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.selectbox("Choose a page", 
                               ["Dashboard", "Data Upload", "Analysis", "Recommendations"])
    
    if page == "Dashboard":
        show_dashboard()
    elif page == "Data Upload":
        show_data_upload()
    elif page == "Analysis":
        show_analysis()
    elif page == "Recommendations":
        show_recommendations()

def show_dashboard():
    st.header("Your Health Overview")
    
    if st.session_state.health_data.empty:
        st.session_state.health_data = generate_sample_data()
    
    # Create three columns for metrics
    col1, col2, col3 = st.columns(3)
    
    latest_data = st.session_state.health_data.iloc[-1]
    
    with col1:
        st.metric("Daily Steps", f"{latest_data['Steps']:,}", 
                 delta=f"{latest_data['Steps'] - 10000:,} from goal")
    with col2:
        st.metric("Sleep Hours", f"{latest_data['SleepHours']:.1f}", 
                 delta=f"{latest_data['SleepHours'] - 8:.1f} from recommended")
    with col3:
        st.metric("Stress Level", f"{latest_data['StressLevel']}/10", 
                 delta=f"{5 - latest_data['StressLevel']}")

    col1, col2 = st.columns(2)
    with col1:
        fig_steps = px.line(st.session_state.health_data, 
                           x='Date', y='Steps',
                           title='Steps Over Time')
        st.plotly_chart(fig_steps, use_container_width=True)
        
    with col2:
        fig_sleep = px.line(st.session_state.health_data, 
                           x='Date', y='SleepHours',
                           title='Sleep Pattern')
        st.plotly_chart(fig_sleep, use_container_width=True)

def show_data_upload():
    st.header("Upload Your Health Data")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type='csv')
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.health_data = data
            st.success("Data uploaded successfully!")
            st.dataframe(data)
        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")
    
    if st.button("Use Sample Data"):
        st.session_state.health_data = generate_sample_data()
        st.success("Sample data loaded successfully!")
        st.dataframe(st.session_state.health_data)

def show_analysis():
    st.header("Health Data Analysis")
    
    if not st.session_state.health_data.empty:
        analytics = HealthAnalytics()
        
        numerical_data = st.session_state.health_data[
            ['Steps', 'HeartRate', 'SleepHours', 'StressLevel']
        ]
        anomalies = analytics.detect_anomalies(numerical_data)
        
        fig = go.Figure()
        for column in numerical_data.columns:
            fig.add_trace(go.Scatter(
                x=st.session_state.health_data['Date'],
                y=st.session_state.health_data[column],
                name=column,
                mode='lines+markers',
                marker=dict(
                    color=['red' if a == -1 else 'blue' for a in anomalies]
                )
            ))
        
        fig.update_layout(title='Health Metrics with Anomaly Detection')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Please upload data or use sample data first!")

def show_recommendations():
    st.header("AI-Powered Health Recommendations")
    
    if not st.session_state.health_data.empty:
        latest_data = st.session_state.health_data.iloc[-1]
        recommendations = []
        
        if latest_data['Steps'] < 10000:
            recommendations.append(
                "🚶‍♂️ Your step count is below 10,000. Walk more!"
            )
        st.info("Recommendation: Your health metrics are improving!")
    else:
        st.warning("Please upload data first!")

if __name__ == "__main__":
    main()
