# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import os

# Set page configuration
st.set_page_config(
    page_title="Health Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
    }
    .reportview-container {
        background: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

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
    
    # Create graphs
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
        # Initialize HealthAnalytics
        analytics = HealthAnalytics()
        
        # Prepare data for anomaly detection
        numerical_data = st.session_state.health_data[
            ['Steps', 'HeartRate', 'SleepHours', 'StressLevel']
        ]
        
        # Detect anomalies
        anomalies = analytics.detect_anomalies(numerical_data)
        
        # Create visualization
        fig = go.Figure()
        
        # Add traces for each health metric
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
        
        fig.update_layout(title='Health Metrics Overview with Anomaly Detection')
        st.plotly_chart(fig, use_container_width=True)
        
        # Show correlation matrix
        st.subheader("Correlation Analysis")
        corr_matrix = numerical_data.corr()
        fig_corr = px.imshow(corr_matrix, 
                            title='Correlation Matrix of Health Metrics')
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.warning("Please upload data or use sample data first!")

def show_recommendations():
    st.header("AI-Powered Health Recommendations")
    
    if not st.session_state.health_data.empty:
        latest_data = st.session_state.health_data.iloc[-1]
        
        # Generate recommendations based on the latest data
        recommendations = []
        
        if latest_data['Steps'] < 10000:
            recommendations.append(
                "🚶‍♂️ Your step count is below the recommended 10,000 steps. "
                "Try to incorporate more walking into your daily routine."
            )
            
        if latest_data['SleepHours'] < 7:
            recommendations.append(
                "😴 You're getting less than 7 hours of sleep. "
                "Consider adjusting your sleep schedule for better rest."
            )
            
        if latest_data['StressLevel'] > 7:
            recommendations.append(
                "🧘‍♂️ Your stress levels are high. "
                "Consider practicing meditation or deep breathing exercises."
            )
            
        if latest_data['WaterIntake'] < 2:
            recommendations.append(
                "💧 Your water intake is below recommended levels. "
                "Try to drink at least 2 liters of water daily."
            )
        
        # Display recommendations
        for i, rec in enumerate(recommendations, 1):
            st.info(f"Recommendation {i}: {rec}")
            
        if not recommendations:
            st.success("Great job! Your health metrics are within recommended ranges.")
            
        # Display weekly summary
        st.subheader("Weekly Summary")
        weekly_data = st.session_state.health_data.tail(7).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Steps", f"{weekly_data['Steps']:,.0f}")
            st.metric("Average Sleep", f"{weekly_data['SleepHours']:.1f} hours")
            
        with col2:
            st.metric("Average Stress", f"{weekly_data['StressLevel']:.1f}/10")
            st.metric("Average Calories", f"{weekly_data['CaloriesBurned']:,.0f}")
    else:
        st.warning("Please upload data or use sample data first!")

if __name__ == "__main__":
    main()