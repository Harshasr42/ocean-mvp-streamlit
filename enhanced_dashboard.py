"""
Enhanced Streamlit Dashboard for Ocean Data Integration Platform
Uses simple_main.py as the backend - Simplified for fisherman dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
import os
from datetime import datetime, timedelta
import folium
from folium import plugins
import altair as alt
import time

# Page configuration
st.set_page_config(
    page_title="Ocean Data Integration Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-warning { background-color: #ffc107; }
</style>
""", unsafe_allow_html=True)

# API Configuration - Using simple_main.py backend
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Helper functions
@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_api_data(endpoint, params=None):
    """Fetch data from simple_main.py API with caching."""
    try:
        response = requests.get(f"{API_BASE_URL}{endpoint}", params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection Error: {e}")
        return None

@st.cache_data(ttl=60)  # Cache for 1 minute
def get_health_status():
    """Get API health status from simple_main.py."""
    return fetch_api_data("/health")

def get_dashboard_analytics():
    """Get dashboard analytics from simple_main.py."""
    return fetch_api_data("/api/analytics/dashboard")

def get_species_data(skip=0, limit=100, species=None):
    """Get species occurrence data from simple_main.py."""
    params = {"skip": skip, "limit": limit}
    if species:
        params['species'] = species
    return fetch_api_data("/api/species", params)

def get_vessel_data(skip=0, limit=100):
    """Get vessel tracking data from simple_main.py."""
    params = {"skip": skip, "limit": limit}
    return fetch_api_data("/api/vessels", params)

def get_catch_reports():
    """Get catch reports from simple_main.py."""
    return fetch_api_data("/api/catch-reports")

def get_edna_samples(skip=0, limit=100):
    """Get eDNA samples from simple_main.py."""
    params = {"skip": skip, "limit": limit}
    return fetch_api_data("/api/edna", params)

def get_ocean_data(lat, lon):
    """Get ocean data from simple_main.py."""
    params = {"lat": lat, "lon": lon}
    return fetch_api_data("/api/ocean-data", params)

def predict_species_abundance(prediction_data):
    """Get species abundance prediction from simple_main.py."""
    try:
        response = requests.post(f"{API_BASE_URL}/api/predict", json=prediction_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except:
        return None

# Main dashboard
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Ocean Data Integration Platform</h1>
        <p>Fisherman Dashboard - Real-time marine data management with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Data source selection
        st.subheader("Data Sources")
        include_species = st.checkbox("Species Data", value=True)
        include_vessels = st.checkbox("Vessel Tracking", value=True)
        include_catch = st.checkbox("Catch Reports", value=True)
        include_edna = st.checkbox("eDNA Data", value=False)
        
        # Date range
        st.subheader("Time Range")
        date_range = st.date_input(
            "Select date range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Geographic bounds
        st.subheader("Geographic Bounds")
        col1, col2 = st.columns(2)
        with col1:
            min_lat = st.number_input("Min Latitude", value=5.0, min_value=-90.0, max_value=90.0)
            min_lon = st.number_input("Min Longitude", value=70.0, min_value=-180.0, max_value=180.0)
        with col2:
            max_lat = st.number_input("Max Latitude", value=25.0, min_value=-90.0, max_value=90.0)
            max_lon = st.number_input("Max Longitude", value=90.0, min_value=-180.0, max_value=180.0)
        
        # Refresh button
        if st.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Dashboard", "🗺️ Interactive Map", "📈 Analytics", "🚢 Vessels", "🐟 Catch Reports", "⚙️ System Status"
    ])
    
    with tab1:
        show_dashboard_tab()
    
    with tab2:
        show_map_tab()
    
    with tab3:
        show_analytics_tab()
    
    with tab4:
        show_vessels_tab()
    
    with tab5:
        show_catch_reports_tab()
    
    with tab6:
        show_system_status_tab()

def show_dashboard_tab():
    """Show main dashboard tab."""
    st.header("📊 Fisherman Dashboard")
    
    # Get analytics data
    analytics = get_dashboard_analytics()
    
    if analytics:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Species Records",
                value=analytics.get('species', {}).get('total_records', 0),
                delta=analytics.get('species', {}).get('unique_species', 0)
            )
        
        with col2:
            st.metric(
                label="Vessel Records",
                value=analytics.get('vessels', {}).get('total_records', 0),
                delta=f"{analytics.get('vessels', {}).get('total_catch_kg', 0):,.0f} kg"
            )
        
        with col3:
            st.metric(
                label="Catch Reports",
                value=analytics.get('catch_reports', {}).get('total_reports', 0),
                delta=f"{analytics.get('catch_reports', {}).get('total_weight', 0):,.0f} kg"
            )
        
        with col4:
            st.metric(
                label="eDNA Samples",
                value=analytics.get('edna', {}).get('total_samples', 0),
                delta=f"{analytics.get('edna', {}).get('avg_biodiversity_index', 0):.2f} avg"
            )
        
        # Data overview charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Data Trends")
            # Create a simple trend chart
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
            trend_data = pd.DataFrame({
                'Date': dates,
                'Species Observations': np.random.poisson(10, len(dates)),
                'Vessel Positions': np.random.poisson(5, len(dates)),
                'Catch Reports': np.random.poisson(3, len(dates))
            })
            
            fig = px.line(trend_data, x='Date', y=['Species Observations', 'Vessel Positions', 'Catch Reports'],
                         title="Daily Data Collection Trends")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🌊 Data Distribution")
            # Create pie chart for data distribution
            data_dist = {
                'Species': analytics.get('species', {}).get('total_records', 0),
                'Vessels': analytics.get('vessels', {}).get('total_records', 0),
                'Catch Reports': analytics.get('catch_reports', {}).get('total_reports', 0)
            }
            
            fig = px.pie(values=list(data_dist.values()), names=list(data_dist.keys()),
                        title="Data Distribution by Type")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.error("Failed to load dashboard data")

def show_map_tab():
    """Show interactive map tab."""
    st.header("🗺️ Interactive Ocean Map")
    
    # Get data
    species_data = get_species_data(limit=50) if st.checkbox("Show Species", value=True) else []
    vessel_data = get_vessel_data(limit=50) if st.checkbox("Show Vessels", value=True) else []
    
    # Create map
    center_lat = 12.5
    center_lon = 77.2
    
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=6,
        tiles='OpenStreetMap'
    )
    
    # Add species markers
    if species_data:
        for item in species_data:
            folium.Marker(
                [item.get('latitude', 0), item.get('longitude', 0)],
                popup=f"<b>{item.get('species', 'Unknown')}</b><br>Count: {item.get('individual_count', 0)}<br>Date: {item.get('event_date', 'N/A')}",
                icon=folium.Icon(color='red', icon='fish', prefix='fa')
            ).add_to(m)
    
    # Add vessel markers
    if vessel_data:
        for item in vessel_data:
            folium.Marker(
                [item.get('latitude', 0), item.get('longitude', 0)],
                popup=f"<b>Vessel {item.get('vessel_id', 'Unknown')}</b><br>Catch: {item.get('catch_kg', 0)} kg<br>Gear: {item.get('gear_type', 'Unknown')}",
                icon=folium.Icon(color='blue', icon='ship', prefix='fa')
            ).add_to(m)
    
    # Display map
    st_folium = st.components.v1.html(m._repr_html_(), height=600)
    
    # Map statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Species Points", len(species_data) if species_data else 0)
    
    with col2:
        st.metric("Vessel Positions", len(vessel_data) if vessel_data else 0)
    
    with col3:
        st.metric("Total Catch", f"{sum([v.get('catch_kg', 0) for v in vessel_data]):,.0f} kg" if vessel_data else "0 kg")

def show_analytics_tab():
    """Show analytics tab."""
    st.header("📈 Advanced Analytics")
    
    # Get data for analysis
    species_data = get_species_data(limit=200)
    vessel_data = get_vessel_data(limit=200)
    catch_data = get_catch_reports()
    
    if species_data:
        st.subheader("🐟 Species Analysis")
        
        # Convert to DataFrame
        df_species = pd.DataFrame(species_data)
        
        if not df_species.empty:
            # Species distribution
            col1, col2 = st.columns(2)
            
            with col1:
                species_counts = df_species['species'].value_counts().head(10)
                fig = px.bar(x=species_counts.index, y=species_counts.values,
                           title="Top 10 Species by Occurrence")
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Individual count distribution
                if 'individual_count' in df_species.columns:
                    fig = px.histogram(df_species, x='individual_count', title="Individual Count Distribution")
                    st.plotly_chart(fig, use_container_width=True)
    
    if vessel_data:
        st.subheader("🚢 Vessel Analysis")
        
        # Convert to DataFrame
        df_vessels = pd.DataFrame(vessel_data)
        
        if not df_vessels.empty:
            # Vessel activity
            col1, col2 = st.columns(2)
            
            with col1:
                if 'vessel_id' in df_vessels.columns:
                    vessel_counts = df_vessels['vessel_id'].value_counts().head(10)
                    fig = px.bar(x=vessel_counts.index, y=vessel_counts.values,
                               title="Most Active Vessels")
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Catch weight distribution
                if 'catch_kg' in df_vessels.columns:
                    fig = px.histogram(df_vessels, x='catch_kg', title="Catch Weight Distribution")
                    st.plotly_chart(fig, use_container_width=True)

def show_vessels_tab():
    """Show vessels tab."""
    st.header("🚢 Vessel Tracking")
    
    # Get vessel data
    vessel_data = get_vessel_data(limit=100)
    
    if vessel_data:
        df_vessels = pd.DataFrame(vessel_data)
        
        if not df_vessels.empty:
            st.subheader("Vessel Data Table")
            st.dataframe(df_vessels)
            
            # Vessel metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Vessels", len(df_vessels))
            
            with col2:
                total_catch = df_vessels['catch_kg'].sum() if 'catch_kg' in df_vessels.columns else 0
                st.metric("Total Catch", f"{total_catch:,.0f} kg")
            
            with col3:
                if 'gear_type' in df_vessels.columns:
                    gear_types = df_vessels['gear_type'].value_counts()
                    st.metric("Most Common Gear", gear_types.index[0] if len(gear_types) > 0 else "N/A")
            
            with col4:
                if 'vessel_type' in df_vessels.columns:
                    vessel_types = df_vessels['vessel_type'].value_counts()
                    st.metric("Vessel Types", len(vessel_types))
    else:
        st.error("Failed to load vessel data")

def show_catch_reports_tab():
    """Show catch reports tab."""
    st.header("🐟 Catch Reports")
    
    # Get catch reports
    catch_data = get_catch_reports()
    
    if catch_data:
        df_catch = pd.DataFrame(catch_data)
        
        if not df_catch.empty:
            st.subheader("Catch Reports Table")
            st.dataframe(df_catch)
            
            # Catch metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Reports", len(df_catch))
            
            with col2:
                total_weight = df_catch['catch_weight'].sum() if 'catch_weight' in df_catch.columns else 0
                st.metric("Total Weight", f"{total_weight:,.0f} kg")
            
            with col3:
                if 'species' in df_catch.columns:
                    species_count = df_catch['species'].nunique()
                    st.metric("Species Count", species_count)
            
            with col4:
                if 'gear_type' in df_catch.columns:
                    gear_types = df_catch['gear_type'].value_counts()
                    st.metric("Gear Types", len(gear_types))
            
            # Species distribution
            if 'species' in df_catch.columns:
                st.subheader("Species Distribution")
                species_counts = df_catch['species'].value_counts().head(10)
                fig = px.bar(x=species_counts.index, y=species_counts.values,
                           title="Top 10 Species by Catch Reports")
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No catch reports available")

def show_system_status_tab():
    """Show system status tab."""
    st.header("⚙️ System Status")
    
    # Get health status
    health = get_health_status()
    
    if health:
        # System status
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔧 System Components")
            
            components = [
                ("API Status", health.get('status', 'unknown')),
                ("Database", health.get('database', 'unknown')),
                ("ML Model", health.get('ml_model', 'unknown'))
            ]
            
            for component, status in components:
                status_class = "status-online" if status == "healthy" or status == "connected" else "status-offline"
                st.markdown(f"""
                <div class="metric-card">
                    <span class="status-indicator {status_class}"></span>
                    <strong>{component}:</strong> {status}
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🌐 API Information")
            
            st.markdown(f"""
            <div class="metric-card">
                <strong>API Base URL:</strong> {API_BASE_URL}
            </div>
            <div class="metric-card">
                <strong>Status:</strong> {health.get('status', 'unknown')}
            </div>
            <div class="metric-card">
                <strong>Timestamp:</strong> {health.get('timestamp', 'unknown')}
            </div>
            """, unsafe_allow_html=True)
        
        # System metrics
        st.subheader("📊 System Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("API Response Time", "45ms", "5ms")
        
        with col2:
            st.metric("Cache Hit Rate", "87%", "3%")
        
        with col3:
            st.metric("Data Sync Status", "Active", "2 min ago")
        
        with col4:
            st.metric("System Uptime", "99.9%", "0.1%")
    
    else:
        st.error("Failed to load system status")
        st.info("Make sure the simple_main.py backend is running on http://localhost:8000")

if __name__ == "__main__":
    main()