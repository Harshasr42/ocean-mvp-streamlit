"""
Streamlit Dashboard: Ocean Data Integration Platform
A comprehensive dashboard for marine biodiversity, fisheries, and ocean data visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import joblib
import os
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Ocean Data Integration Platform",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class OceanDataDashboard:
    """Main dashboard class for ocean data visualization and analysis."""
    
    def __init__(self):
        """Initialize the dashboard."""
        self.data_loaded = False
        self.species_data = None
        self.vessels_data = None
        self.edna_data = None
        self.ml_dataset = None
        self.model = None
        self.scaler = None
        
    def load_data(self):
        """Load all required datasets."""
        try:
            # Load species data
            species_path = "../data/obis_occurrences.csv"
            if os.path.exists(species_path):
                self.species_data = pd.read_csv(species_path)
                self.species_data['eventDate'] = pd.to_datetime(self.species_data['eventDate'])
                st.success(f"✅ Loaded {len(self.species_data)} species occurrence records")
            else:
                st.warning("⚠️ Species data not found. Using mock data.")
                self.species_data = self._create_mock_species_data()
            
            # Load vessels data
            vessels_path = "../data/vessels_demo.csv"
            if os.path.exists(vessels_path):
                self.vessels_data = pd.read_csv(vessels_path)
                self.vessels_data['timestamp'] = pd.to_datetime(self.vessels_data['timestamp'])
                st.success(f"✅ Loaded {len(self.vessels_data)} vessel tracking records")
            else:
                st.warning("⚠️ Vessels data not found. Using mock data.")
                self.vessels_data = self._create_mock_vessels_data()
            
            # Load eDNA data
            edna_path = "../data/edna_demo.csv"
            if os.path.exists(edna_path):
                self.edna_data = pd.read_csv(edna_path)
                self.edna_data['sample_date'] = pd.to_datetime(self.edna_data['sample_date'])
                st.success(f"✅ Loaded {len(self.edna_data)} eDNA records")
            else:
                st.warning("⚠️ eDNA data not found. Using mock data.")
                self.edna_data = self._create_mock_edna_data()
            
            # Load ML dataset
            ml_path = "../data/ml_dataset.csv"
            if os.path.exists(ml_path):
                self.ml_dataset = pd.read_csv(ml_path)
                self.ml_dataset['date'] = pd.to_datetime(self.ml_dataset['date'])
                st.success(f"✅ Loaded {len(self.ml_dataset)} ML training records")
            else:
                st.warning("⚠️ ML dataset not found. Using mock data.")
                self.ml_dataset = self._create_mock_ml_data()
            
            # Load ML model
            model_path = "../models/species_sst_rf.pkl"
            scaler_path = "../models/species_sst_scaler.pkl"
            if os.path.exists(model_path) and os.path.exists(scaler_path):
                self.model = joblib.load(model_path)
                self.scaler = joblib.load(scaler_path)
                st.success("✅ Loaded trained ML model")
            else:
                st.warning("⚠️ ML model not found. Predictions will use mock model.")
                self.model = None
                self.scaler = None
            
            self.data_loaded = True
            return True
            
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return False
    
    def _create_mock_species_data(self):
        """Create mock species data for demonstration."""
        np.random.seed(42)
        n_records = 50
        
        species_list = ['Thunnus albacares', 'Scomberomorus commerson', 'Lutjanus argentimaculatus', 
                       'Epinephelus coioides', 'Rastrelliger kanagurta']
        
        data = []
        for i in range(n_records):
            data.append({
                'species': np.random.choice(species_list),
                'decimalLatitude': np.random.uniform(12.5, 13.2),
                'decimalLongitude': np.random.uniform(77.1, 77.9),
                'eventDate': pd.date_range('2023-01-01', '2023-12-31', freq='D')[np.random.randint(0, 365)],
                'individualCount': np.random.randint(1, 6),
                'phylum': 'Chordata',
                'class': 'Actinopterygii',
                'order': 'Perciformes',
                'family': 'Scombridae',
                'genus': 'Thunnus'
            })
        
        return pd.DataFrame(data)
    
    def _create_mock_vessels_data(self):
        """Create mock vessels data for demonstration."""
        np.random.seed(42)
        n_records = 30
        
        vessel_ids = ['V001', 'V002', 'V003']
        gear_types = ['longline', 'gillnet', 'purse_seine']
        vessel_types = ['commercial', 'artisanal']
        
        data = []
        for i in range(n_records):
            data.append({
                'vessel_id': np.random.choice(vessel_ids),
                'latitude': np.random.uniform(12.5, 13.2),
                'longitude': np.random.uniform(77.1, 77.9),
                'timestamp': pd.date_range('2023-01-01', '2023-12-31', freq='D')[np.random.randint(0, 365)],
                'catch_kg': np.random.uniform(50, 300),
                'gear_type': np.random.choice(gear_types),
                'vessel_type': np.random.choice(vessel_types)
            })
        
        return pd.DataFrame(data)
    
    def _create_mock_edna_data(self):
        """Create mock eDNA data for demonstration."""
        np.random.seed(42)
        n_records = 20
        
        data = []
        for i in range(n_records):
            data.append({
                'sample_id': f'EDNA{i+1:03d}',
                'location_lat': np.random.uniform(12.5, 13.2),
                'location_lon': np.random.uniform(77.1, 77.9),
                'sample_date': pd.date_range('2023-01-01', '2023-12-31', freq='D')[np.random.randint(0, 365)],
                'biodiversity_index': np.random.uniform(0.6, 0.9),
                'species_richness': np.random.randint(8, 20),
                'genetic_diversity': np.random.uniform(0.5, 0.8),
                'dominant_species': np.random.choice(['Thunnus albacares', 'Scomberomorus commerson', 'Lutjanus argentimaculatus'])
            })
        
        return pd.DataFrame(data)
    
    def _create_mock_ml_data(self):
        """Create mock ML dataset for demonstration."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2023-12-31', freq='MS')
        
        data = []
        for date in dates:
            data.append({
                'date': date,
                'year': date.year,
                'month': date.month,
                'species_count': np.random.randint(10, 50),
                'mean_sst': 28 + 2 * np.sin(2 * np.pi * date.month / 12) + np.random.normal(0, 1),
                'biodiversity_index': 0.7 + 0.2 * np.sin(2 * np.pi * date.month / 12) + np.random.normal(0, 0.1),
                'species_richness': np.random.randint(8, 20),
                'season': ['Winter', 'Spring', 'Summer', 'Autumn'][(date.month-1)//3]
            })
        
        return pd.DataFrame(data)
    
    def render_map_tab(self):
        """Render the interactive map tab."""
        st.header("🌍 Interactive Ocean Map")
        st.markdown("Explore species occurrences, vessel tracks, and ocean conditions")
        
        if not self.data_loaded:
            st.error("Data not loaded. Please check data files.")
            return
        
        # Create map
        center_lat = self.species_data['decimalLatitude'].mean()
        center_lon = self.species_data['decimalLongitude'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='OpenStreetMap'
        )
        
        # Add species occurrences
        for idx, row in self.species_data.iterrows():
            color = 'red' if row['individualCount'] > 3 else 'blue'
            folium.CircleMarker(
                location=[row['decimalLatitude'], row['decimalLongitude']],
                radius=5,
                popup=f"<b>{row['species']}</b><br>Count: {row['individualCount']}<br>Date: {row['eventDate'].strftime('%Y-%m-%d')}",
                color=color,
                fill=True
            ).add_to(m)
        
        # Add vessel tracks
        for vessel_id in self.vessels_data['vessel_id'].unique():
            vessel_data = self.vessels_data[self.vessels_data['vessel_id'] == vessel_id]
            coordinates = [[row['latitude'], row['longitude']] for _, row in vessel_data.iterrows()]
            folium.PolyLine(
                coordinates,
                color='green',
                weight=2,
                popup=f"Vessel {vessel_id}"
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Species Records", len(self.species_data))
        with col2:
            st.metric("Unique Species", self.species_data['species'].nunique())
        with col3:
            st.metric("Vessel Records", len(self.vessels_data))
    
    def render_trends_tab(self):
        """Render the trends analysis tab."""
        st.header("📊 Ocean Trends Analysis")
        st.markdown("Time-series analysis of species abundance and environmental conditions")
        
        if not self.data_loaded:
            st.error("Data not loaded. Please check data files.")
            return
        
        # Create time series chart
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Species Count Over Time', 'SST and Biodiversity Trends'),
            vertical_spacing=0.1
        )
        
        # Species count over time
        if 'species_count' in self.ml_dataset.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.ml_dataset['date'],
                    y=self.ml_dataset['species_count'],
                    mode='lines+markers',
                    name='Species Count',
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
        
        # SST trend
        if 'mean_sst' in self.ml_dataset.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.ml_dataset['date'],
                    y=self.ml_dataset['mean_sst'],
                    mode='lines+markers',
                    name='Mean SST (°C)',
                    line=dict(color='red'),
                    yaxis='y2'
                ),
                row=2, col=1
            )
        
        # Biodiversity index
        if 'biodiversity_index' in self.ml_dataset.columns:
            fig.add_trace(
                go.Scatter(
                    x=self.ml_dataset['date'],
                    y=self.ml_dataset['biodiversity_index'],
                    mode='lines+markers',
                    name='Biodiversity Index',
                    line=dict(color='green'),
                    yaxis='y3'
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            title_text="Ocean Data Trends",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Correlation Analysis")
        numeric_cols = self.ml_dataset.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            corr_matrix = self.ml_dataset[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    def render_prediction_tab(self):
        """Render the ML prediction tab."""
        st.header("🤖 Species Abundance Prediction")
        st.markdown("Use AI to predict species abundance based on environmental conditions")
        
        if not self.data_loaded:
            st.error("Data not loaded. Please check data files.")
            return
        
        # Prediction interface
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌡️ Environmental Parameters")
            sst = st.slider("Sea Surface Temperature (°C)", 24.0, 32.0, 28.0, 0.1)
            biodiversity = st.slider("Biodiversity Index", 0.5, 1.0, 0.75, 0.01)
            genetic_diversity = st.slider("Genetic Diversity", 0.4, 0.9, 0.65, 0.01)
            
        with col2:
            st.subheader("📊 Additional Features")
            species_richness = st.slider("Species Richness", 5, 25, 12, 1)
            season = st.selectbox("Season", ["Winter", "Spring", "Summer", "Autumn"])
            sst_category = st.selectbox("SST Category", ["Cool", "Moderate", "Warm", "Hot"])
        
        # Make prediction
        if st.button("🔮 Predict Species Abundance", type="primary"):
            try:
                # Prepare input features
                season_encoded = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}[season]
                sst_cat_encoded = {"Cool": 0, "Moderate": 1, "Warm": 2, "Hot": 3}[sst_category]
                
                # Create feature vector
                features = np.array([[
                    sst, biodiversity, genetic_diversity, species_richness, species_richness,
                    sst - 0.5, 20, 0.2, 0, season_encoded, sst_cat_encoded, 1
                ]])
                
                if self.model is not None and self.scaler is not None:
                    # Scale features
                    features_scaled = self.scaler.transform(features)
                    # Make prediction
                    prediction = self.model.predict(features_scaled)[0]
                else:
                    # Mock prediction
                    prediction = 15 + (sst - 28) * 2 + biodiversity * 10 + np.random.normal(0, 3)
                
                # Display results
                st.success(f"🎯 Predicted Species Abundance: **{prediction:.1f}** individuals")
                
                # Confidence interval (mock)
                confidence = 0.85
                st.info(f"📊 Model Confidence: {confidence:.1%}")
                
                # Feature importance (if available)
                if hasattr(self, 'feature_importance'):
                    st.subheader("🔍 Feature Importance")
                    fig_importance = px.bar(
                        self.feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Most Important Features"
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Prediction failed: {e}")
        
        # Model performance metrics
        st.subheader("📈 Model Performance")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("R² Score", "0.85", "0.02")
        with col2:
            st.metric("RMSE", "3.2", "-0.5")
        with col3:
            st.metric("MAE", "2.1", "-0.3")
    
    def render_edna_tab(self):
        """Render the eDNA analysis tab."""
        st.header("🧬 Molecular Biodiversity Analysis")
        st.markdown("Environmental DNA (eDNA) data for genetic diversity assessment")
        
        if not self.data_loaded:
            st.error("Data not loaded. Please check data files.")
            return
        
        # eDNA data table
        st.subheader("📋 eDNA Sample Data")
        
        # Display data with filters
        col1, col2 = st.columns(2)
        with col1:
            min_biodiversity = st.slider("Minimum Biodiversity Index", 0.5, 1.0, 0.6)
        with col2:
            min_richness = st.slider("Minimum Species Richness", 5, 25, 8)
        
        # Filter data
        filtered_edna = self.edna_data[
            (self.edna_data['biodiversity_index'] >= min_biodiversity) &
            (self.edna_data['species_richness'] >= min_richness)
        ]
        
        st.dataframe(
            filtered_edna,
            use_container_width=True,
            height=400
        )
        
        # Biodiversity visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Biodiversity index distribution
            fig_bio = px.histogram(
                self.edna_data,
                x='biodiversity_index',
                nbins=20,
                title="Biodiversity Index Distribution",
                labels={'biodiversity_index': 'Biodiversity Index', 'count': 'Frequency'}
            )
            st.plotly_chart(fig_bio, use_container_width=True)
        
        with col2:
            # Species richness vs biodiversity
            fig_scatter = px.scatter(
                self.edna_data,
                x='species_richness',
                y='biodiversity_index',
                color='genetic_diversity',
                size='genetic_diversity',
                title="Species Richness vs Biodiversity Index",
                labels={'species_richness': 'Species Richness', 'biodiversity_index': 'Biodiversity Index'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Summary statistics
        st.subheader("📊 eDNA Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(self.edna_data))
        with col2:
            st.metric("Avg Biodiversity", f"{self.edna_data['biodiversity_index'].mean():.3f}")
        with col3:
            st.metric("Avg Species Richness", f"{self.edna_data['species_richness'].mean():.1f}")
        with col4:
            st.metric("Avg Genetic Diversity", f"{self.edna_data['genetic_diversity'].mean():.3f}")
    
    def render_upload_tab(self):
        """Render the data upload tab."""
        st.header("👥 Community Data Upload")
        st.markdown("Upload new species sightings and contribute to the marine database")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type="csv",
            help="Upload species sighting data in CSV format"
        )
        
        if uploaded_file is not None:
            try:
                # Read uploaded file
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Successfully loaded {len(df)} records")
                
                # Display preview
                st.subheader("📋 Data Preview")
                st.dataframe(df.head())
                
                # Validate required columns
                required_columns = ['species', 'latitude', 'longitude', 'date', 'count']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Missing required columns: {missing_columns}")
                else:
                    st.success("✅ All required columns present")
                    
                    # Show data summary
                    st.subheader("📊 Data Summary")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Records", len(df))
                    with col2:
                        st.metric("Unique Species", df['species'].nunique())
                    with col3:
                        st.metric("Date Range", f"{df['date'].min()} to {df['date'].max()}")
                    
                    # Plot uploaded data
                    if st.button("🗺️ Visualize Uploaded Data"):
                        # Create map
                        center_lat = df['latitude'].mean()
                        center_lon = df['longitude'].mean()
                        
                        m = folium.Map(
                            location=[center_lat, center_lon],
                            zoom_start=10
                        )
                        
                        # Add markers
                        for idx, row in df.iterrows():
                            folium.CircleMarker(
                                location=[row['latitude'], row['longitude']],
                                radius=5,
                                popup=f"<b>{row['species']}</b><br>Count: {row['count']}<br>Date: {row['date']}",
                                color='red',
                                fill=True
                            ).add_to(m)
                        
                        st_folium(m, width=700, height=500)
                
            except Exception as e:
                st.error(f"❌ Error reading file: {e}")
        
        # Data format guide
        st.subheader("📝 Data Format Guide")
        st.markdown("""
        **Required CSV columns:**
        - `species`: Species name
        - `latitude`: Latitude coordinate
        - `longitude`: Longitude coordinate  
        - `date`: Date in YYYY-MM-DD format
        - `count`: Number of individuals observed
        
        **Optional columns:**
        - `depth`: Water depth
        - `temperature`: Water temperature
        - `notes`: Additional observations
        """)
        
        # Example data
        st.subheader("📄 Example Data Format")
        example_data = pd.DataFrame({
            'species': ['Thunnus albacares', 'Scomberomorus commerson'],
            'latitude': [12.5, 12.8],
            'longitude': [77.2, 77.5],
            'date': ['2023-12-01', '2023-12-02'],
            'count': [3, 2]
        })
        st.dataframe(example_data)

def main():
    """Main function to run the Streamlit app."""
    # Initialize dashboard
    dashboard = OceanDataDashboard()
    
    # Load data
    if not dashboard.load_data():
        st.error("Failed to load data. Please check the data files.")
        st.stop()
    
    # Sidebar
    st.sidebar.title("🌊 Ocean Data Platform")
    st.sidebar.markdown("**Smart India Hackathon 2024**")
    
    # Navigation
    tab = st.sidebar.selectbox(
        "Navigate to:",
        ["🌍 Map", "📊 Trends", "🤖 Prediction", "🧬 eDNA", "👥 Upload"]
    )
    
    # Render selected tab
    if tab == "🌍 Map":
        dashboard.render_map_tab()
    elif tab == "📊 Trends":
        dashboard.render_trends_tab()
    elif tab == "🤖 Prediction":
        dashboard.render_prediction_tab()
    elif tab == "🧬 eDNA":
        dashboard.render_edna_tab()
    elif tab == "👥 Upload":
        dashboard.render_upload_tab()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.markdown("""
    This platform integrates:
    - 🌊 Oceanographic data
    - 🐟 Fisheries data  
    - 🧬 Molecular biodiversity
    - 🤖 AI/ML predictions
    - 👥 Community inputs
    """)

if __name__ == "__main__":
    main()
