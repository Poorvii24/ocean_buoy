import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
import pydeck as pdk

# ==========================================
# 0. CONFIG & STYLING
# ==========================================
st.set_page_config(
    page_title="Intelligent Shield | Global Monitor",
    layout="wide",
    page_icon="🌍"
)

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .stMetric {
        background-color: #262730;
        border: 1px solid #464B5C;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. BACKEND: DATA & AI LOGIC
# ==========================================

def generate_ocean_data(n_points=1000, noise_level=0.1):
    """Generates synthetic ocean wave data"""
    t = np.linspace(0, 50, n_points)
    data = np.sin(t) + 0.5 * np.sin(t * 0.5)
    noise = np.random.normal(0, noise_level, n_points)
    return data + noise

def inject_anomalies(data, anomaly_type):
    """Injects specific failures"""
    modified_data = data.copy()
    n = len(data)
    start_idx = int(n * 0.85)
    end_idx = int(n * 0.95)

    if anomaly_type == "Spike (Sudden Jump)":
        modified_data[start_idx] = modified_data[start_idx] + 4.0 
    elif anomaly_type == "Drift (Sensor Aging)":
        drift = np.linspace(0, 2, end_idx - start_idx)
        modified_data[start_idx:end_idx] += drift
    elif anomaly_type == "Flatline (Dead Sensor)":
        modified_data[start_idx:end_idx] = modified_data[start_idx]
    
    return modified_data, start_idx, end_idx

def create_sequences(data, time_steps=30):
    """Converts stream into windows of 30 steps"""
    output = []
    for i in range(len(data) - time_steps):
        output.append(data[i : (i + time_steps)])
    return np.stack(output)

# ==========================================
# 2. FRONTEND: DASHBOARD LAYOUT
# ==========================================

# --- Sidebar ---
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/earth-planet.png", width=80)
    st.header("Global Command")
    
    st.subheader("1. Simulation Settings")
    anomaly_choice = st.selectbox(
        "Inject Scenario:",
        ["None (Healthy)", "Spike (Sudden Jump)", "Drift (Sensor Aging)", "Flatline (Dead Sensor)"]
    )
    noise_lvl = st.slider("Sea State Noise", 0.0, 0.4, 0.1)
    
    st.info("""
    **Legend:**
    🟢 Operational Network
    🔴 ANOMALY DETECTED
    """)

# --- Title ---
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title("🌍 Intelligent Shield")
    st.markdown("**Global Ocean Sensor Network (GOSN)**")
with col_status:
    if anomaly_choice == "None (Healthy)":
        st.success("GLOBAL STATUS: OPTIMAL")
    else:
        st.error(f"ALERT: {anomaly_choice.upper()}")

# --- Data Gen ---
raw_data = generate_ocean_data(n_points=1200, noise_level=noise_lvl)
display_data, s_idx, e_idx = inject_anomalies(raw_data, anomaly_choice)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(display_data.reshape(-1, 1))

# --- MAP SECTION (GLOBAL UPGRADE) ---
st.markdown("### 1. Live Global Map")

# 1. Generate 300 Buoys across major oceans (approximate coords)
np.random.seed(101) # Fixed seed

# Pacific Cluster
lat_pac = np.random.uniform(-40, 50, 120)
lon_pac = np.random.uniform(-180, -100, 120)

# Atlantic Cluster
lat_atl = np.random.uniform(-40, 60, 100)
lon_atl = np.random.uniform(-60, 0, 100)

# Indian Ocean Cluster
lat_ind = np.random.uniform(-30, 20, 80)
lon_ind = np.random.uniform(50, 100, 80)

# Combine them all
all_lats = np.concatenate([lat_pac, lat_atl, lat_ind])
all_lons = np.concatenate([lon_pac, lon_atl, lon_ind])
n_buoys = len(all_lats)

status = ["Healthy"] * n_buoys
colors = [[0, 255, 128, 140]] * n_buoys # Teal Green, semi-transparent
sizes = [80000] * n_buoys # Standard size

# 2. Assign the "Target Buoy" (Index 0 in Pacific)
if anomaly_choice != "None (Healthy)":
    # If anomaly exists, make Buoy #0 RED and HUGE
    status[0] = f"CRITICAL: {anomaly_choice}"
    colors[0] = [255, 0, 0, 255] # Bright Red
    sizes[0] = 500000 # Massive size so it stands out on world map
    target_name = "PACIFIC-BUOY-001"
else:
    status[0] = "Monitored (Healthy)"
    colors[0] = [0, 255, 0, 255] # Bright Green
    sizes[0] = 150000 
    target_name = "PACIFIC-BUOY-001"

# 3. Create DataFrame
map_df = pd.DataFrame({
    "lat": all_lats,
    "lon": all_lons,
    "status": status,
    "color": colors,
    "size": sizes
})

# 4. Render Map with PyDeck
layer = pdk.Layer(
    "ScatterplotLayer",
    map_df,
    get_position='[lon, lat]',
    get_color='color',
    get_radius='size',
    pickable=True, 
    auto_highlight=True
)

# Zoomed out view to see the whole world
view_state = pdk.ViewState(latitude=0, longitude=-120, zoom=0.8, pitch=0)

tooltip = {
    "html": "<b>Status:</b> {status}",
    "style": {"backgroundColor": "black", "color": "white"}
}

st.pydeck_chart(pdk.Deck(
    layers=[layer], 
    initial_view_state=view_state,
    tooltip=tooltip,
    map_style=None
))

# --- LIVE CHART SECTION ---
st.markdown(f"### 2. Live Telemetry: {target_name}")
col_chart, col_raw = st.columns([3, 1])
with col_chart:
    fig_raw = go.Figure()
    fig_raw.add_trace(go.Scatter(y=display_data[-200:], mode='lines', name='Wave Height', line=dict(color='#00CC96')))
    fig_raw.update_layout(height=250, margin=dict(l=0, r=0, t=0, b=0), template="plotly_dark", title="Incoming Data Stream")
    st.plotly_chart(fig_raw, use_container_width=True)
with col_raw:
    st.info("The AI monitors this specific buoy. If the pattern breaks the laws of physics, it triggers the Global Alert System.")

# --- AI ANALYSIS SECTION ---
st.markdown("### 3. AI Diagnostics & Detection")

if st.button("🛡️ Run Global Diagnostics", type="primary"):
    
    with st.spinner("Analyzing Sensor Data Stream..."):
        TIME_STEPS = 30
        sequences = create_sequences(scaled_data, TIME_STEPS)
        
        # Split Data
        train_size = int(len(sequences) * 0.6)
        X_train = sequences[:train_size]
        X_test = sequences[train_size:]
        
        # Flatten
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        X_test_flat = X_test.reshape(X_test.shape[0], -1)
        
        # Model
        model = MLPRegressor(hidden_layer_sizes=(16, 8, 16), max_iter=200, random_state=42)
        model.fit(X_train_flat, X_train_flat)
        
        # Prediction
        train_pred = model.predict(X_train_flat)
        train_mae = np.mean(np.abs(train_pred - X_train_flat), axis=1)
        threshold = np.mean(train_mae) + 3 * np.std(train_mae)
        
        test_pred = model.predict(X_test_flat)
        test_mae = np.mean(np.abs(test_pred - X_test_flat), axis=1)
        
        anomalies = test_mae > threshold
        num_anomalies = np.sum(anomalies)
        
        # Metrics
        m1, m2, m3 = st.columns(3)
        m1.metric("Global Threshold", f"{threshold:.4f}")
        m2.metric("Max Deviation", f"{np.max(test_mae):.4f}")
        m3.metric("Anomalies Found", f"{num_anomalies}", delta_color="inverse")
        
        # Visuals
        c1, c2 = st.columns([2, 1])
        
        with c1:
            st.subheader("Reconstruction Error")
            fig_err = go.Figure()
            fig_err.add_trace(go.Scatter(y=test_mae, name='Error', line=dict(color='cyan')))
            fig_err.add_trace(go.Scatter(y=[threshold]*len(test_mae), name='Threshold', line=dict(color='red', dash='dash')))
            
            anomaly_indices = np.where(anomalies)[0]
            if len(anomaly_indices) > 0:
                 fig_err.add_trace(go.Scatter(
                    x=anomaly_indices, y=test_mae[anomaly_indices], 
                    mode='markers', name='ANOMALY', marker=dict(color='red', size=8)
                ))
            fig_err.update_layout(template="plotly_dark", height=350)
            st.plotly_chart(fig_err, use_container_width=True)
            
        with c2:
            st.subheader("AI 'Brain' View (3D)")
            st.caption("Visualizing Latent Space (Normal vs Anomaly)")
            
            pca = PCA(n_components=3)
            latent_space = pca.fit_transform(X_test_flat)
            
            df_3d = pd.DataFrame(latent_space, columns=['x', 'y', 'z'])
            df_3d['Type'] = ['Anomaly' if x else 'Normal' for x in anomalies]
            
            fig_3d = px.scatter_3d(
                df_3d, x='x', y='y', z='z', 
                color='Type', 
                color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'},
                opacity=0.8
            )
            fig_3d.update_layout(template="plotly_dark", height=350, margin=dict(l=0,r=0,b=0,t=0))
            st.plotly_chart(fig_3d, use_container_width=True)

        if num_anomalies > 0:
            st.warning(f"⚠️ GLOBAL ALERT: Anomaly confirmed at {target_name}. Initiating protocols.")
        else:
            st.success("✅ System Status: All 300 Global Stations Green.")
