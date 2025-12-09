import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# ==========================================
# BACKEND: LOGIC
# ==========================================

def generate_ocean_data(n_points=1000, noise_level=0.1):
    """Generates synthetic 'normal' ocean data."""
    t = np.linspace(0, 50, n_points)
    data = np.sin(t) + 0.5 * np.sin(t * 0.5) 
    noise = np.random.normal(0, noise_level, n_points)
    return data + noise

def inject_anomalies(data, anomaly_type):
    """Injects specific errors: Spikes, Drifts, Flatlines."""
    modified_data = data.copy()
    n = len(data)
    start_idx = int(n * 0.85)
    end_idx = int(n * 0.95)
    
    if anomaly_type == "Spike (Sudden Jump)":
        modified_data[start_idx] += 3.0 
    elif anomaly_type == "Drift (Sensor Aging)":
        drift = np.linspace(0, 2, end_idx - start_idx)
        modified_data[start_idx:end_idx] += drift
    elif anomaly_type == "Flatline (Dead Sensor)":
        modified_data[start_idx:end_idx] = modified_data[start_idx]
        
    return modified_data

def build_autoencoder(input_dim):
    """Builds the Autoencoder Neural Network."""
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(16, activation="relu")(input_layer)
    encoded = Dense(8, activation="relu")(encoded) 
    decoded = Dense(16, activation="relu")(encoded)
    output_layer = Dense(input_dim, activation="linear")(decoded)
    
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def create_sequences(data, time_steps=10):
    """Converts data into sequences for the model."""
    output = []
    for i in range(len(data) - time_steps):
        output.append(data[i : (i + time_steps)])
    return np.stack(output)

# ==========================================
# FRONTEND: UI
# ==========================================

st.set_page_config(page_title="Intelligent Shield", layout="wide")

st.title("🌊 The Intelligent Shield: AI Ocean Anomaly Detection")
st.markdown("**Project:** Early Anomaly Detection using Autoencoders (Digital Twin).")

# Sidebar
st.sidebar.header("Control Panel")
anomaly_choice = st.sidebar.selectbox(
    "Select Anomaly to Simulate:",
    ["None (Clean Data)", "Spike (Sudden Jump)", "Drift (Sensor Aging)", "Flatline (Dead Sensor)"]
)
noise_lvl = st.sidebar.slider("Sensor Noise Level", 0.0, 0.5, 0.1)

# Simulation
st.subheader("1. Data Stream Simulation")
raw_data = generate_ocean_data(n_points=1000, noise_level=noise_lvl)

if anomaly_choice != "None (Clean Data)":
    display_data = inject_anomalies(raw_data, anomaly_choice)
else:
    display_data = raw_data

# Data Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(display_data.reshape(-1, 1))

col1, col2 = st.columns([3, 1])
with col1:
    st.line_chart(display_data[-200:], height=250)
with col2:
    st.info(f"**Status:** Simulating {anomaly_choice}")

# AI Analysis
st.subheader("2. The 'Digital Twin' Analysis")

if st.button("🛡️ Activate Shield"):
    with st.spinner("Training Autoencoder on historical data..."):
        TIME_STEPS = 30
        
        # 1. Create Sequences
        sequences = create_sequences(scaled_data, TIME_STEPS)
        
        # 2. Fix Shapes (Remove extra dimension for Dense layer)
        # Squeeze (N, 30, 1) -> (N, 30) to match model expectation
        sequences = sequences.squeeze()

        # 3. Split Data
        train_size = int(len(sequences) * 0.5)
        X_train = sequences[:train_size]
        X_test = sequences[train_size:] 
        
        # 4. Train Model
        model = build_autoencoder(TIME_STEPS)
        history = model.fit(X_train, X_train, epochs=20, batch_size=32, verbose=0, shuffle=True)
        
        # 5. Detect Anomalies
        # Predict on Test Data
        test_pred = model.predict(X_test)
        test_mae = np.mean(np.abs(test_pred - X_test), axis=1)
        
        # Calculate Threshold from Training Data
        train_pred = model.predict(X_train)
        train_mae = np.mean(np.abs(train_pred - X_train), axis=1)
        threshold = np.mean(train_mae) + 3 * np.std(train_mae)
        
        anomalies = test_mae > threshold
        
        # 6. Visualization
        st.success("Analysis Complete!")
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(test_mae, label='Reconstruction Error', color='blue')
        ax.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold')
        
        anomaly_indices = np.where(anomalies)[0]
        if len(anomaly_indices) > 0:
            ax.scatter(anomaly_indices, test_mae[anomaly_indices], color='red', s=30, label='Anomaly Detected', zorder=5)
            st.error(f"🚨 ALERT: {len(anomaly_indices)} anomalies detected!")
        else:
            st.success("✅ System Healthy.")
        ax.legend()
        st.pyplot(fig)
