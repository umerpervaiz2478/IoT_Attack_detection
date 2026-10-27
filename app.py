streamlit_code = """import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from scapy.all import rdpcap, IP, TCP, UDP

# Page config
st.set_page_config(page_title="IoT Intrusion Detection Dashboard", layout="wide")

st.title("🛡️ IoT Network Intrusion Detection Dashboard")
st.write("Upload a network capture file (CSV or raw Wireshark PCAP) to instantly evaluate security threats.")

# Load Model Artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('model/xgb_model.joblib')
    scaler = joblib.load('model/scaler.joblib')
    classes = joblib.load('model/label_classes.joblib')
    feature_names = joblib.load('model/feature_names.joblib')
    return model, scaler, classes, feature_names

try:
    model, scaler, classes, feature_names = load_artifacts()
    st.sidebar.success("✅ Model & Scaler Loaded successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Failed loading model assets: {e}")
    st.stop()

# PCAP feature extractor function
def extract_features_from_pcap(pcap_file_path, expected_cols):
    packets = rdpcap(pcap_file_path)
    features_list = []
    for pkt in packets:
        if IP in pkt:
            row = {col: 0.0 for col in expected_cols}
            
            # Map parsed metrics
            row['Header_Length'] = float(len(pkt[IP]))
            row['Protocol Type'] = float(pkt[IP].proto)
            row['Tot size'] = float(len(pkt))
            
            if TCP in pkt:
                row['TCP'] = 1.0
                row['fin_flag_number'] = 1.0 if pkt[TCP].flags.F else 0.0
                row['syn_flag_number'] = 1.0 if pkt[TCP].flags.S else 0.0
                row['rst_flag_number'] = 1.0 if pkt[TCP].flags.R else 0.0
                row['psh_flag_number'] = 1.0 if pkt[TCP].flags.P else 0.0
                row['ack_flag_number'] = 1.0 if pkt[TCP].flags.A else 0.0
            elif UDP in pkt:
                row['UDP'] = 1.0
            
            features_list.append(row)
            
    return pd.DataFrame(features_list)

# File Upload Section
uploaded_file = st.file_uploader("Choose a file...", type=["csv", "pcap", "pcapng"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    st.info(f"Processing file: {file_name}")
    
    # 1. Processing Input
    if file_name.endswith('.csv'):
        input_df = pd.read_csv(uploaded_file)
        is_pcap = False
    else:
        with tempfile.NamedTemporaryFile(delete=False) as temp_pcap:
            temp_pcap.write(uploaded_file.read())
            temp_pcap_path = temp_pcap.name
        
        with st.spinner("Parsing raw packets from Wireshark PCAP... "):
            input_df = extract_features_from_pcap(temp_pcap_path, feature_names)
        os.unlink(temp_pcap_path)
        is_pcap = True
    
    # Ensure we only work with expected model features
    available_features = [col for col in feature_names if col in input_df.columns]
    missing_features = [col for col in feature_names if col not in input_df.columns]
    
    # Fill missing columns with 0.0
    for col in missing_features:
        input_df[col] = 0.0
        
    X_eval = input_df[feature_names]
    
    if len(X_eval) == 0:
        st.warning("No IP packets discovered or features matching the training data.")
        st.stop()
        
    # Display feature completeness warning for PCAPs
    if is_pcap:
        st.warning(
            "⚠️ Note on PCAP Parsing: Raw PCAPs lack advanced flow calculations (like IAT, Rates, and Variances) "
            "present in your CSV dataset. These columns have been defaulted to 0.0. Due to class imbalance in "
            "your model, this can bias classification results toward dominant classes like 'Flood Attacks'. "
            "For high-fidelity tests (like SQL Injections), upload a fully preprocessed CSV!"
        )
        
    # 2. Model Prediction
    with st.spinner("Evaluating dataset against XGBoost classifier..."):
        X_eval_scaled = scaler.transform(X_eval)
        predictions = model.predict(X_eval_scaled)
        pred_labels = [classes[p] for p in predictions]
        input_df['Predicted_Threat_Family'] = pred_labels

    # 3. UI Dashboard visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Threat Class Distribution")
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = input_df['Predicted_Threat_Family'].value_counts()
        sns.barplot(x=counts.index, y=counts.values, ax=ax, palette="viridis")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Count")
        st.pyplot(fig)
        
    with col2:
        st.subheader("🛡️ Threat Mitigation Verdict")
        benign_count = (input_df['Predicted_Threat_Family'] == 'Benign').sum()
        malicious_count = len(input_df) - benign_count
        pct_malicious = (malicious_count / len(input_df)) * 100 if len(input_df) > 0 else 0
        
        st.metric(label="Total Packets Scanned", value=len(input_df))
        st.metric(label="Malicious Alerts Identified", value=malicious_count, delta=f"{pct_malicious:.1f}% ratio", delta_color="inverse")
        
        if pct_malicious > 15.0:
            st.error("🚨 CRITICAL WARNING: High density of threat anomalies detected in active traffic!")
        else:
            st.success("✅ SAFE: Active session traffic matches healthy signatures.")
            
    st.subheader("📝 Analyzed Session Logs")
    st.dataframe(input_df[['Header_Length', 'Protocol Type', 'Tot size', 'Predicted_Threat_Family']].head(100))
"""

with open('streamlit_app/app.py', 'w') as f:
    f.write(streamlit_code)

print("App file successfully updated with warnings and smart feature mappings!")
