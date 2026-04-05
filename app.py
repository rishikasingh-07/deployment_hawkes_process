import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne
import tempfile

from hawkes_core import (
    eeg_to_spikes,
    sliding_window_eta,
    adaptive_window_detection
)

st.set_page_config(layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Header */
.header {
    text-align: center;
    margin-bottom: 30px;
}
.header h1 {
    color: #7c3aed;
    font-size: 40px;
}
.header p {
    color: #6b7280;
}

/* Card */
.card {
    background: white;
    padding: 25px;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Section title */
.section-title {
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 10px;
    color: #374151;
}

/* Metrics */
.metric-box {
    text-align: center;
    padding: 20px;
    border-radius: 14px;
    background: #ffffff;
    box-shadow: 0 5px 15px rgba(0,0,0,0.05);
}

.metric-value {
    font-size: 22px;
    font-weight: 600;
}

.metric-label {
    color: #6b7280;
    font-size: 14px;
}

/* Button */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #7c3aed, #a78bfa);
    color: white;
    border-radius: 12px;
    padding: 14px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h1>Hawkes Process Seizure Detection</h1>
    <p>Modeling neural instability using stochastic processes</p>
</div>
""", unsafe_allow_html=True)

# ---------- INPUT SECTION ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-title">Upload EEG</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload .edf file", type=["edf"])

with col2:
    st.markdown('<div class="section-title">Settings</div>', unsafe_allow_html=True)
    use_seizure = st.checkbox("Provide seizure time")
    seizure_start = None
    if use_seizure:
        seizure_start = st.number_input("Seizure time (seconds)", min_value=0.0)

st.markdown('</div>', unsafe_allow_html=True)

# ---------- PROCESS ----------
if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    st.success("EEG loaded")

    channel = st.selectbox("Select Channel", raw.ch_names)

    if st.button("Run Analysis"):

        with st.spinner("Running Hawkes model..."):

            T_total = raw.times[-1]
            spikes = eeg_to_spikes(raw, channel)

            if len(spikes) < 20:
                st.error("Not enough spikes")
            else:

                hyp_time, conf_time, status, prob, _ = \
                    adaptive_window_detection(spikes, T_total)

                centers, etas = sliding_window_eta(
                    spikes, T_total,
                    window_size=200,
                    step_size=50
                )

                # ---------- METRICS ----------
                col1, col2, col3, col4 = st.columns(4)

                col1.markdown(f"<div class='metric-box'><div class='metric-value'>{status}</div><div class='metric-label'>Status</div></div>", unsafe_allow_html=True)
                col2.markdown(f"<div class='metric-box'><div class='metric-value'>{prob:.2f}</div><div class='metric-label'>Probability</div></div>", unsafe_allow_html=True)
                col3.markdown(f"<div class='metric-box'><div class='metric-value'>{hyp_time}</div><div class='metric-label'>Warning Time</div></div>", unsafe_allow_html=True)
                col4.markdown(f"<div class='metric-box'><div class='metric-value'>{conf_time}</div><div class='metric-label'>Confirm Time</div></div>", unsafe_allow_html=True)

                # ---------- VALIDATION ----------
                if use_seizure and seizure_start and hyp_time:
                    lead = seizure_start - hyp_time

                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### Validation")

                    st.write(f"Model Warning: {hyp_time:.2f} s")
                    st.write(f"Actual Seizure: {seizure_start:.2f} s")
                    st.write(f"Lead Time: {lead:.2f} s")

                    st.markdown('</div>', unsafe_allow_html=True)

                # ---------- GRAPH ----------
                st.markdown('<div class="card">', unsafe_allow_html=True)

                fig, ax = plt.subplots(figsize=(12,4))
                ax.plot(centers, etas, color="#7c3aed", linewidth=2)

                if hyp_time:
                    ax.axvline(hyp_time, color='orange')

                if use_seizure and seizure_start:
                    ax.axvline(seizure_start, color='red')

                ax.set_title("Branching Ratio η")

                st.pyplot(fig)

                st.markdown('</div>', unsafe_allow_html=True)
