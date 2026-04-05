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

st.set_page_config(page_title="Hawkes Detection", layout="wide")

# ---------- CSS ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #f4f6fb;
}

/* Header */
.header {
    font-size: 32px;
    font-weight: 600;
    color: #6d28d9;
    margin-bottom: 10px;
}

.subheader {
    color: #64748b;
    margin-bottom: 30px;
}

/* Cards */
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0 6px 20px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}

/* Metrics */
.metric {
    font-size: 20px;
    font-weight: 600;
    color: #111827;
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
    border-radius: 10px;
    padding: 12px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="header">Hawkes Seizure Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subheader">Modeling neural instability using stochastic processes</div>', unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader("Upload EEG (.edf)", type=["edf"])

channel = None
seizure_start = None
use_seizure = False

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    st.sidebar.success("Loaded")

    channel = st.sidebar.selectbox("Channel", raw.ch_names)

    use_seizure = st.sidebar.checkbox("Add seizure time")

    if use_seizure:
        seizure_start = st.sidebar.number_input("Seizure time (sec)", min_value=0.0)

run = st.sidebar.button("Run Analysis")

# ---------- MAIN AREA ----------
if uploaded_file and run:

    with st.spinner("Running full Hawkes model..."):

        T_total = raw.times[-1]
        spikes = eeg_to_spikes(raw, channel)

        if len(spikes) < 20:
            st.error("Not enough spikes detected")
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

            col1.markdown(f"<div class='metric'>{status}</div><div class='metric-label'>Status</div>", unsafe_allow_html=True)
            col2.markdown(f"<div class='metric'>{prob:.2f}</div><div class='metric-label'>Probability</div>", unsafe_allow_html=True)
            col3.markdown(f"<div class='metric'>{hyp_time}</div><div class='metric-label'>Warning Time</div>", unsafe_allow_html=True)
            col4.markdown(f"<div class='metric'>{conf_time}</div><div class='metric-label'>Confirm Time</div>", unsafe_allow_html=True)

            # ---------- VALIDATION ----------
            if use_seizure and seizure_start and hyp_time:
                lead = seizure_start - hyp_time

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.subheader("Validation")

                st.write(f"Model Warning: {hyp_time:.2f} s")
                st.write(f"Actual Seizure: {seizure_start:.2f} s")
                st.write(f"Lead Time: {lead:.2f} s")

                if lead > 0:
                    st.success("Early detection")
                else:
                    st.warning("Late detection")

                st.markdown("</div>", unsafe_allow_html=True)

            # ---------- GRAPH ----------
            st.markdown("<div class='card'>", unsafe_allow_html=True)

            fig, ax = plt.subplots(figsize=(12,4))
            ax.plot(centers, etas, color="#7c3aed", linewidth=2)

            if hyp_time:
                ax.axvline(hyp_time, color='orange', label='Warning')

            if use_seizure and seizure_start:
                ax.axvline(seizure_start, color='red', label='Seizure')

            ax.set_title("Branching Ratio η")
            ax.legend()

            st.pyplot(fig)

            st.markdown("</div>", unsafe_allow_html=True)
