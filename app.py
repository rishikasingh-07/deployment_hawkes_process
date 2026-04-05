import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import mne

from hawkes_core import (
    eeg_to_spikes,
    sliding_window_eta,
    adaptive_window_detection
)

st.set_page_config(page_title="Hawkes Seizure Detection", layout="wide")

# ---------- THEME ----------
st.markdown("""
<style>
body {
    background-color: #0b0b12;
    color: white;
}
h1, h2, h3 {
    color: #8b5cf6;
}
</style>
""", unsafe_allow_html=True)

# ---------- TITLE ----------
st.title("Hawkes Process Seizure Detection")
st.markdown("Full MLE-based stochastic modeling of EEG data")

# ---------- INSTRUCTIONS ----------
with st.expander("Instructions"):
    st.markdown("""
    Dataset: CHB-MIT EEG (PhysioNet)

    Steps:
    1. Download an EDF file from the dataset
    2. Upload the file here
    3. Select a channel
    4. Enter seizure start time (seconds)
    5. Run full analysis

    Note: This uses full MLE and may take a few minutes.
    """)

# ---------- UPLOAD ----------
uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

if uploaded_file:

    raw = mne.io.read_raw_edf(uploaded_file, preload=True, verbose=False)

    st.success("File loaded successfully")

    channel = st.selectbox("Select Channel", raw.ch_names)

    seizure_start = st.number_input("Seizure Start Time (seconds)", min_value=0.0)

    if st.button("Run Full Analysis"):

        if seizure_start == 0:
            st.error("Enter seizure start time")
        else:
            with st.spinner("Running full Hawkes pipeline..."):

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

                    st.subheader("Results")

                    st.write(f"Hypothesis Time: {hyp_time}")
                    st.write(f"Confirmation Time: {conf_time}")
                    st.write(f"Status: {status}")
                    st.write(f"Probability: {prob:.2f}")

                    if hyp_time:
                        lead = seizure_start - hyp_time
                        st.write(f"Lead Time: {lead:.2f} seconds")

                    fig, ax = plt.subplots(figsize=(10,4))
                    ax.plot(centers, etas)

                    if hyp_time:
                        ax.axvline(hyp_time, color='orange', label='Hypothesis')

                    ax.axvline(seizure_start, color='red', label='Seizure')

                    ax.legend()
                    ax.set_title("η over time")

                    st.pyplot(fig)
