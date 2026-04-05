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

st.set_page_config(layout="centered")

# ---------------- TITLE ----------------
st.title("Hawkes Process Seizure Detection")

st.markdown("""
<div style='text-align:center; font-size:16px; color:#6b7280; margin-bottom:20px;'>
Stochastic Modeling for Early Seizure Warning
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ---------------- INSTRUCTIONS ----------------
st.markdown("""
### Instructions

1. Download an EEG `.edf` file from the CHB-MIT dataset  
2. Upload the file below  
3. Select a channel  
4. (Optional) Enter seizure start time  

**How to find seizure time:**
- Open the `summary.txt` file  
- Look for: `Seizure Start Time: XXXX seconds`  

Then click **Run Analysis**
""")

st.markdown("---")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    st.success("EEG loaded successfully")

    channel = st.selectbox("Select Channel", raw.ch_names)

    seizure_input = st.text_input("Seizure Start Time (optional, in seconds)")

    seizure_start = None
    if seizure_input.strip() != "":
        try:
            seizure_start = float(seizure_input)
        except:
            st.warning("Invalid seizure time format")

    st.markdown("---")

    if st.button("Run Analysis"):

        with st.spinner("Running Hawkes model..."):

            T_total = raw.times[-1]
            spikes = eeg_to_spikes(raw, channel)

            if len(spikes) < 20:
                st.error("Not enough spikes detected")
            else:

                hyp_time, conf_time, status, prob, rejected = \
                    adaptive_window_detection(spikes, T_total)

                st.markdown("### Detection Summary")

                st.write(f"**Status:** {status}")

                if status == "confirmed":
                    st.write(f"**Confidence Score:** {prob:.2f}")
                    st.write(f"**Early Warning Time:** {hyp_time} s")
                    st.write(f"**Confirmation Time:** {conf_time} s")

                elif status == "rejected":
                    st.warning("Detection rejected (transient activity)")
                    st.write(f"Rejected at: {hyp_time} s")
                    st.write(f"Final Probability: {prob:.2f}")

                else:
                    st.info("No significant detection found")

                if seizure_start is not None and hyp_time is not None:
                    lead = seizure_start - hyp_time

                    st.markdown("### Validation")
                    st.write(f"**Actual Seizure Time:** {seizure_start} s")
                    st.write(f"**Early Warning Lead:** {lead:.2f} seconds")

                st.markdown("---")

                centers, etas = sliding_window_eta(spikes, T_total)

                fig, ax = plt.subplots(figsize=(7,3))
                ax.plot(centers, etas)

                if hyp_time:
                    ax.axvline(hyp_time, color='orange', label='Warning')

                if seizure_start:
                    ax.axvline(seizure_start, color='red', label='Seizure')

                ax.legend()
                st.pyplot(fig)
