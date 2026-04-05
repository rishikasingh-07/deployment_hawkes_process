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

# ---------------- INSTRUCTIONS ----------------
st.markdown("""
### Instructions
Download data from:
https://physionet.org/content/chbmit/1.0.0/
1. Open any patient folder (e.g., chb01) and download a .edf file
2. Upload the file below  
3. Select a channel  
4. (Optional) Enter seizure start time  

**How to find seizure time:**
- Open the `summary.txt` file from the dataset  
- Look for lines like:  
  `Seizure Start Time: XXXX seconds`  
- Enter that value below  

Then click **Run Analysis**
""")

# ---------------- UPLOAD ----------------
uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    st.success("EEG loaded successfully")

    # ---------------- CHANNEL ----------------
    channel = st.selectbox("Select Channel", raw.ch_names)

    # ---------------- OPTIONAL SEIZURE TIME ----------------
    seizure_input = st.text_input(
        "Seizure Start Time (optional, in seconds)"
    )

    seizure_start = None
    if seizure_input.strip() != "":
        try:
            seizure_start = float(seizure_input)
        except:
            st.warning("Invalid seizure time format")

    # ---------------- RUN ----------------
    if st.button("Run Analysis"):

        with st.spinner("Running Hawkes model..."):

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

                # ---------------- RESULTS ----------------
                st.markdown("### Results")

                st.write(f"Status: {status}")
                st.write(f"Probability: {prob:.2f}")
                st.write(f"Warning Time: {hyp_time}")
                st.write(f"Confirmation Time: {conf_time}")

                # ---------------- VALIDATION ----------------
                if seizure_start is not None and hyp_time is not None:
                    lead = seizure_start - hyp_time

                    st.markdown("### Validation")

                    st.write(f"Actual Seizure Time: {seizure_start}")
                    st.write(f"Lead Time: {lead:.2f} seconds")

                    if lead > 0:
                        st.success("Early detection")
                    else:
                        st.warning("Detection after seizure onset")

                # ---------------- GRAPH ----------------
                st.markdown("### η (Branching Ratio) Over Time")

                fig, ax = plt.subplots(figsize=(6,3))  # smaller graph
                ax.plot(centers, etas)

                if hyp_time:
                    ax.axvline(hyp_time, color='orange')

                if seizure_start:
                    ax.axvline(seizure_start, color='red')

                st.pyplot(fig)
