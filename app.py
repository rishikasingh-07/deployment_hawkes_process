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

# ---------------- UI POLISH ----------------
st.markdown("""
<style>

/* Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Title */
h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

/* Section headings */
h3 {
    font-weight: 600;
    margin-top: 20px;
}

/* Spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Divider */
hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 25px 0;
}

/* Buttons */
.stButton > button {
    font-weight: 600;
    border-radius: 8px;
}

/* Inputs */
input, select {
    border-radius: 6px !important;
}

/* Highlight important values */
strong {
    color: #7c3aed;
}

</style>
""", unsafe_allow_html=True)

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
- Open the `summary.txt` file from the dataset  
- Look for: `Seizure Start Time: XXXX seconds`  
- Enter that value below  

Then click **Run Analysis**
""")

st.markdown("---")

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

    st.markdown("---")

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
                st.markdown("### Detection Summary")

                st.write(f"**Status:** {status}")
                st.write(f"**Confidence Score:** {prob:.2f}")
                st.write(f"**Early Warning Time:** {hyp_time} s")
                st.write(f"**Confirmation Time:** {conf_time} s")

                # ---------------- VALIDATION ----------------
                if seizure_start is not None and hyp_time is not None:
                    lead = seizure_start - hyp_time

                    st.markdown("### Validation")

                    st.write(f"**Actual Seizure Time:** {seizure_start} s")
                    st.write(f"**Early Warning Lead:** {lead:.2f} seconds")

                    if lead > 0:
                        st.success("Early detection achieved")
                    else:
                        st.warning("Detection occurred after onset")

                st.markdown("---")

                # ---------------- GRAPH ----------------
                st.markdown("### η (Branching Ratio) Over Time")

                fig, ax = plt.subplots(figsize=(7,3))
                ax.plot(centers, etas)

                if hyp_time:
                    ax.axvline(hyp_time, color='orange', label='Warning')

                if seizure_start:
                    ax.axvline(seizure_start, color='red', label='Seizure')

                ax.legend()
                ax.set_title("η (Branching Ratio)", fontsize=11)

                st.pyplot(fig)
