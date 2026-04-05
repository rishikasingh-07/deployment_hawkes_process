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

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Hawkes Seizure Detection",
    layout="wide"
)

# ---------------- CUSTOM UI ----------------
st.markdown("""
<style>
/* Background */
body {
    background-color: #0b0b12;
}

/* Main container */
.block-container {
    padding-top: 2rem;
}

/* Titles */
h1 {
    color: #8b5cf6;
    text-align: center;
}
h2, h3 {
    color: #8b5cf6;
}

/* Cards */
.card {
    background-color: #111827;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0px 0px 20px rgba(139, 92, 246, 0.15);
    margin-bottom: 20px;
}

/* Buttons */
.stButton > button {
    background-color: #8b5cf6;
    color: white;
    border-radius: 10px;
    padding: 10px 20px;
    font-weight: bold;
}
.stButton > button:hover {
    background-color: #7c3aed;
}

/* Info text */
.highlight {
    color: #22d3ee;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<h1>Hawkes Process Seizure Detection</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Stochastic Modeling of Neural Excitability</p>",
    unsafe_allow_html=True
)

# ---------------- INSTRUCTIONS ----------------
with st.expander("Dataset & Instructions"):
    st.markdown("""
    Dataset: CHB-MIT EEG (PhysioNet)

    Steps:
    1. Download an EDF file from the dataset  
    2. Upload the file  
    3. Select a channel  
    4. (Optional) Enter seizure start time  
    5. Run analysis  

    Note: Full MLE is used. Processing may take a few minutes.
    """)

# ---------------- UPLOAD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file:

    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)

    st.success("EEG loaded successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        channel = st.selectbox("Select Channel", raw.ch_names)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        use_seizure = st.checkbox("Provide seizure time for validation")

        seizure_start = None
        if use_seizure:
            seizure_start = st.number_input(
                "Seizure Start Time (seconds)",
                min_value=0.0
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- RUN ----------------
    if st.button("Run Full Analysis"):

        if use_seizure and seizure_start == 0:
            st.error("Enter valid seizure time")
        else:
            with st.spinner("Running Hawkes MLE... please wait"):

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
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.subheader("Detection Results")

                    st.write(f"Hypothesis Time: {hyp_time}")
                    st.write(f"Confirmation Time: {conf_time}")
                    st.write(f"Status: {status}")
                    st.write(f"Probability: {prob:.2f}")

                    if not use_seizure:
                        st.info("Model operating without ground truth")

                    if use_seizure and seizure_start:

                        if hyp_time:
                            lead = seizure_start - hyp_time

                            st.markdown("### Validation")

                            st.write(f"Model Warning Time: {hyp_time:.2f} s")
                            st.write(f"Actual Seizure Time: {seizure_start:.2f} s")
                            st.write(f"Lead Time: {lead:.2f} seconds")

                            if lead > 0:
                                st.success("Early detection achieved")
                            else:
                                st.warning("Detection after onset")

                    st.markdown("</div>", unsafe_allow_html=True)

                    # ---------------- PLOT ----------------
                    st.markdown("<div class='card'>", unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(12, 4))
                    ax.plot(centers, etas, color="#22d3ee")

                    if hyp_time:
                        ax.axvline(hyp_time, color='orange', label='Hypothesis')

                    if use_seizure and seizure_start:
                        ax.axvline(seizure_start, color='red', label='Seizure')

                    ax.set_title("Branching Ratio η Over Time", color="white")
                    ax.set_facecolor("#0b0b12")
                    ax.tick_params(colors='white')
                    ax.legend()

                    st.pyplot(fig)

                    st.markdown("</div>", unsafe_allow_html=True)
