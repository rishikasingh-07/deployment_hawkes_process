import streamlit as st
import matplotlib.pyplot as plt
import mne
import tempfile

from hawkes_core import (
    eeg_to_spikes,
    sliding_window_eta,
    adaptive_window_detection,
    critical_threshold_warning
)

st.set_page_config(layout="centered")

st.title("Hawkes Process Seizure Detection")

st.markdown("""
### Instructions

1. Download EEG (.edf) from CHB-MIT dataset  
2. Upload below  
3. Select channel  
4. (Optional) enter seizure start time  
""")

uploaded_file = st.file_uploader("Upload EEG (.edf)", type=["edf"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
    st.success("EEG loaded")

    channel = st.selectbox("Select Channel", raw.ch_names)

    seizure_input = st.text_input("Seizure Start Time (optional)")

    seizure_start = None
    if seizure_input.strip():
        seizure_start = float(seizure_input)

    if st.button("Run Analysis"):

        T_total = raw.times[-1]

        spikes = eeg_to_spikes(raw, channel)

        if len(spikes) < 50:
            st.error("Not enough spikes")
        else:

            # -------- ORIGINAL DETECTION --------
            hyp_time, conf_time, status, prob, rejected = \
                adaptive_window_detection(spikes, T_total)

            # -------- ALSO CHECK SUPERCRITICAL --------
            centers, etas = sliding_window_eta(spikes, T_total)

            mask = centers > 300
            centers = centers[mask]
            etas = etas[mask]

            critical_time = critical_threshold_warning(centers, etas)

            # -------- RESULT LOGIC (THIS WAS MISSING) --------
            final_status = status
            final_time = hyp_time

            if status == "confirmed":
                final_status = "confirmed"

            elif status == "rejected":
                final_status = "rejected"

            elif critical_time:
                final_status = "supercritical"
                final_time = critical_time

            else:
                final_status = "no_detection"

            # -------- DISPLAY --------
            st.markdown("### Detection Summary")

            st.write(f"**Status:** {final_status}")

            if final_status == "confirmed":
                st.write(f"Confidence: {prob:.2f}")
                st.write(f"Warning Time: {hyp_time}")
                st.write(f"Confirmation Time: {conf_time}")

            elif final_status == "rejected":
                st.warning("Transient detected — correctly rejected")
                st.write(f"Rejected at: {hyp_time}")
                st.write(f"Final P: {prob:.2f}")

            elif final_status == "supercritical":
                st.error("η ≥ 1.0 — system unstable")
                st.write(f"Time: {critical_time}")

            else:
                st.info("No detection")

            # -------- VALIDATION --------
            if seizure_start and final_time:
                lead = seizure_start - final_time

                st.markdown("### Validation")
                st.write(f"Actual seizure: {seizure_start}")
                st.write(f"Lead time: {lead:.2f} sec")

            # -------- PLOT --------
            fig, ax = plt.subplots(figsize=(6,3))
            ax.plot(centers, etas)

            if hyp_time:
                ax.axvline(hyp_time, color='orange')

            if seizure_start:
                ax.axvline(seizure_start, color='red')

            st.pyplot(fig)
