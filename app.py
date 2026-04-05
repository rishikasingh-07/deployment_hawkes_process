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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

h1 {
    font-weight: 700;
    letter-spacing: -0.5px;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

hr {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 25px 0;
}

.stButton > button {
    font-weight: 600;
    border-radius: 8px;
}

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

    # ---------------- CHANNEL ----------------
    channel = st.selectbox("Select Channel", raw.ch_names)

    # ---------------- OPTIONAL SEIZURE TIME ----------------
    seizure_input = st.text_input("Seizure Start Time (optional, in seconds)")

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

                # -------- SLIDING WINDOW --------
                centers, etas = sliding_window_eta(
                    spikes, T_total,
                    window_size=200,
                    step_size=50
                )

                final_result = None
                rejected_cases = []

                i = 20  # start after baseline

                # -------- FIXED DETECTION LOOP --------
                while i < len(centers):

                    # take only events AFTER current time window
                    sub_events = spikes[spikes >= centers[i] - 1000]

                    hyp_time, conf_time, status, prob, _ = \
                        adaptive_window_detection(sub_events, T_total)

                    if hyp_time is None:
                        i += 1
                        continue

                    if status == "confirmed":
                        final_result = (hyp_time, conf_time, status, prob)
                        break

                    elif status == "rejected":
                        rejected_cases.append((hyp_time, prob))

                        # 🔥 move forward to avoid same detection
                        i += 5
                        continue

                    else:
                        i += 1

                # fallback
                if final_result is None:
                    if rejected_cases:
                        last = rejected_cases[-1]
                        final_result = (last[0], None, "rejected", last[1])
                    else:
                        final_result = (None, None, "no_detection", 0)

                hyp_time, conf_time, status, prob = final_result

                # ---------------- RESULTS ----------------
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
