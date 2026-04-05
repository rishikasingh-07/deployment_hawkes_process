import streamlit as st
import matplotlib.pyplot as plt
import mne
import tempfile

from hawkes_core import *

st.title("Hawkes Seizure Detection")

uploaded_file = st.file_uploader("Upload EDF", type=["edf"])

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

    st.success("EEG loaded")

    channel = st.selectbox("Select Channel", raw.ch_names)

    if st.button("Run"):

        T_total = raw.times[-1]

        spikes = eeg_to_spikes(raw, channel)

        hyp, conf, status, prob, rejected = \
            adaptive_window_detection(spikes, T_total)

        st.write("Status:", status)
        st.write("Hypothesis:", hyp)
        st.write("Confirm:", conf)
        st.write("Probability:", prob)

        centers, etas = sliding_window_eta(spikes, T_total)

        fig, ax = plt.subplots()
        ax.plot(centers, etas)

        if hyp:
            ax.axvline(hyp)

        st.pyplot(fig)
