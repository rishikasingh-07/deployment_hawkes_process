"""
Hawkes Process Seizure Early Warning System
Author: Rishika Singh
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize
import tempfile

st.set_page_config(
    page_title="Hawkes Seizure Detection",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0a0e17;
    color: #c8d6e8;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem 3rem; max-width: 1100px; }

.title-block {
    border-left: 3px solid #3b82f6;
    padding: 1.2rem 1.8rem;
    margin-bottom: 2.5rem;
    background: linear-gradient(90deg, rgba(59,130,246,0.06) 0%, transparent 100%);
}
.title-block h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.7rem; font-weight: 600;
    color: #e8f0ff; letter-spacing: -0.02em; margin: 0 0 0.3rem 0;
}
.title-block p { font-size: 0.88rem; color: #6b8aad; margin: 0; font-weight: 300; }

.section-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem; font-weight: 500;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: #3b82f6; margin-bottom: 0.8rem; margin-top: 2rem;
}

.instr-card {
    background: #0f1623; border: 1px solid #1e2d42;
    border-radius: 6px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.instr-card ol { margin: 0; padding-left: 1.2rem; color: #8aa5c4; font-size: 0.88rem; line-height: 2; }
.instr-card ol li strong { color: #c8d6e8; }
.instr-card a { color: #60a5fa; }

.instr-sub {
    background: #0a1120; border: 1px solid #1a2535;
    border-radius: 6px; padding: 1rem 1.4rem; margin-top: 0.8rem;
    font-size: 0.84rem; color: #6b8aad; line-height: 1.8;
}
.instr-sub strong { color: #94a3b8; }
.instr-sub code {
    background: #151f30; padding: 0.1rem 0.4rem; border-radius: 3px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; color: #60a5fa;
}

.badge {
    display: inline-block; font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem; font-weight: 500; padding: 0.25rem 0.7rem;
    border-radius: 3px; letter-spacing: 0.08em; text-transform: uppercase;
}
.badge-confirmed { background: rgba(34,197,94,0.12); color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.badge-rejected  { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
.badge-uncertain { background: rgba(234,179,8,0.12);  color: #fbbf24; border: 1px solid rgba(234,179,8,0.25); }

.result-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1.2rem 0; }
.result-card { background: #0f1623; border: 1px solid #1e2d42; border-radius: 6px; padding: 1.2rem 1.4rem; }
.result-card .label { font-size: 0.72rem; color: #4a6a8a; text-transform: uppercase; letter-spacing: 0.1em; font-family: 'IBM Plex Mono', monospace; margin-bottom: 0.4rem; }
.result-card .value { font-family: 'IBM Plex Mono', monospace; font-size: 1.3rem; font-weight: 600; color: #e8f0ff; }
.result-card .unit { font-size: 0.75rem; color: #4a6a8a; margin-left: 0.2rem; }

.val-block {
    background: #0f1623; border: 1px solid #1e2d42;
    border-left: 3px solid #3b82f6; border-radius: 6px;
    padding: 1.2rem 1.6rem; margin-top: 1rem;
}
.val-row { display: flex; justify-content: space-between; align-items: center; padding: 0.4rem 0; border-bottom: 1px solid #111d2e; font-size: 0.88rem; }
.val-row:last-child { border-bottom: none; }
.val-row .vk { color: #4a6a8a; }
.val-row .vv { font-family: 'IBM Plex Mono', monospace; color: #c8d6e8; font-weight: 500; }
.lead-positive { color: #4ade80 !important; }
.lead-negative { color: #f87171 !important; }

.msg-box {
    background: #0a1120; border: 1px dashed #1e2d42; border-radius: 6px;
    padding: 1.2rem 1.6rem; text-align: center; color: #4a6a8a;
    font-size: 0.88rem; margin: 1rem 0;
}

div.stButton > button {
    background: #1d4ed8; color: #e8f0ff; border: none; border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace; font-size: 0.82rem; font-weight: 500;
    letter-spacing: 0.08em; padding: 0.65rem 2rem; text-transform: uppercase;
    transition: background 0.15s; width: 100%;
}
div.stButton > button:hover { background: #2563eb; }
label { color: #6b8aad !important; font-size: 0.84rem !important; }
</style>
""", unsafe_allow_html=True)


# ── core functions ──────────────────────────────────────────

def hawkes_log_likelihood(params, events, T):
    mu, alpha, beta = params
    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e10
    events = np.array(events)
    dt = events[:, None] - events[None, :]
    mask = dt > 0
    exponent = np.clip(np.where(mask, -beta * dt, -np.inf), -500, 0)
    intensities = mu + np.where(mask, alpha * np.exp(exponent), 0).sum(axis=1)
    if np.any(intensities <= 0):
        return 1e10
    part1 = np.sum(np.log(intensities))
    part2 = mu * T + (alpha / beta) * np.sum(1 - np.exp(-beta * (T - events)))
    return -(part1 - part2)


def fit_hawkes(events, T, n_restarts=8):
    best, best_ll = None, np.inf
    for _ in range(n_restarts):
        x0 = [np.random.uniform(0.1, 2.0),
               np.random.uniform(0.1, 0.9),
               np.random.uniform(0.5, 5.0)]
        r = minimize(hawkes_log_likelihood, x0, args=(events, T),
                     method='L-BFGS-B',
                     bounds=[(1e-6, None)] * 3,
                     options={'maxiter': 800})
        if r.fun < best_ll:
            best_ll, best = r.fun, r
    mu, alpha, beta = best.x
    return {'mu': mu, 'alpha': alpha, 'beta': beta, 'eta': alpha / beta}


def eeg_to_spikes(raw, channel, low_freq=80, high_freq=120, thr=3.0):
    try:
        sfreq = raw.info['sfreq']
        raw_f = raw.copy().filter(low_freq, high_freq, picks=[channel], verbose=False)
        data  = raw_f.get_data()[raw.ch_names.index(channel)]
        ws    = int(60 * sfreq)
        norm  = np.zeros_like(data)
        for i in range(0, len(data), ws):
            c = data[i:i + ws]
            if np.std(c) > 0:
                norm[i:i + ws] = (c - np.mean(c)) / np.std(c)
        cross = np.where(np.diff((np.abs(norm) > thr).astype(int)) == 1)[0]
        if len(cross) == 0:
            return np.array([])
        st_ = cross / sfreq
        filt = [st_[0]]
        for t in st_[1:]:
            if t - filt[-1] > 0.05:
                filt.append(t)
        return np.array(filt)
    except Exception:
        return np.array([])


def sliding_window_eta(events, T_total, window_size=200, step_size=50, min_spikes=15):
    centers, etas = [], []
    start = 0
    while start + window_size <= T_total:
        end  = start + window_size
        w    = events[(events >= start) & (events < end)] - start
        if len(w) >= min_spikes:
            etas.append(fit_hawkes(w, T=window_size)['eta'])
            centers.append(start + window_size / 2)
        start += step_size
    return np.array(centers), np.array(etas)


def probabilistic_verification(ac, ae, threshold, p_confirm=0.85, p_reject=0.15):
    p = 0.5
    for t, e in zip(ac, ae):
        p = p + (1 - p) * 0.4 if e > threshold else p - p * 0.4
        if p >= p_confirm:
            return t, 'confirmed', p
        if p <= p_reject:
            return t, 'rejected', p
    return None, 'uncertain', p


def run_detection(events, T_total, progress_cb=None):
    cn, en = sliding_window_eta(events, T_total)
    if progress_cb: progress_cb(0.55)

    mask = cn > 300
    vc, ve = cn[mask], en[mask]
    rejected, hyp_time, in_hyp, hyp_count = [], None, False, 0

    for i in range(20, len(ve)):
        recent    = ve[i - 20:i]
        threshold = np.mean(recent) + 1.5 * max(np.std(recent), 0.05)
        cond      = (ve[i] > threshold and
                     ve[i] > ve[i - 2] and
                     np.min(ve[max(0, i - 10):i]) > 0.3)

        if not in_hyp:
            hyp_count = hyp_count + 1 if cond else 0
            if hyp_count >= 2:
                hyp_time = vc[i - 1]
                in_hyp   = True
                t_hyp    = vc[i]

                ca, ea = sliding_window_eta(events, T_total, 100, 20)
                am     = ca > t_hyp
                ac, ae = ca[am], ea[am]

                if len(ac) < 2:
                    return hyp_time, None, 'uncertain', 0.5, rejected, vc, ve

                vt, status, fp = probabilistic_verification(ac, ae, threshold)
                if progress_cb: progress_cb(0.9)

                if status == 'confirmed':
                    return hyp_time, vt, status, fp, rejected, vc, ve

                elif status == 'rejected':
                    print(f"  REJECTED at t={verify_time:.0f}s"
                          f" — P(seizure)={final_p:.2f} transient")
                    rejected_times.append({
                        'hypothesis': hypothesis_time,  # t=1750
                        'rejected_at': verify_time,     # t=1910
                        'final_p': final_p              # 0.09
                    })
                    hypothesis_time = None
                    in_hypothesis   = False
                    hyp_count       = 0
                
                else:
                    return hyp_time, None, status, fp, rejected, vc, ve

    return hyp_time, None, 'uncertain', 0.5, rejected, vc, ve


def make_plot(centers, etas, hyp_time, conf_time, seizure_start, channel):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    fig.patch.set_facecolor('#0a0e17')
    ax.set_facecolor('#0a0e17')

    ax.plot(centers, etas, color='#60a5fa', linewidth=1.8, zorder=3)
    ax.fill_between(centers, etas, alpha=0.07, color='#3b82f6')
    ax.axhline(y=1.0, color='#475569', linewidth=1, linestyle='--', alpha=0.6)

    handles = [mpatches.Patch(color='#60a5fa', label='η (branching ratio)'),
               mpatches.Patch(color='#475569', label='Supercritical η=1.0')]

    if hyp_time:
        ax.axvline(x=hyp_time, color='#f59e0b', linewidth=1.8, linestyle='--', alpha=0.9)
        ax.text(hyp_time + (centers[-1] - centers[0]) * 0.01, 1.15,
                f't={hyp_time:.0f}s', color='#f59e0b', fontsize=7.5, fontfamily='monospace')
        handles.append(mpatches.Patch(color='#f59e0b', label=f'Warning t={hyp_time:.0f}s'))

    if conf_time:
        ax.axvline(x=conf_time, color='#22c55e', linewidth=1.5, linestyle=':', alpha=0.8)
        handles.append(mpatches.Patch(color='#22c55e', label=f'Confirmed t={conf_time:.0f}s'))

    if seizure_start:
        ax.axvline(x=seizure_start, color='#ef4444', linewidth=2, linestyle='-', alpha=0.9)
        ax.text(seizure_start + (centers[-1] - centers[0]) * 0.01, 1.15,
                f't={seizure_start}s', color='#ef4444', fontsize=7.5, fontfamily='monospace')
        handles.append(mpatches.Patch(color='#ef4444', label=f'Seizure t={seizure_start}s'))

    ax.set_xlabel('Time (s)', color='#4a6a8a', fontsize=9)
    ax.set_ylabel('η', color='#4a6a8a', fontsize=9)
    ax.set_title(f'Hawkes η — {channel}', color='#8aa5c4', fontsize=10,
                 pad=10, fontfamily='monospace')
    ax.set_ylim(0, 1.3)
    ax.tick_params(colors='#2d4460', labelsize=8)
    for s in ax.spines.values():
        s.set_color('#1e2d42')
    ax.legend(handles=handles, loc='upper left', framealpha=0.15,
              facecolor='#0f1623', edgecolor='#1e2d42',
              fontsize=8, labelcolor='#8aa5c4')
    ax.grid(axis='y', color='#1e2d42', linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    return fig


# ── UI ──────────────────────────────────────────────────────

st.markdown("""
<div class="title-block">
  <h1>⚡ HAWKES SEIZURE DETECTION</h1>
  <p>Stochastic Modeling for Early Seizure Warning using EEG Data &nbsp;·&nbsp; Rishika Singh</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Instructions</div>', unsafe_allow_html=True)
st.markdown("""
<div class="instr-card">
  <ol>
    <li><strong>Download</strong> an EEG <code>.edf</code> file from the
      <a href="https://physionet.org/content/chbmit/1.0.0/" target="_blank">PhysioNet CHB-MIT dataset</a></li>
    <li><strong>Upload</strong> the EEG file below</li>
    <li><strong>Select</strong> the EEG channel to analyse</li>
    <li><strong>Optionally</strong> enter the seizure start time from <code>summary.txt</code></li>
    <li><strong>Click</strong> Run Analysis</li>
  </ol>
  <div class="instr-sub">
    <strong>How to find seizure time:</strong><br>
    Open the <code>summary.txt</code> file for that patient and look for:<br>
    <code>Seizure Start Time: XXXX seconds</code><br>
    Enter that value in the input field below.
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Input</div>', unsafe_allow_html=True)

col_up, col_ch, col_sz = st.columns([2, 2, 1.2])

with col_up:
    uploaded = st.file_uploader("Upload EEG (.edf)", type=["edf"])

raw = None
channels_available = []

if uploaded:
    try:
        import mne
        with tempfile.NamedTemporaryFile(suffix='.edf', delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        raw = mne.io.read_raw_edf(tmp_path, preload=True, verbose=False)
        channels_available = raw.ch_names
        st.success(f"EEG loaded — {len(channels_available)} channels · "
                   f"{raw.times[-1]:.0f}s duration")
    except Exception as e:
        st.error(f"Could not load file: {e}")

with col_ch:
    selected_channel = st.selectbox(
        "Select Channel",
        options=channels_available if channels_available else ["Upload EEG first"],
        disabled=not channels_available
    )

with col_sz:
    sz_input = st.number_input("Seizure Start Time (s)", min_value=0,
                                max_value=100000, value=0, step=1,
                                help="Leave at 0 if unknown")
    seizure_start = int(sz_input) if sz_input > 0 else None

st.markdown("<br>", unsafe_allow_html=True)
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("▶  Run Analysis", disabled=(raw is None))

if run and raw is not None:
    T_total = raw.times[-1]
    bar = st.progress(0, text="Extracting spike train...")

    spikes = eeg_to_spikes(raw, channel=selected_channel)

    if len(spikes) < 30:
        st.warning("Too few spikes on this channel — try another channel.")
        bar.empty()
        st.stop()

    bar.progress(0.3, text=f"Fitting Hawkes MLE — {len(spikes)} spikes...")

    (hyp_time, conf_time, status, prob,
     rejected, centers, etas) = run_detection(
        spikes, T_total,
        progress_cb=lambda v: bar.progress(v, text="Running two-stage detection...")
    )

    crit_mask = etas >= 1.0
    crit_time = centers[crit_mask][0] if crit_mask.any() else None

    bar.progress(1.0, text="Complete.")
    bar.empty()

    # detection summary
    st.markdown('<div class="section-label">Detection Summary</div>',
                unsafe_allow_html=True)

    no_detection = (hyp_time is None and not rejected)
    if no_detection:
        badge_cls, display = 'badge-uncertain', 'NO DETECTION'
    else:
        badge_cls = {'confirmed': 'badge-confirmed',
                     'rejected':  'badge-rejected',
                     'uncertain': 'badge-uncertain'}.get(status, 'badge-uncertain')
        display = status.upper()

    st.markdown(f'<span class="badge {badge_cls}">{display}</span>',
                unsafe_allow_html=True)

    if hyp_time and status == 'confirmed':
        st.markdown(f"""
        <div class="result-grid">
          <div class="result-card">
            <div class="label">Confidence Score</div>
            <div class="value">{prob:.2f}<span class="unit">P(seizure)</span></div>
          </div>
          <div class="result-card">
            <div class="label">Early Warning Time</div>
            <div class="value">{hyp_time:.0f}<span class="unit">s</span></div>
          </div>
          <div class="result-card">
            <div class="label">Confirmation Time</div>
            <div class="value">{conf_time:.0f}<span class="unit">s</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

    elif rejected:
        st.markdown(f"""
        <div class="msg-box">
          Transient neural activity detected and automatically rejected
          by probabilistic verification.<br>
          Rejected at t={rejected[-1]:.0f}s &nbsp;·&nbsp; Final P(seizure) = {prob:.2f}
        </div>""", unsafe_allow_html=True)

    elif hyp_time and status == 'uncertain':
        st.markdown(f"""
        <div class="result-grid">
          <div class="result-card">
            <div class="label">Hypothesis Raised</div>
            <div class="value">{hyp_time:.0f}<span class="unit">s</span></div>
          </div>
          <div class="result-card">
            <div class="label">Final Probability</div>
            <div class="value">{prob:.2f}</div>
          </div>
          <div class="result-card">
            <div class="label">Status</div>
            <div class="value" style="font-size:0.9rem;color:#fbbf24">Uncertain</div>
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        st.markdown('<div class="msg-box">No significant detection found on this channel.</div>',
                    unsafe_allow_html=True)

    if crit_time:
        st.info(f"⚡ Supercritical alert (η ≥ 1.0) detected at t={crit_time:.0f}s")

    # validation
    if seizure_start and hyp_time:
        st.markdown('<div class="section-label">Validation</div>',
                    unsafe_allow_html=True)
        lead = seizure_start - hyp_time
        lead_cls   = "lead-positive" if lead > 0 else "lead-negative"
        lead_label = "Early detection ✓" if lead > 0 else "Late detection ✗"
        st.markdown(f"""
        <div class="val-block">
          <div class="val-row">
            <span class="vk">Actual Seizure Time</span>
            <span class="vv">{seizure_start}s</span>
          </div>
          <div class="val-row">
            <span class="vk">Predicted Warning Time</span>
            <span class="vv">{hyp_time:.0f}s</span>
          </div>
          <div class="val-row">
            <span class="vk">Lead Time</span>
            <span class="vv {lead_cls}">{lead:+.0f}s &nbsp;—&nbsp; {lead_label}</span>
          </div>
        </div>""", unsafe_allow_html=True)

    # plot
    st.markdown('<div class="section-label">η Timeseries</div>',
                unsafe_allow_html=True)
    fig = make_plot(centers, etas, hyp_time, conf_time,
                    seizure_start, selected_channel)
    st.pyplot(fig, use_container_width=True)

    with st.expander("Spike train statistics"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spikes", len(spikes))
        c2.metric("Mean Firing Rate", f"{len(spikes)/T_total:.3f} /s")
        c3.metric("η Windows", len(centers))
