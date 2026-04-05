import numpy as np
from scipy.optimize import minimize
import mne


# ================= MLE =================

def hawkes_log_likelihood(params, events, T):
    mu, alpha, beta = params

    if mu <= 0 or alpha <= 0 or beta <= 0:
        return 1e10

    events = np.array(events)

    dt = events[:, None] - events[None, :]
    mask = dt > 0

    exponent = np.where(mask, -beta * dt, -np.inf)
    exponent = np.clip(exponent, -500, 0)

    excitation = np.where(mask, alpha * np.exp(exponent), 0)
    intensities = mu + excitation.sum(axis=1)

    if np.any(intensities <= 0):
        return 1e10

    part1 = np.sum(np.log(intensities))
    part2 = mu * T + (alpha / beta) * np.sum(
        1 - np.exp(-beta * (T - events))
    )

    return -(part1 - part2)


def fit_hawkes(events, T, n_restarts=10):

    best_result = None
    best_ll = np.inf

    for _ in range(n_restarts):

        x0 = [
            np.random.uniform(0.1, 2.0),
            np.random.uniform(0.1, 0.9),
            np.random.uniform(0.5, 5.0)
        ]

        result = minimize(
            hawkes_log_likelihood,
            x0,
            args=(events, T),
            method='L-BFGS-B',
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
            options={'maxiter': 1000}
        )

        if result.fun < best_ll:
            best_ll = result.fun
            best_result = result

    mu, alpha, beta = best_result.x

    return {'mu': mu, 'alpha': alpha, 'beta': beta,
            'eta': alpha / beta}


# ================= SLIDING WINDOW =================

def sliding_window_eta(events, T_total,
                       window_size=200, step_size=50,
                       min_spikes=15):

    centers = []
    etas = []

    start = 0

    while start + window_size <= T_total:

        end = start + window_size

        mask = (events >= start) & (events < end)
        window_events = events[mask] - start

        if len(window_events) >= min_spikes:
            result = fit_hawkes(window_events, T=window_size)
            centers.append(start + window_size / 2)
            etas.append(result['eta'])

        start += step_size

    return np.array(centers), np.array(etas)


# ================= DETECTION =================

def probabilistic_verification(alert_centers, alert_etas,
                              threshold,
                              p_initial=0.5,
                              p_confirm=0.85,
                              p_reject=0.15):

    p = p_initial

    for i in range(len(alert_etas)):

        if alert_etas[i] > threshold:
            p = p + (1 - p) * 0.4
        else:
            p = p - p * 0.4

        if p >= p_confirm:
            return alert_centers[i], 'confirmed', p

        if p <= p_reject:
            return alert_centers[i], 'rejected', p

    return None, 'uncertain', p


def adaptive_window_detection(events, T_total,
                               normal_window=200,
                               normal_step=50,
                               alert_window=100,
                               alert_step=20,
                               baseline_window=20,
                               z_score=1.5,
                               hypothesis_consecutive=2,
                               suppression_floor=0.3,
                               p_confirm=0.85,
                               p_reject=0.15):

    centers, etas = sliding_window_eta(events, T_total,
                                       normal_window, normal_step)

    mask = centers > 300
    centers = centers[mask]
    etas = etas[mask]

    hypothesis_time = None
    in_hypothesis = False
    count = 0
    rejected = []

    for i in range(baseline_window, len(etas)):

        recent = etas[i - baseline_window:i]
        baseline = np.mean(recent)
        std = max(np.std(recent), 0.05)

        threshold = baseline + z_score * std

        cond1 = etas[i] > threshold
        cond2 = etas[i] > etas[i-2]
        cond3 = np.min(etas[max(0, i-10):i]) > suppression_floor

        if cond1 and cond2 and cond3:
            count += 1
        else:
            count = 0

        if count >= hypothesis_consecutive:

            hypothesis_time = centers[i-1]

            alert_centers, alert_etas = sliding_window_eta(
                events, T_total, alert_window, alert_step
            )

            mask2 = alert_centers > hypothesis_time
            alert_centers = alert_centers[mask2]
            alert_etas = alert_etas[mask2]

            verify_time, status, prob = probabilistic_verification(
                alert_centers, alert_etas, threshold
            )

            if status == "confirmed":
                return hypothesis_time, verify_time, status, prob, rejected

            elif status == "rejected":
                rejected.append(hypothesis_time)
                hypothesis_time = None
                count = 0

    return hypothesis_time, None, "uncertain", 0.5, rejected


# ================= EEG =================

def eeg_to_spikes(raw, channel):

    sfreq = raw.info['sfreq']

    raw_filtered = raw.copy().filter(80, 120, picks=[channel], verbose=False)
    data = raw_filtered.get_data()[raw.ch_names.index(channel)]

    window_samples = int(60 * sfreq)
    normalized = np.zeros_like(data)

    for i in range(0, len(data), window_samples):
        chunk = data[i:i + window_samples]
        if np.std(chunk) > 0:
            normalized[i:i + window_samples] = (
                (chunk - np.mean(chunk)) / np.std(chunk)
            )

    above = np.abs(normalized) > 3
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(crossings) == 0:
        return np.array([])

    spikes = crossings / sfreq

    filtered = [spikes[0]]
    for t in spikes[1:]:
        if t - filtered[-1] > 0.05:
            filtered.append(t)

    return np.array(filtered)
