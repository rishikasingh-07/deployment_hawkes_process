import numpy as np
from scipy.optimize import minimize
import mne

# ============================================================
# HAWKES MLE
# ============================================================

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
            hawkes_log_likelihood, x0,
            args=(events, T),
            method='L-BFGS-B',
            bounds=[(1e-6, None), (1e-6, None), (1e-6, None)],
            options={'maxiter': 1000}
        )

        if result.fun < best_ll:
            best_ll = result.fun
            best_result = result

    mu, alpha, beta = best_result.x
    return {
        'mu': mu,
        'alpha': alpha,
        'beta': beta,
        'eta': alpha / beta
    }


# ============================================================
# SLIDING WINDOW
# ============================================================

def sliding_window_eta(events, T_total,
                       window_size=200,
                       step_size=50,
                       min_spikes=15):

    window_centers = []
    recovered_etas = []
    start = 0

    while start + window_size <= T_total:
        end = start + window_size
        mask = (events >= start) & (events < end)
        window_events = events[mask] - start

        if len(window_events) >= min_spikes:
            result = fit_hawkes(window_events, T=window_size)
            recovered_etas.append(result['eta'])
            window_centers.append(start + window_size / 2)

        start += step_size

    return np.array(window_centers), np.array(recovered_etas)


# ============================================================
# DETECTION
# ============================================================

def probabilistic_verification(alert_centers, alert_etas,
                              alert_threshold,
                              p_initial=0.5,
                              p_confirm=0.85,
                              p_reject=0.15):

    p = p_initial

    for i in range(len(alert_etas)):
        e = alert_etas[i]

        if e > alert_threshold:
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

    centers_normal, etas_normal = sliding_window_eta(
        events, T_total,
        window_size=normal_window,
        step_size=normal_step
    )

    valid_mask = centers_normal > 300
    valid_centers = centers_normal[valid_mask]
    valid_etas = etas_normal[valid_mask]

    hypothesis_time = None
    hyp_count = 0

    for i in range(baseline_window, len(valid_etas)):

        recent = valid_etas[i - baseline_window:i]
        baseline = np.mean(recent)
        std = np.std(recent)
        effective_std = max(std, 0.05)
        threshold = baseline + z_score * effective_std

        is_elevated = valid_etas[i] > threshold
        is_rising = valid_etas[i] > valid_etas[i - 2]
        recent_short = valid_etas[max(0, i - 10):i]
        not_recovering = np.min(recent_short) > suppression_floor

        if is_elevated and is_rising and not_recovering:
            hyp_count += 1
        else:
            hyp_count = 0

        if hyp_count >= hypothesis_consecutive:

            hypothesis_time = valid_centers[i - hypothesis_consecutive + 1]

            centers_alert, etas_alert = sliding_window_eta(
                events, T_total,
                window_size=alert_window,
                step_size=alert_step
            )

            alert_mask = centers_alert > hypothesis_time
            alert_centers = centers_alert[alert_mask]
            alert_etas = etas_alert[alert_mask]

            if len(alert_centers) < 2:
                return hypothesis_time, None, 'uncertain', 0.5, []

            verify_time, status, final_p = probabilistic_verification(
                alert_centers, alert_etas,
                threshold,
                p_confirm=p_confirm,
                p_reject=p_reject
            )

            return hypothesis_time, verify_time, status, final_p, []

    return None, None, 'uncertain', 0.5, []


# ============================================================
# EEG PREPROCESSING
# ============================================================

def eeg_to_spikes(raw, channel,
                  low_freq=80,
                  high_freq=120,
                  threshold_std=3.0):

    sfreq = raw.info['sfreq']

    raw_filtered = raw.copy().filter(
        low_freq, high_freq, picks=[channel], verbose=False
    )

    data = raw_filtered.get_data()[raw.ch_names.index(channel)]

    window_samples = int(60 * sfreq)
    normalized = np.zeros_like(data)

    for i in range(0, len(data), window_samples):
        chunk = data[i:i + window_samples]
        if np.std(chunk) > 0:
            normalized[i:i + window_samples] = (
                (chunk - np.mean(chunk)) / np.std(chunk)
            )

    above = np.abs(normalized) > threshold_std
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]

    if len(crossings) == 0:
        return np.array([])

    spike_times = crossings / sfreq

    filtered = [spike_times[0]]
    for t in spike_times[1:]:
        if t - filtered[-1] > 0.05:
            filtered.append(t)

    return np.array(filtered)
