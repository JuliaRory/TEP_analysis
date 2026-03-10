from scipy.signal import find_peaks, savgol_filter
import numpy as np


def compute_gfp(teps):
    """Compute Global Field Power."""
    return np.std(teps, axis=1)

def extract_gfp_windows(
    TEPs,
    times,
    windows,
    mode="mean",          # "mean" or "peak"
    sfreq=5000,
    smooth_ms=10,
):
    """
    Extract GFP amplitude inside predefined latency windows.

    Parameters
    ----------
    TEPs : array [samples, channels]
    times : array (ms)
    windows : dict
        e.g. {"P60": (50,80), "N100": (80,140)}
    mode : str
        "mean" (recommended) or "peak"
    """

    # ---------- GFP ----------
    gfp = compute_gfp(TEPs)

    # ---------- smoothing ----------
    win = int(smooth_ms * sfreq / 1000)
    win = win + 1 if win % 2 == 0 else win
    gfp_smooth = savgol_filter(gfp, win, polyorder=3)

    results = {}

    # ---------- window extraction ----------
    for name, (tmin, tmax) in windows.items():

        mask = (times >= tmin) & (times <= tmax)

        if not np.any(mask):
            results[name] = dict(
                amplitude=np.nan,
                latency=np.nan,
                index=None
            )
            continue

        gfp_win = gfp_smooth[mask]
        time_win = times[mask]
        idx_win = np.where(mask)[0]

        # ===== MODE: MEAN (recommended) =====
        if mode == "mean":
            amp = np.mean(gfp_win)
            lat = np.mean(time_win)
            idx = None

        # ===== MODE: PEAK (forced peak) =====
        elif mode == "peak":
            local_idx = np.argmax(gfp_win)
            amp = gfp_win[local_idx]
            lat = time_win[local_idx]
            idx = idx_win[local_idx]

        else:
            raise ValueError("mode must be 'mean' or 'peak'")

        results[name] = dict(
            amplitude=amp,
            latency=lat,
            index=idx
        )

    return results, gfp_smooth


def find_gfp_peaks(
    TEPs,
    times,
    start_sample = 1000, 
    sfreq=5000,
    smooth_ms=10
):
    """
    Robust GFP peak detection.
    
    Parameters
    ----------
    TEPs : [samples, channels]
    times: ms
    smooth_ms : smoothing window (ms)
    min_distance_ms : minimal distance between peaks
    prominence_factor : peak prominence relative to noise
    width_ms : minimal peak width
    """


    # --- GFP ---
    

    gfp = compute_gfp(TEPs)

    # --- smoothing (Savitzky-Golay preserves peaks) ---
    win = int(smooth_ms * sfreq / 1000)
    win = win + 1 if win % 2 == 0 else win  # must be odd
    gfp_smooth = savgol_filter(gfp, win, polyorder=3)

    gfp_cut  = gfp_smooth[start_sample:]
    times_cut = times[start_sample:]

    # адаптивный prominence
    dynamic_range = gfp_cut .max() - gfp_cut .min()
    prominence = 0.07 * dynamic_range  # 5% от размаха
    height = np.mean(gfp_cut )
    distance = int(0.015 * sfreq)  # 15 ms

    
    peaks, props = find_peaks(
        gfp_cut,
        prominence=prominence,
        height=height,
        distance=distance
    )
    
    peaks_full = peaks + start_sample

    t_peaks = times[peaks_full]
    a_peaks = gfp_cut[peaks]

    return t_peaks, a_peaks, gfp_smooth, peaks, props, peaks_full
