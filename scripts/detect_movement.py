from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, iirnotch, sosfilt, tf2sos


FS = 5000
DEFAULT_DATA_DIR = Path("data/pilot/movement_MEPs_check")
DEFAULT_RESULTS_DIR = Path("results/movement_detection")


def ms_to_samples(ms: float, fs: int = FS) -> int:
    return int(ms * fs / 1000)


def get_trigger(data: np.ndarray, bit: int = 0) -> np.ndarray:
    ttl = np.asarray(data, dtype=np.uint8)
    return ((ttl >> bit) & 0b1).astype(int)


def calculate_tkeo(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    tkeo = np.zeros_like(x)
    tkeo[1:-1] = x[1:-1] ** 2 - x[:-2] * x[2:]
    tkeo[0] = tkeo[1]
    tkeo[-1] = tkeo[-2]
    return tkeo


def make_online_filters(fs: int = FS, wn: tuple[float, float] = (10, 450)):
    notch_fr = 50
    notch_width = 1
    q = notch_fr / notch_width
    b_notch, a_notch = iirnotch(notch_fr, q, fs=fs)
    sos_notch = tf2sos(b_notch, a_notch)
    sos_butter = butter(4, wn, btype="bandpass", output="sos", fs=fs)
    return sos_notch, sos_butter


def robust_noise_level(x: np.ndarray) -> tuple[float, float]:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        raise ValueError("Baseline interval is empty")

    median = np.median(x)
    mad = np.median(np.abs(x - median))
    sigma = 1.4826 * mad

    if sigma <= np.finfo(float).eps:
        sigma = max(
            np.std(x),
            np.percentile(x, 75) - np.percentile(x, 25),
            np.finfo(float).eps,
        )

    return float(median), float(sigma)


def smooth_boxcar(x: np.ndarray, time: np.ndarray, window_ms: float) -> np.ndarray:
    dt = np.median(np.diff(time))
    n_samples = max(1, int(round(window_ms / dt)))
    if n_samples <= 1:
        return x.copy()
    kernel = np.ones(n_samples) / n_samples
    return np.convolve(x, kernel, mode="same")


def get_epochs(
    emg: np.ndarray,
    trigger: np.ndarray,
    start_ms: float,
    end_ms: float,
    fs: int = FS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trigger_diff = np.diff(trigger)
    events = np.where(trigger_diff == 1)[0]

    start = ms_to_samples(start_ms, fs)
    end = ms_to_samples(end_ms, fs)
    time = np.linspace(start_ms, end_ms, end - start)

    epochs = []
    for timestamp in events:
        epoch = emg[timestamp + start : timestamp + end]
        if len(epoch) == len(time):
            epochs.append(epoch)

    epochs = np.asarray(epochs)
    tkeo_epochs = np.asarray([calculate_tkeo(epoch * 1e-3) for epoch in epochs])
    return time, epochs, tkeo_epochs


def detect_movement_in_epoch(
    time: np.ndarray,
    emg_tkeo: np.ndarray,
    baseline: tuple[float, float] = (-500, -300),
    art_limits: tuple[float, float] = (-1.5, 8),
    post_tms_ignore_until: float = 75,
    threshold_k: float = 15,
    baseline_percentile: float = 99.5,
    prominence_k: float = 4,
    smooth_ms: float = 3,
    min_width_ms: float = 2,
    min_distance_ms: float = 10,
    confirmation_window_ms: float = 8,
    required_fraction: float = 0.25,
    min_peak_area: float = 1.5e-10,
    better_candidate_area_ratio: float = 3.0,
    better_candidate_min_separation_ms: float = 40,
    pre_tms_ignore_after: float = -8,
    detect_pre_tms: bool = True,
) -> dict:
    time = np.asarray(time)
    emg_tkeo = np.asarray(emg_tkeo)

    baseline_mask = (time >= baseline[0]) & (time <= baseline[1])
    base = emg_tkeo[baseline_mask]
    noise_median, noise_sigma = robust_noise_level(base)

    threshold = max(
        noise_median + threshold_k * noise_sigma,
        np.percentile(base, baseline_percentile),
    )
    min_prominence = max(
        prominence_k * noise_sigma,
        0.5 * (threshold - noise_median),
    )

    valid_mask = time > post_tms_ignore_until
    if detect_pre_tms:
        valid_mask |= time < min(art_limits[0], pre_tms_ignore_after)

    signal_smooth = smooth_boxcar(emg_tkeo, time, smooth_ms)
    signal_for_peaks = signal_smooth.copy()
    signal_for_peaks[~valid_mask] = noise_median

    dt = np.median(np.diff(time))
    min_width_samples = max(1, int(round(min_width_ms / dt)))
    min_distance_samples = max(1, int(round(min_distance_ms / dt)))

    peaks, props = find_peaks(
        signal_for_peaks,
        height=threshold,
        prominence=min_prominence,
        width=min_width_samples,
        distance=min_distance_samples,
    )

    half_window = max(1, int(round((confirmation_window_ms / 2) / dt)))
    accepted = []
    accepted_prop_idxs = []
    onset_by_peak = []
    area_by_peak = []
    fraction_by_peak = []

    for prop_idx, peak_idx in enumerate(peaks):
        lo = max(0, peak_idx - half_window)
        hi = min(len(signal_smooth), peak_idx + half_window + 1)
        local_idxs = np.arange(lo, hi)
        local_idxs = local_idxs[valid_mask[local_idxs]]

        if local_idxs.size == 0:
            continue

        fraction = np.mean(signal_smooth[local_idxs] > threshold)
        if fraction < required_fraction:
            continue

        area_lo = max(0, peak_idx - 2 * half_window)
        area_hi = min(len(signal_smooth), peak_idx + 2 * half_window + 1)
        peak_area = np.trapezoid(
            np.maximum(signal_smooth[area_lo:area_hi] - threshold, 0),
            time[area_lo:area_hi],
        )
        if peak_area < min_peak_area:
            continue

        onset_idx = peak_idx
        while (
            onset_idx > 0
            and valid_mask[onset_idx - 1]
            and signal_smooth[onset_idx - 1] > threshold
        ):
            onset_idx -= 1

        accepted.append(peak_idx)
        accepted_prop_idxs.append(prop_idx)
        onset_by_peak.append(onset_idx)
        area_by_peak.append(float(peak_area))
        fraction_by_peak.append(float(fraction))

    movement_found = len(accepted) > 0
    onset_time = np.nan
    peak_time = np.nan
    peak_amp = np.nan
    peak_prominence = np.nan
    peak_area = np.nan
    peak_times: list[float] = []
    peak_onsets: list[float] = []
    peak_amps: list[float] = []
    peak_areas: list[float] = []
    fraction_max = 0 if len(fraction_by_peak) == 0 else np.max(fraction_by_peak)

    if movement_found:
        peak_times = time[accepted].astype(float).tolist()
        peak_onsets = time[onset_by_peak].astype(float).tolist()
        peak_amps = signal_smooth[accepted].astype(float).tolist()
        peak_areas = area_by_peak

        selected_pos = 0
        for candidate_pos in range(1, len(accepted)):
            separated_enough = (
                time[accepted[candidate_pos]] - time[accepted[selected_pos]]
                >= better_candidate_min_separation_ms
            )
            much_stronger = (
                area_by_peak[candidate_pos]
                >= area_by_peak[selected_pos] * better_candidate_area_ratio
            )
            if separated_enough and much_stronger:
                selected_pos = candidate_pos
                break

        peak_idx = accepted[selected_pos]
        prop_idx = accepted_prop_idxs[selected_pos]
        peak_time = float(time[peak_idx])
        peak_amp = float(signal_smooth[peak_idx])
        peak_prominence = float(props["prominences"][prop_idx])
        peak_area = float(area_by_peak[selected_pos])
        onset_time = float(time[onset_by_peak[selected_pos]])

    return {
        "movement_found": bool(movement_found),
        "onset_time": onset_time,
        "peak_time": peak_time,
        "peak_amp": peak_amp,
        "peak_prominence": peak_prominence,
        "peak_area": peak_area,
        "peak_times": json.dumps(peak_times),
        "peak_onsets": json.dumps(peak_onsets),
        "peak_amps": json.dumps(peak_amps),
        "peak_areas": json.dumps(peak_areas),
        "threshold": float(threshold),
        "noise_median": float(noise_median),
        "noise_sigma": float(noise_sigma),
        "fraction_max": float(fraction_max),
        "n_peaks": int(len(accepted)),
    }


def plot_epoch_segments(
    ax,
    time: np.ndarray,
    data: np.ndarray,
    mep_limits: tuple[float, float],
    art_limits: tuple[float, float],
    color: str = "tab:blue",
):
    signal_masks = [
        time < art_limits[0],
        (time > art_limits[1]) & (time < mep_limits[0]),
        time > mep_limits[1],
    ]

    for mask in signal_masks:
        ax.plot(time[mask], data[mask], color=color, lw=1)

    art_mask = (time >= art_limits[0]) & (time <= art_limits[1])
    ax.plot(time[art_mask], data[art_mask], lw=0.8, color="darkgrey", alpha=0.7)

    mep_mask = (time >= mep_limits[0]) & (time <= mep_limits[1])
    ax.plot(time[mep_mask], data[mep_mask], lw=0.8, color="darkgrey", alpha=0.7)


def save_epoch_figure(
    out_path: Path,
    record_name: str,
    epoch_idx: int,
    time: np.ndarray,
    emg_epoch: np.ndarray,
    tkeo_epoch: np.ndarray,
    result: pd.Series,
    mep_limits: tuple[float, float],
    art_limits: tuple[float, float],
):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4), constrained_layout=True)
    fig.suptitle(
        f"{record_name} | epoch {epoch_idx} | movement={bool(result['movement_found'])}",
        fontsize=12,
    )

    ax = axes[0]
    plot_epoch_segments(ax, time, tkeo_epoch, mep_limits=mep_limits, art_limits=art_limits)
    ax.axhline(result["threshold"], color="black", lw=1, label="threshold")
    if np.isfinite(result["onset_time"]):
        ax.axvline(result["onset_time"], color="red", lw=1, label="onset")
    if np.isfinite(result["peak_time"]):
        ax.axvline(result["peak_time"], color="tab:green", lw=1, ls="--", label="peak")
    ax.set_xlim(-200, 255)
    ax.set_ylim(0, max(12e-10, result["threshold"] * 1.4))
    area_text = "nan" if not np.isfinite(result["peak_area"]) else f"{result['peak_area']:.2e}"
    ax.set_title(
        f"TKEO | thr {result['threshold']:.2e} | "
        f"area {area_text} | peaks {int(result['n_peaks'])}"
    )
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("[mV2]")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    ax = axes[1]
    plot_epoch_segments(ax, time, emg_epoch, mep_limits=mep_limits, art_limits=art_limits)
    if np.isfinite(result["onset_time"]):
        ax.axvline(result["onset_time"], color="red", lw=1, label="onset")
    if np.isfinite(result["peak_time"]):
        ax.axvline(result["peak_time"], color="tab:green", lw=1, ls="--", label="peak")
    ax.set_xlim(-200, 255)
    ax.set_ylim(-3, 3)
    ax.set_title("Filtered EMG")
    ax.set_xlabel("time [ms]")
    ax.set_ylabel("[mV]")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="upper right")

    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def process_hdf(
    hdf_path: Path,
    output_dir: Path,
    start_ms: float = -500,
    end_ms: float = 300,
    art_limits: tuple[float, float] = (-1.5, 8),
    mep_limits: tuple[float, float] = (15, 75),
) -> pd.DataFrame:
    record_dir = output_dir / hdf_path.stem
    figures_dir = record_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf_path, "r") as h5f:
        data = h5f["eeg/data"][:-1]

    emg = -data[:, 0]
    trigger = get_trigger(data[:, -1])

    sos_notch, sos_butter = make_online_filters()
    emg_f = sosfilt(sos_notch, emg, axis=0)
    emg_f = sosfilt(sos_butter, emg_f, axis=0)

    time, epochs, tkeo_epochs = get_epochs(
        emg_f * 1e3,
        trigger,
        start_ms=start_ms,
        end_ms=end_ms,
    )

    rows = []
    for epoch_idx, tkeo_epoch in enumerate(tkeo_epochs, start=1):
        result = detect_movement_in_epoch(
            time=time,
            emg_tkeo=tkeo_epoch,
            baseline=(-500, -300),
            art_limits=art_limits,
            post_tms_ignore_until=mep_limits[1],
            threshold_k=15,
            baseline_percentile=99.5,
            prominence_k=4,
            smooth_ms=3,
            min_width_ms=2,
            min_distance_ms=10,
            confirmation_window_ms=8,
            required_fraction=0.25,
            min_peak_area=1.5e-10,
            better_candidate_area_ratio=3.0,
            better_candidate_min_separation_ms=40,
            pre_tms_ignore_after=-8,
            detect_pre_tms=True,
        )
        result["record"] = hdf_path.name
        result["n_epoch"] = epoch_idx
        rows.append(result)

    df_results = pd.DataFrame(rows)
    first_cols = ["record", "n_epoch", "movement_found", "onset_time", "peak_time"]
    other_cols = [col for col in df_results.columns if col not in first_cols]
    df_results = df_results[first_cols + other_cols]

    table_path = record_dir / f"{hdf_path.stem}_movement_detection.csv"
    df_results.to_csv(table_path, index=False)

    for row_idx, row in df_results.iterrows():
        epoch_idx = int(row["n_epoch"])
        out_path = figures_dir / f"{hdf_path.stem}_epoch_{epoch_idx:03d}.png"
        save_epoch_figure(
            out_path=out_path,
            record_name=hdf_path.name,
            epoch_idx=epoch_idx,
            time=time,
            emg_epoch=epochs[row_idx],
            tkeo_epoch=tkeo_epochs[row_idx],
            result=row,
            mep_limits=mep_limits,
            art_limits=art_limits,
        )

    return df_results


def run(
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_RESULTS_DIR,
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf_files = sorted(data_dir.glob("*.hdf"))
    if not hdf_files:
        raise FileNotFoundError(f"No .hdf files found in {data_dir}")

    all_results = []
    for hdf_path in hdf_files:
        print(f"Processing {hdf_path}")
        df = process_hdf(hdf_path, output_dir)
        all_results.append(df)
        print(
            f"  saved {len(df)} figures and table; "
            f"movements={int(df['movement_found'].sum())}/{len(df)}"
        )

    df_all = pd.concat(all_results, ignore_index=True)
    df_all.to_csv(output_dir / "movement_detection_all_records.csv", index=False)
    return df_all


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect movement in pilot MEP HDF files.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
