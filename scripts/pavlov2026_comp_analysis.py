from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "exp"
EXTERNAL_DATA_DIR = Path(r"D:\2025 - TEP\data - trans\CLEAN_EPOCHS_UI")
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "pavlov2026"

SESSION_BY_SUBJECT = {
    "01AV": 38,
    "04KK": 39,
    "02NS": 41,
    "03AZ": 40,
    "07IK": 42,
    "05UB": 43,
    "08EN": 45,
    "10ES": 52,
}

RUN_CONDS = ("real", "MI")
SPOTS = ("M1_PA", "M1_AP", "PPC_L")
INNER_CONDITIONS = ("rest", "onset", "after")

STIMULUS_TO_CONDITION = {
    "rest1500_tms_0ms_bar.mkv": "rest",
    "animatedSingle1500_tms_0ms_nosounds_bar.mkv": "onset",
    "animatedSingle1500_tms_+200ms_nosounds_bar.mkv": "after",
}

COMP_WINDOWS = {
    "n20": (10.0, 25.0),
    "p30": (20.0, 40.0),
    "n45": (40.0, 55.0),
    "p60": (55.0, 70.0),
    "n100": (80.0, 120.0),
}

COMP_FUNS = {
    "n20": np.nanmin,
    "p30": np.nanmax,
    "n45": np.nanmin,
    "p60": np.nanmax,
    "n100": np.nanmin,
}

COMP_COLORS = {
    "n20": "#d62728",
    "p30": "#1f77b4",
    "n45": "#2ca02c",
    "p60": "#9467bd",
    "n100": "#ff7f0e",
}


@dataclass(frozen=True)
class SoundRecord:
    path: Path
    session_id: int
    original_file: str


def resolve_default_data_dir() -> Path:
    if EXTERNAL_DATA_DIR.exists():
        return EXTERNAL_DATA_DIR
    return LOCAL_DATA_DIR


def decode_matlab_string_payload(ds: h5py.Dataset) -> np.ndarray:
    arr = np.asarray(ds[()]).ravel().astype(np.uint64)

    ndims = int(arr[1])
    shape = tuple(int(x) for x in arr[2 : 2 + ndims])
    n_items = int(np.prod(shape))

    lengths_start = 2 + ndims
    lengths = [int(x) for x in arr[lengths_start : lengths_start + n_items]]
    packed = arr[lengths_start + n_items :]

    codes: list[int] = []
    for value in packed:
        codes.extend(np.frombuffer(int(value).to_bytes(8, "little"), dtype="<u2").tolist())

    strings = []
    pos = 0
    for length in lengths:
        strings.append("".join(chr(c) for c in codes[pos : pos + length]))
        pos += length

    return np.array(strings, dtype=object).reshape(shape, order="F")


def read_matlab_string(h5f: h5py.File, dataset_path: str) -> np.ndarray:
    desc = np.asarray(h5f[dataset_path][()]).ravel()
    mcos_index = int(desc[4]) + 1
    ref = h5f["#subsystem#/MCOS"][()].ravel()[mcos_index]
    return decode_matlab_string_payload(h5f[ref])


def load_channel_colors() -> list[str]:
    ced_file = PROJECT_ROOT / "resources" / "mks64_standard.ced"
    colors_file = PROJECT_ROOT / "resources" / "channel_colors.json"

    with colors_file.open("r", encoding="utf-8") as f:
        color_by_channel = json.load(f)

    montage = pd.read_csv(ced_file, sep="\t")
    return montage["labels"].map(color_by_channel).fillna("#777777").tolist()


def load_channel_names() -> list[str]:
    ced_file = PROJECT_ROOT / "resources" / "mks64_standard.ced"
    montage = pd.read_csv(ced_file, sep="\t")
    return montage["labels"].astype(str).tolist()


def parse_session_id(path: Path) -> int | None:
    match = re.search(r"session_(\d+)__", path.name)
    if not match:
        return None
    return int(match.group(1))


def discover_records(data_dir: Path) -> list[SoundRecord]:
    records: list[SoundRecord] = []
    for path in sorted(data_dir.glob("*SOUND.mat")):
        session_id = parse_session_id(path)
        if session_id is None:
            print(f"Skip {path.name}: cannot parse session id")
            continue

        try:
            with h5py.File(path, "r") as h5f:
                original = read_matlab_string(
                    h5f,
                    "cleanedResult/sourceDatasets/OriginalFile",
                ).ravel(order="F")[0]
        except Exception as exc:
            print(f"Skip {path.name}: cannot read OriginalFile ({exc})")
            continue

        records.append(SoundRecord(path=path, session_id=session_id, original_file=str(original)))

    return records


def stimulus_to_condition(stimulus: str) -> str:
    if stimulus in STIMULUS_TO_CONDITION:
        return STIMULUS_TO_CONDITION[stimulus]
    if "rest" in stimulus:
        return "rest"
    if "+200ms" in stimulus:
        return "after"
    if "0ms" in stimulus:
        return "onset"
    return "unknown"


def load_record_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as h5f:
        tvec = np.asarray(h5f["cleanedResult/tvec"][:][0], dtype=float)
        epochs = np.asarray(h5f["cleanedResult/epochs_clean"][:], dtype=float)

        raw_labels = read_matlab_string(h5f, "cleanedResult/epoch_types")
        raw_labels = raw_labels.ravel(order="F").tolist()

    stimuli = []
    for label in raw_labels:
        try:
            stimuli.append(json.loads(label)["stimulus"])
        except (json.JSONDecodeError, KeyError, TypeError):
            stimuli.append(str(label))

    labels = np.asarray([stimulus_to_condition(stimulus) for stimulus in stimuli])
    if epochs.shape[1] != len(labels):
        raise ValueError(f"{path.name}: epochs/labels mismatch {epochs.shape[1]} != {len(labels)}")

    return tvec, epochs, labels


def matches_record(record: SoundRecord, run_cond: str, spot: str) -> bool:
    original = record.original_file.lower()
    return run_cond.lower() in original and spot.lower() in original


def collect_epochs(
    records: list[SoundRecord],
    subject: str,
    run_cond: str,
    spot: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[SoundRecord]]:
    session_id = SESSION_BY_SUBJECT[subject]
    matched = [
        record
        for record in records
        if record.session_id == session_id and matches_record(record, run_cond, spot)
    ]

    all_epochs = []
    all_labels = []
    tvec = None
    for record in matched:
        print(f"  {subject} session {session_id} {run_cond} {spot}: {record.path.name}")
        print(f"    original: {record.original_file}")
        record_tvec, epochs, labels = load_record_data(record.path)
        if tvec is None:
            tvec = record_tvec
        elif not np.allclose(tvec, record_tvec):
            raise ValueError(f"{record.path.name}: tvec differs from previous matched records")

        all_epochs.append(epochs)
        all_labels.append(labels)

    if not all_epochs:
        raise FileNotFoundError(f"No files for {subject}, session {session_id}, {run_cond}, {spot}")

    return (
        np.concatenate(all_epochs, axis=1),
        np.concatenate(all_labels, axis=0),
        tvec,
        matched,
    )


def select_component_points(
    teps: np.ndarray,
    tvec: np.ndarray,
    component: str,
    n_mean: int,
) -> tuple[list[dict], dict]:
    window = COMP_WINDOWS[component]
    window_idxs = np.where((tvec >= window[0]) & (tvec <= window[1]))[0]
    if window_idxs.size == 0:
        raise ValueError(f"No samples in component window {component}: {window}")

    data = teps[:, window_idxs]
    is_negative_component = component.startswith("n")

    channel_candidates = []
    for channel_idx, channel_data in enumerate(data):
        if not np.all(np.isfinite(channel_data)):
            continue

        local_peak_idxs, _ = find_peaks(-channel_data if is_negative_component else channel_data)
        if local_peak_idxs.size == 0:
            continue

        local_peak_values = channel_data[local_peak_idxs]
        if is_negative_component:
            directed_idxs = local_peak_idxs[local_peak_values < 0]
            directed_values = channel_data[directed_idxs]
            if directed_idxs.size == 0:
                continue
            best_pos = int(np.argmin(directed_values))
        else:
            directed_idxs = local_peak_idxs[local_peak_values > 0]
            directed_values = channel_data[directed_idxs]
            if directed_idxs.size == 0:
                continue
            best_pos = int(np.argmax(directed_values))

        local_idx = int(directed_idxs[best_pos])
        amplitude = float(channel_data[local_idx])
        channel_candidates.append(
            {
                "channel_idx": int(channel_idx),
                "local_idx": local_idx,
                "amplitude_uv": amplitude,
                "score": -amplitude if is_negative_component else amplitude,
            }
        )

    if not channel_candidates:
        summary = {
            "amplitude_abs_mean_topn_uv": np.nan,
            "amplitude_signed_mean_topn_uv": np.nan,
            "latency_mean_topn_ms": np.nan,
            "peak_amplitude_uv": np.nan,
            "peak_latency_ms": np.nan,
            "peak_channel_idx": np.nan,
            "n_selected": 0,
        }
        return [], summary

    channel_candidates = sorted(channel_candidates, key=lambda item: item["score"], reverse=True)
    selected_candidates = channel_candidates[: min(n_mean, len(channel_candidates))]

    points = []
    for rank, candidate in enumerate(selected_candidates, start=1):
        channel_idx = int(candidate["channel_idx"])
        time_idx = int(window_idxs[candidate["local_idx"]])
        points.append(
            {
                "selected_rank": rank,
                "channel_idx": channel_idx,
                "latency_ms": float(tvec[time_idx]),
                "amplitude_uv": float(candidate["amplitude_uv"]),
            }
        )

    amplitudes = np.asarray([point["amplitude_uv"] for point in points], dtype=float)
    latencies = np.asarray([point["latency_ms"] for point in points], dtype=float)
    summary = {
        "amplitude_abs_mean_topn_uv": float(np.mean(np.abs(amplitudes))),
        "amplitude_signed_mean_topn_uv": float(np.mean(amplitudes)),
        "latency_mean_topn_ms": float(np.mean(latencies)),
        "peak_amplitude_uv": float(points[0]["amplitude_uv"]),
        "peak_latency_ms": float(points[0]["latency_ms"]),
        "peak_channel_idx": int(points[0]["channel_idx"]),
        "n_selected": int(len(points)),
    }
    return points, summary


def analyze_selection(
    subject: str,
    run_cond: str,
    spot: str,
    epochs: np.ndarray,
    labels: np.ndarray,
    tvec: np.ndarray,
    channel_names: list[str],
    n_mean: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray], dict[str, dict[str, list[dict]]]]:
    detail_rows = []
    summary_rows = []
    teps_by_condition = {}
    points_by_condition = {}

    for inner_condition in INNER_CONDITIONS:
        epoch_mask = labels == inner_condition
        if not np.any(epoch_mask):
            print(f"    no epochs for inner condition {inner_condition}")
            continue

        teps = np.mean(epochs[:, epoch_mask, :], axis=1)
        teps_by_condition[inner_condition] = teps
        points_by_condition[inner_condition] = {}

        for component in COMP_WINDOWS:
            points, summary = select_component_points(teps, tvec, component, n_mean=n_mean)
            points_by_condition[inner_condition][component] = points

            row_base = {
                "subject": subject,
                "session": SESSION_BY_SUBJECT[subject],
                "run_cond": run_cond,
                "spot": spot,
                "condition": inner_condition,
                "component": component,
                "window_start_ms": COMP_WINDOWS[component][0],
                "window_end_ms": COMP_WINDOWS[component][1],
            }

            summary_row = {**row_base, **summary}
            if np.isfinite(summary["peak_channel_idx"]):
                summary_row["peak_channel"] = channel_names[int(summary["peak_channel_idx"])]
            else:
                summary_row["peak_channel"] = np.nan
            summary_rows.append(summary_row)

            for point in points:
                detail_row = {**row_base, **point}
                detail_row["channel"] = channel_names[point["channel_idx"]]
                detail_rows.append(detail_row)

    return (
        pd.DataFrame(detail_rows),
        pd.DataFrame(summary_rows),
        teps_by_condition,
        points_by_condition,
    )


def save_selection_figure(
    output_path: Path,
    subject: str,
    run_cond: str,
    spot: str,
    tvec: np.ndarray,
    teps_by_condition: dict[str, np.ndarray],
    points_by_condition: dict[str, dict[str, list[dict]]],
    channel_colors: list[str],
    y_limit: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)
    fig.suptitle(
        f"{subject} | session {SESSION_BY_SUBJECT[subject]} | {run_cond} | {spot}",
        fontsize=16,
    )

    for ax, inner_condition in zip(axes, INNER_CONDITIONS):
        teps = teps_by_condition.get(inner_condition)
        if teps is None:
            ax.set_title(f"{inner_condition}: no epochs", fontsize=13)
            ax.axis("off")
            continue

        for color, tep in zip(channel_colors, teps):
            ax.plot(tvec, tep, color=color, linewidth=0.75, alpha=0.9)

        for component, points in points_by_condition[inner_condition].items():
            xs = [point["latency_ms"] for point in points]
            ys = [point["amplitude_uv"] for point in points]
            ax.scatter(
                xs,
                ys,
                s=82,
                color=COMP_COLORS[component],
                edgecolor="black",
                linewidth=0.9,
                zorder=10,
                label=component,
            )
            if points:
                ax.text(
                    points[0]["latency_ms"],
                    points[0]["amplitude_uv"],
                    f" {component}",
                    color="black",
                    fontsize=9,
                    weight="bold",
                    va="center",
                    zorder=11,
                )

        ax.axvline(0, color="black", linewidth=1.5)
        ax.set_xlim(5, 150)
        ax.set_ylim(-y_limit, y_limit)
        ax.grid(color="lightgrey", linewidth=0.8)
        ax.set_title(inner_condition, fontsize=14)
        ax.set_xlabel("Время [мс]", fontsize=12)

    axes[0].set_ylabel("сигнал ЭЭГ [мкВ]", fontsize=12)
    handles, labels = axes[-1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="lower center",
            ncol=len(by_label),
            frameon=False,
        )

    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def process_selection(
    records: list[SoundRecord],
    output_dir: Path,
    subject: str,
    run_cond: str,
    spot: str,
    channel_names: list[str],
    channel_colors: list[str],
    n_mean: int,
    y_limit: float,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    try:
        epochs, labels, tvec, matched = collect_epochs(records, subject, run_cond, spot)
    except FileNotFoundError:
        return None

    detail_df, summary_df, teps_by_condition, points_by_condition = analyze_selection(
        subject=subject,
        run_cond=run_cond,
        spot=spot,
        epochs=epochs,
        labels=labels,
        tvec=tvec,
        channel_names=channel_names,
        n_mean=n_mean,
    )

    if detail_df.empty:
        print(f"  no analyzable epochs for {subject} {run_cond} {spot}")
        return None

    figure_name = (
        f"pavlov2026_{sanitize_filename_part(subject)}_"
        f"session{SESSION_BY_SUBJECT[subject]}_"
        f"{sanitize_filename_part(run_cond)}_{sanitize_filename_part(spot)}.png"
    )
    save_selection_figure(
        output_path=output_dir / "figures" / figure_name,
        subject=subject,
        run_cond=run_cond,
        spot=spot,
        tvec=tvec,
        teps_by_condition=teps_by_condition,
        points_by_condition=points_by_condition,
        channel_colors=channel_colors,
        y_limit=y_limit,
    )

    source_files = "; ".join(record.path.name for record in matched)
    source_originals = "; ".join(record.original_file for record in matched)
    for df in (detail_df, summary_df):
        df["source_files"] = source_files
        df["source_originals"] = source_originals

    return detail_df, summary_df


def save_tables(
    output_dir: Path,
    details: pd.DataFrame,
    summaries: pd.DataFrame,
) -> None:
    tables_dir = output_dir / "tables"
    details.to_csv(tables_dir / "pavlov2026_amplitudes_detail_all.csv", index=False)
    summaries.to_csv(tables_dir / "pavlov2026_amplitudes_summary_all.csv", index=False)

    for subject, df_subject in details.groupby("subject", sort=True):
        df_subject.to_csv(tables_dir / f"pavlov2026_{subject}_amplitudes_detail.csv", index=False)

    for subject, df_subject in summaries.groupby("subject", sort=True):
        df_subject.to_csv(tables_dir / f"pavlov2026_{subject}_amplitudes_summary.csv", index=False)


def run(
    data_dir: Path | None = None,
    output_dir: Path = DEFAULT_RESULTS_DIR,
    subjects: list[str] | None = None,
    run_conds: list[str] | None = None,
    spots: list[str] | None = None,
    n_mean: int = 3,
    y_limit: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(data_dir) if data_dir is not None else resolve_default_data_dir()
    output_dir = Path(output_dir)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)

    subjects = subjects or list(SESSION_BY_SUBJECT)
    run_conds = run_conds or list(RUN_CONDS)
    spots = spots or list(SPOTS)

    unknown_subjects = sorted(set(subjects) - set(SESSION_BY_SUBJECT))
    if unknown_subjects:
        raise ValueError(f"Unknown subjects: {', '.join(unknown_subjects)}")

    records = discover_records(data_dir)
    if not records:
        raise FileNotFoundError(f"No *SOUND.mat files found in {data_dir}")

    print(f"Found {len(records)} SOUND files in {data_dir}")
    channel_colors = load_channel_colors()
    channel_names = load_channel_names()

    all_details = []
    all_summaries = []
    for subject in subjects:
        print(f"\nSubject {subject} / session {SESSION_BY_SUBJECT[subject]}")
        for run_cond in run_conds:
            for spot in spots:
                result = process_selection(
                    records=records,
                    output_dir=output_dir,
                    subject=subject,
                    run_cond=run_cond,
                    spot=spot,
                    channel_names=channel_names,
                    channel_colors=channel_colors,
                    n_mean=n_mean,
                    y_limit=y_limit,
                )
                if result is None:
                    print(f"  skip: no files for {subject} {run_cond} {spot}")
                    continue
                detail_df, summary_df = result
                all_details.append(detail_df)
                all_summaries.append(summary_df)
                print(f"  saved {len(summary_df)} component summaries")

    if not all_details:
        raise RuntimeError("No matching records were processed")

    details = pd.concat(all_details, ignore_index=True)
    summaries = pd.concat(all_summaries, ignore_index=True)
    save_tables(output_dir, details, summaries)

    print(f"\nSaved figures to {output_dir / 'figures'}")
    print(f"Saved tables to {output_dir / 'tables'}")
    print(f"Detail rows: {len(details)}; summary rows: {len(summaries)}")

    return details, summaries


def parse_list_arg(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Pavlov 2026 component-amplitude analysis for cleaned SOUND MAT files."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--subjects", nargs="*", default=None, help="Subjects, e.g. 01AV 10ES")
    parser.add_argument("--conds", nargs="*", default=None, help="Run conditions: real, MI")
    parser.add_argument("--spots", nargs="*", default=None, help="Spots: M1_PA, M1_AP, PPC_L")
    parser.add_argument("--n-mean", type=int, default=3)
    parser.add_argument("--y-limit", type=float, default=30.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        subjects=parse_list_arg(args.subjects),
        run_conds=parse_list_arg(args.conds),
        spots=parse_list_arg(args.spots),
        n_mean=args.n_mean,
        y_limit=args.y_limit,
    )


if __name__ == "__main__":
    main()
