from __future__ import annotations

import argparse
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.pavlov2026_comp_analysis import (
    COMP_COLORS,
    COMP_WINDOWS,
    DEFAULT_RESULTS_DIR,
    INNER_CONDITIONS,
    RUN_CONDS,
    SESSION_BY_SUBJECT,
    SoundRecord,
    collect_epochs,
    load_channel_colors,
    parse_session_id,
    read_matlab_string,
    resolve_default_data_dir,
)


DEFAULT_TABLE_PATH = DEFAULT_RESULTS_DIR / "short_table.xlsx"
DEFAULT_FIGURES_DIR = DEFAULT_RESULTS_DIR / "figures_short_table"
TABLE_CONDITION_BY_PANEL = {
    "rest": "rest",
    "onset": None,
}


def sanitize_filename_part(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def parse_list_arg(values: list[str] | None) -> list[str] | None:
    if not values:
        return None
    parsed = []
    for value in values:
        parsed.extend(part.strip() for part in value.split(",") if part.strip())
    return parsed or None


def normalize_short_table(table_path: Path) -> pd.DataFrame:
    df = pd.read_excel(table_path)
    rename_map = {
        "comp": "component",
        "amp": "amplitude_uv",
        "lat": "latency_ms",
    }
    df = df.rename(columns=rename_map)

    required_cols = {"subject", "component", "spot", "amplitude_uv", "latency_ms", "condition"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"{table_path} is missing required columns: {', '.join(missing_cols)}")

    df = df.loc[:, ["subject", "component", "spot", "amplitude_uv", "latency_ms", "condition"]].copy()
    df["subject"] = df["subject"].astype(str).str.strip()
    df["component"] = df["component"].astype(str).str.strip().str.lower()
    df["spot"] = df["spot"].astype(str).str.strip()
    df["condition"] = df["condition"].astype(str).str.strip()
    df["amplitude_uv"] = pd.to_numeric(df["amplitude_uv"], errors="coerce")
    df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")

    unknown_components = sorted(set(df["component"].dropna()) - set(COMP_WINDOWS))
    if unknown_components:
        raise ValueError(f"Unknown components in {table_path}: {', '.join(unknown_components)}")

    key_cols = ["subject", "component", "spot", "condition"]
    duplicated = df[df.duplicated(key_cols, keep=False)]
    if not duplicated.empty:
        duplicated_keys = duplicated[key_cols].drop_duplicates().head(10).to_dict("records")
        raise ValueError(f"Duplicate short-table keys found, examples: {duplicated_keys}")

    return df


def discover_records_recursive(data_dir: Path) -> list[SoundRecord]:
    records: list[SoundRecord] = []
    for path in sorted(data_dir.rglob("*SOUND.mat")):
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


def collect_teps_by_condition(
    records,
    subject: str,
    run_cond: str,
    spot: str,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    epochs, labels, tvec, _ = collect_epochs(records, subject, run_cond, spot)

    teps_by_condition = {}
    for inner_condition in INNER_CONDITIONS:
        epoch_mask = labels == inner_condition
        if np.any(epoch_mask):
            teps_by_condition[inner_condition] = np.mean(epochs[:, epoch_mask, :], axis=1)

    return tvec, teps_by_condition


def make_points_by_panel(
    table_df: pd.DataFrame,
    subject: str,
    run_cond: str,
    spot: str,
) -> dict[str, dict[str, list[dict]]]:
    points_by_panel: dict[str, dict[str, list[dict]]] = {
        panel: {component: [] for component in COMP_WINDOWS}
        for panel in INNER_CONDITIONS
    }

    for panel in INNER_CONDITIONS:
        table_condition = TABLE_CONDITION_BY_PANEL.get(panel)
        if table_condition is None and panel == "onset":
            table_condition = run_cond
        if table_condition is None:
            continue

        panel_df = table_df.loc[
            table_df["subject"].eq(subject)
            & table_df["spot"].eq(spot)
            & table_df["condition"].eq(table_condition)
        ]

        for _, row in panel_df.iterrows():
            if not np.isfinite(row["latency_ms"]) or not np.isfinite(row["amplitude_uv"]):
                continue

            component = row["component"]
            points_by_panel[panel][component].append(
                {
                    "latency_ms": float(row["latency_ms"]),
                    "amplitude_uv": float(row["amplitude_uv"]),
                    "table_condition": table_condition,
                }
            )

    return points_by_panel


def save_short_table_figure(
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
        f"{subject} | session {SESSION_BY_SUBJECT[subject]} | {run_cond} | {spot} | short_table",
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
            if not points:
                continue

            ax.scatter(
                xs,
                ys,
                s=96,
                color=COMP_COLORS[component],
                edgecolor="black",
                linewidth=1.0,
                zorder=10,
                label=component,
            )
            ax.text(
                xs[0],
                ys[0],
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
        ax.set_xlabel("Time [ms]", fontsize=12)

    axes[0].set_ylabel("EEG signal [uV]", fontsize=12)
    handles = []
    labels = []
    for ax in axes:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        handles.extend(ax_handles)
        labels.extend(ax_labels)
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


def run(
    data_dir: Path | None = None,
    table_path: Path = DEFAULT_TABLE_PATH,
    output_dir: Path = DEFAULT_FIGURES_DIR,
    subjects: list[str] | None = None,
    run_conds: list[str] | None = None,
    spots: list[str] | None = None,
    y_limit: float = 18.0,
) -> list[Path]:
    data_dir = Path(data_dir) if data_dir is not None else resolve_default_data_dir()
    table_path = Path(table_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table_df = normalize_short_table(table_path)

    subjects = subjects or sorted(table_df["subject"].dropna().unique().tolist())
    run_conds = run_conds or list(RUN_CONDS)
    spots = spots or sorted(table_df["spot"].dropna().unique().tolist())

    unknown_subjects = sorted(set(subjects) - set(SESSION_BY_SUBJECT))
    if unknown_subjects:
        raise ValueError(f"Unknown subjects: {', '.join(unknown_subjects)}")

    records = discover_records_recursive(data_dir)
    if not records:
        raise FileNotFoundError(f"No *SOUND.mat files found in {data_dir}")

    print(f"Found {len(records)} SOUND files in {data_dir}")
    print(f"Loaded {len(table_df)} short-table rows from {table_path}")
    channel_colors = load_channel_colors()

    saved_paths = []
    for subject in subjects:
        print(f"\nSubject {subject} / session {SESSION_BY_SUBJECT[subject]}")
        for run_cond in run_conds:
            for spot in spots:
                if table_df.loc[table_df["subject"].eq(subject) & table_df["spot"].eq(spot)].empty:
                    print(f"  skip: no table rows for {subject} {spot}")
                    continue

                try:
                    tvec, teps_by_condition = collect_teps_by_condition(records, subject, run_cond, spot)
                except FileNotFoundError:
                    print(f"  skip: no files for {subject} {run_cond} {spot}")
                    continue

                points_by_condition = make_points_by_panel(table_df, subject, run_cond, spot)
                plotted_points = sum(
                    len(points)
                    for by_component in points_by_condition.values()
                    for points in by_component.values()
                )
                if plotted_points == 0:
                    print(f"  skip: no finite table points for {subject} {run_cond} {spot}")
                    continue

                figure_name = (
                    f"pavlov2026_{sanitize_filename_part(subject)}_"
                    f"session{SESSION_BY_SUBJECT[subject]}_"
                    f"{sanitize_filename_part(run_cond)}_"
                    f"{sanitize_filename_part(spot)}_short_table.png"
                )
                output_path = output_dir / figure_name
                save_short_table_figure(
                    output_path=output_path,
                    subject=subject,
                    run_cond=run_cond,
                    spot=spot,
                    tvec=tvec,
                    teps_by_condition=teps_by_condition,
                    points_by_condition=points_by_condition,
                    channel_colors=channel_colors,
                    y_limit=y_limit,
                )
                saved_paths.append(output_path)
                print(f"  saved {output_path.name} ({plotted_points} points)")

    if not saved_paths:
        raise RuntimeError("No short-table figures were created")

    print(f"\nSaved {len(saved_paths)} figures to {output_dir}")
    return saved_paths


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Draw Pavlov 2026 TEP figures with component points from short_table.xlsx."
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--subjects", nargs="*", default=None, help="Subjects, e.g. 01AV 10ES")
    parser.add_argument("--conds", nargs="*", default=None, help="Run conditions: real, MI")
    parser.add_argument("--spots", nargs="*", default=None, help="Spots: M1_PA, M1_AP")
    parser.add_argument("--y-limit", type=float, default=18.0)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    run(
        data_dir=args.data_dir,
        table_path=args.table,
        output_dir=args.output_dir,
        subjects=parse_list_arg(args.subjects),
        run_conds=parse_list_arg(args.conds),
        spots=parse_list_arg(args.spots),
        y_limit=args.y_limit,
    )


if __name__ == "__main__":
    main()
