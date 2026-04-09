import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "comparison" / "profile_fit_examples_vis_vs_gad.json"

TIME_UNIT_LABEL_DEFAULT = "min"
CONCENTRATION_LABEL_DEFAULT = "Concentration (mg/mL)"
FIGURE_TITLE_DEFAULT = "Profile Fit Comparison"

FIT_ONLY_FILL_ALPHA = 0.10
ROI_FIXED_FILL_ALPHA = 0.08
ROI_FIXED_1SD_FILL_ALPHA = 0.10
COMBINED_FILL_ALPHA = 0.09
COMBINED_1SD_FILL_ALPHA = 0.10
CENTER_LINEWIDTH = 2.3
SAMPLE_LINEWIDTH = 1.1
SAMPLE_LINE_ALPHA = 0.20
MEASURED_MARKER_SIZE = 18
MEASURED_SAMPLE_ALPHA = 0.65
Z95 = 1.96

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
DEFAULT_COLORS = [
    "#1f77b4",
    "#d95f02",
    "#1b9e77",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
]

PLOT_MODE_SPECS = [
    ("none", "no_uncertainty", "profile_fit_examples_no_uncertainty.png"),
    ("fit_only", "model_fit_only", "profile_fit_examples_model_fit_only_uncertainty.png"),
    ("combined_95ci", "combined_95CI", "profile_fit_examples_combined_95CI.png"),
    ("combined_1sd", "combined_1SD", "profile_fit_examples_combined_1SD.png"),
    ("roi_fixed", "fixedROI_95CI", "profile_fit_examples_fixedROI_95CI.png"),
    ("roi_fixed_1sd", "fixedROI_1SD", "profile_fit_examples_fixedROI_1SD.png"),
]

METRIC_COLUMN_CANDIDATES = {
    "effective_diffusivity_mm2_s": [
        "effective_diffusivity_mm2_s",
        "regularized_effective_diffusivity_mm2_s",
        "global_effective_diffusivity_mm2_s",
    ],
    "fitted_Cs": ["fitted_Cs", "regularized_fitted_Cs", "global_smooth_Cs"],
    "fitted_velocity_mm_s": [
        "fitted_velocity_mm_s",
        "regularized_fitted_velocity_mm_s",
        "global_fitted_velocity_mm_s",
    ],
    "profile_fit_rmse": ["profile_fit_rmse", "regularized_profile_fit_rmse", "global_profile_fit_rmse"],
    "profile_fit_r2": ["profile_fit_r2", "regularized_profile_fit_r2", "global_profile_fit_r2"],
    "peclet_number": ["peclet_number"],
}

METRIC_LABELS = {
    "effective_diffusivity_mm2_s": "Effective Diffusivity (mm^2/s)",
    "fitted_Cs": "Fitted Surface Concentration (mg/mL)",
    "fitted_velocity_mm_s": "Fitted Velocity (mm/s)",
    "profile_fit_rmse": "Profile Fit RMSE",
    "profile_fit_r2": "Profile Fit R^2",
    "peclet_number": "Peclet Number",
    "measured_profile_auc_concentration_x_mm": "Measured Profile AUC (conc x mm)",
    "fitted_profile_auc_concentration_x_mm": "Fitted Profile AUC (conc x mm)",
    "peak_measured_concentration": "Measured Peak Concentration (mg/mL)",
    "peak_fitted_concentration": "Fitted Peak Concentration (mg/mL)",
    "surface_measured_concentration": "Measured Surface Concentration (mg/mL)",
    "surface_fitted_concentration": "Fitted Surface Concentration (mg/mL)",
    "penetration_depth_fit_10pct_peak_mm": "Penetration Depth at 10% Fit Peak (mm)",
}

DEFAULT_STATS_METRICS = [
    "effective_diffusivity_mm2_s",
    "fitted_Cs",
    "profile_fit_r2",
    "fitted_profile_auc_concentration_x_mm",
    "peak_fitted_concentration",
]


@dataclass
class SampleData:
    tracer_name: str
    tracer_label: str
    tracer_color: str
    tracer_marker: str
    sample_id: str
    roi_folder: str
    dx_mm: float
    run_display: str
    selection_mode: str
    paths_used: Dict[str, str]
    measured: np.ndarray
    fitted: np.ndarray
    fit_params_df: pd.DataFrame
    times: np.ndarray
    r2: Optional[np.ndarray]
    fit_std: Optional[np.ndarray]
    hu_std: Optional[np.ndarray]
    calib_std: Optional[np.ndarray]
    combined_std: Optional[np.ndarray]
    combined_ci_low: Optional[np.ndarray]
    combined_ci_high: Optional[np.ndarray]
    roi_fixed_std: Optional[np.ndarray]
    depth: np.ndarray
    valid_rows: np.ndarray
    metric_series: Dict[str, Optional[np.ndarray]]


def _is_zip_path(path: str) -> bool:
    return str(path).lower().endswith(".zip")


def _normalize_rel_path(rel_path: str) -> str:
    return str(rel_path).replace("\\", "/").strip("/")


def _resolve_path_from_config(path_value: Optional[str], config_path: Path) -> Optional[str]:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if path.is_absolute():
        return str(path)
    return str((config_path.parent / path).resolve())


def _read_csv_any(base: str, relative_path: str) -> pd.DataFrame:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return pd.read_csv(io.BytesIO(zf.read(rel)))
    return pd.read_csv(Path(base) / rel)


def _path_exists_any(base: str, relative_path: str) -> bool:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return rel in zf.namelist()
    return (Path(base) / rel).exists()


def load_csv_required_any(base: str, relative_path: str) -> pd.DataFrame:
    if not _path_exists_any(base, relative_path):
        raise FileNotFoundError(f"Required file not found: {relative_path}")
    return _read_csv_any(base, relative_path)


def load_csv_optional_any(base: str, relative_path: Optional[str]) -> Optional[pd.DataFrame]:
    if relative_path in (None, ""):
        return None
    if not _path_exists_any(base, relative_path):
        print(f"Optional file not found, skipping: {relative_path}")
        return None
    return _read_csv_any(base, relative_path)


def load_csv_required_path(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_csv_optional_path(path_value: str) -> Optional[pd.DataFrame]:
    path = Path(path_value)
    if not path.exists():
        print(f"Optional file not found, skipping: {path}")
        return None
    return pd.read_csv(path)


def find_time_column(df: pd.DataFrame) -> str:
    for col in ["time", "time_plot", "time_min", "time_minutes"]:
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find a time column. Checked: time, time_plot, time_min, time_minutes."
        "\nAvailable columns:\n" + "\n".join(df.columns.tolist())
    )


def to_numeric_2d(df: pd.DataFrame) -> np.ndarray:
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D numeric array.")
    return arr


def nearest_index(values: np.ndarray, target: float, valid_mask: Optional[np.ndarray] = None) -> int:
    vals = np.asarray(values, dtype=float)
    if valid_mask is None:
        valid_mask = np.isfinite(vals)
    if not np.any(valid_mask):
        raise ValueError("No valid values available for nearest-index selection.")
    idx_valid = np.where(valid_mask)[0]
    nearest_pos = np.argmin(np.abs(vals[idx_valid] - float(target)))
    return int(idx_valid[nearest_pos])


def build_depth_mm(n_depth: int, dx_mm: float) -> np.ndarray:
    return np.arange(n_depth, dtype=float) * float(dx_mm)


def build_roi_fixed_std(
    fit_std: Optional[np.ndarray],
    hu_std: Optional[np.ndarray],
    calib_std: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    parts = []
    for arr in [fit_std, hu_std, calib_std]:
        if arr is not None:
            parts.append(np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    if not parts:
        return None
    return np.sqrt(np.sum([part ** 2 for part in parts], axis=0))


def compute_axis_limits(*arrays: np.ndarray) -> tuple[tuple[float, float], float]:
    valid_parts = []
    for arr in arrays:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size:
            valid_parts.append(finite)
    if not valid_parts:
        return (0.0, 1.0), 1.0
    all_y = np.concatenate(valid_parts)
    ymin = float(np.min(all_y))
    ymax = float(np.max(all_y))
    pad = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
    return (max(0.0, ymin - pad), ymax + pad), pad


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def extract_optional_numeric_series(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[np.ndarray]:
    column = first_existing_column(df, candidates)
    if column is None:
        return None
    return pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)


def build_profile_rel_paths(run_rel: str, roi_folder: str) -> Dict[str, str]:
    run_rel = _normalize_rel_path(run_rel)
    roi_folder = str(roi_folder).strip("/").strip("\\")
    base = f"{run_rel}/{roi_folder}"
    return {
        "measured_profiles_csv": f"{base}/CSVs_Profiles/measured_profiles_depth_vs_time.csv",
        "fitted_profiles_csv": f"{base}/CSVs_Profiles/fitted_profiles_depth_vs_time.csv",
        "fit_parameters_csv": f"{base}/CSVs_Summaries/fit_parameters_vs_time.csv",
        "fit_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_std_depth_vs_time.csv",
        "fit_ci_low_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_ci_low_depth_vs_time.csv",
        "fit_ci_high_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_ci_high_depth_vs_time.csv",
        "hu_noise_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_hu_noise_std_depth_vs_time.csv",
        "roi_sensitivity_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_roi_sensitivity_std_depth_vs_time.csv",
        "calibration_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_calibration_std_depth_vs_time.csv",
        "combined_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_combined_std_depth_vs_time.csv",
        "combined_ci_low_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_combined_ci_low_depth_vs_time.csv",
        "combined_ci_high_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_combined_ci_high_depth_vs_time.csv",
    }


def build_profile_abs_paths(run_path: str, roi_folder: str) -> Dict[str, str]:
    run_dir = Path(run_path)
    roi_dir = run_dir / str(roi_folder)
    return {
        "measured_profiles_csv": str(roi_dir / "CSVs_Profiles" / "measured_profiles_depth_vs_time.csv"),
        "fitted_profiles_csv": str(roi_dir / "CSVs_Profiles" / "fitted_profiles_depth_vs_time.csv"),
        "fit_parameters_csv": str(roi_dir / "CSVs_Summaries" / "fit_parameters_vs_time.csv"),
        "fit_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_std_depth_vs_time.csv"),
        "fit_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_ci_low_depth_vs_time.csv"),
        "fit_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_ci_high_depth_vs_time.csv"),
        "hu_noise_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_hu_noise_std_depth_vs_time.csv"),
        "roi_sensitivity_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_roi_sensitivity_std_depth_vs_time.csv"),
        "calibration_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_calibration_std_depth_vs_time.csv"),
        "combined_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_combined_std_depth_vs_time.csv"),
        "combined_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_combined_ci_low_depth_vs_time.csv"),
        "combined_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_combined_ci_high_depth_vs_time.csv"),
    }


def choose_target_times(samples: Sequence[SampleData], manual_target_times: Optional[Sequence[float]]) -> List[float]:
    if manual_target_times is not None:
        if len(manual_target_times) == 0:
            raise ValueError("target_times_min cannot be empty.")
        return [float(x) for x in manual_target_times]

    min_times = []
    max_times = []
    for sample in samples:
        valid_times = sample.times[sample.valid_rows & np.isfinite(sample.times)]
        if valid_times.size == 0:
            raise ValueError(f"Sample {sample.sample_id} has no valid fitted profile times.")
        min_times.append(float(np.min(valid_times)))
        max_times.append(float(np.max(valid_times)))

    overlap_start = max(min_times)
    overlap_end = min(max_times)
    if overlap_end < overlap_start:
        raise ValueError("Samples do not share an overlapping valid time window.")
    return [overlap_start, 0.5 * (overlap_start + overlap_end), overlap_end]


def build_panel_names(n_panels: int) -> List[str]:
    if n_panels == 3:
        return ["Beginning", "Middle", "End"]
    return [f"Time {idx + 1}" for idx in range(n_panels)]


def interp_profile(depth: np.ndarray, values: np.ndarray, target_depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)
    target_depth = np.asarray(target_depth, dtype=float)
    valid = np.isfinite(depth) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return np.full(target_depth.shape, np.nan, dtype=float)
    return np.interp(target_depth, depth[valid], values[valid], left=np.nan, right=np.nan)


def build_group_grid(samples: Sequence[SampleData]) -> np.ndarray:
    max_depth = max(float(np.nanmax(sample.depth)) for sample in samples)
    dx_candidates = []
    for sample in samples:
        if sample.depth.size > 1:
            diffs = np.diff(sample.depth)
            finite = diffs[np.isfinite(diffs) & (diffs > 0)]
            if finite.size:
                dx_candidates.append(float(np.min(finite)))
    dx = min(dx_candidates) if dx_candidates else 1.0
    n_points = int(np.floor(max_depth / dx)) + 1
    return np.arange(n_points, dtype=float) * dx


def compute_band_from_sample(sample: SampleData, idx: int, mode: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    fit_row = sample.fitted[idx]
    if mode == "fit_only" and sample.fit_std is not None:
        band = sample.fit_std[idx]
        return fit_row - band, fit_row + band
    if mode == "roi_fixed" and sample.roi_fixed_std is not None:
        band = Z95 * sample.roi_fixed_std[idx]
        return fit_row - band, fit_row + band
    if mode == "roi_fixed_1sd" and sample.roi_fixed_std is not None:
        band = sample.roi_fixed_std[idx]
        return fit_row - band, fit_row + band
    if mode == "combined_95ci":
        if sample.combined_ci_low is not None and sample.combined_ci_high is not None:
            return sample.combined_ci_low[idx], sample.combined_ci_high[idx]
        if sample.combined_std is not None:
            band = Z95 * sample.combined_std[idx]
            return fit_row - band, fit_row + band
    if mode == "combined_1sd" and sample.combined_std is not None:
        band = sample.combined_std[idx]
        return fit_row - band, fit_row + band
    return None, None


def aggregate_tracer_panel(
    samples: Sequence[SampleData],
    matched_indices: Dict[str, Dict[float, int]],
    target_time: float,
    mode: str,
) -> Dict[str, Any]:
    grid = build_group_grid(samples)
    measured_rows = []
    fitted_rows = []
    band_low_rows = []
    band_high_rows = []
    actual_times = []
    fit_r2_values = []
    sample_curves = []

    for sample in samples:
        idx = matched_indices[sample.sample_id][target_time]
        measured_row = interp_profile(sample.depth, sample.measured[idx], grid)
        fitted_row = interp_profile(sample.depth, sample.fitted[idx], grid)
        measured_rows.append(measured_row)
        fitted_rows.append(fitted_row)

        low_row, high_row = compute_band_from_sample(sample, idx, mode)
        if low_row is not None and high_row is not None:
            band_low_rows.append(interp_profile(sample.depth, low_row, grid))
            band_high_rows.append(interp_profile(sample.depth, high_row, grid))

        actual_times.append(float(sample.times[idx]))
        if sample.r2 is not None:
            fit_r2_values.append(float(sample.r2[idx]))

        sample_curves.append(
            {
                "sample_id": sample.sample_id,
                "depth": grid,
                "measured": measured_row,
                "fitted": fitted_row,
            }
        )

    measured_stack = np.vstack(measured_rows)
    fitted_stack = np.vstack(fitted_rows)
    aggregate = {
        "depth": grid,
        "measured_mean": np.nanmean(measured_stack, axis=0),
        "fitted_mean": np.nanmean(fitted_stack, axis=0),
        "actual_times": actual_times,
        "sample_curves": sample_curves,
        "sample_count": len(samples),
        "r2_values": fit_r2_values,
        "band_low": None,
        "band_high": None,
    }
    if band_low_rows and band_high_rows:
        aggregate["band_low"] = np.nanmean(np.vstack(band_low_rows), axis=0)
        aggregate["band_high"] = np.nanmean(np.vstack(band_high_rows), axis=0)
    return aggregate


def build_panel_subtitle(tracer_panel_data: Dict[str, Dict[str, Any]], time_unit_label: str) -> str:
    parts = []
    for tracer_label, panel_data in tracer_panel_data.items():
        times = np.asarray(panel_data["actual_times"], dtype=float)
        finite = times[np.isfinite(times)]
        if finite.size == 0:
            continue
        time_text = f"{finite[0]:.2f}" if finite.size == 1 else f"{np.min(finite):.2f}-{np.max(finite):.2f}"
        parts.append(f"{tracer_label}: {time_text} {time_unit_label} (n={panel_data['sample_count']})")
    return "; ".join(parts)


def add_panel_title(
    ax: plt.Axes,
    panel_name: str,
    target_time: float,
    target_subtitle: str,
    time_unit_label: str,
) -> None:
    title = f"{panel_name} (~{target_time:.2f} {time_unit_label})"
    if target_subtitle:
        title += "\nActual matched frames: " + target_subtitle
    ax.set_title(title)


def save_plot(fig: plt.Figure, out_folder: str, subfolder: str, out_name: str) -> str:
    folder = Path(out_folder) / "profile_fit_examples" / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def build_uncertainty_note(mode: str) -> str:
    notes = {
        "none": "Mean tracer profiles with no band.",
        "fit_only": "Band shows the average within-sample fit-only uncertainty.",
        "combined_95ci": "Band shows the average within-sample combined 95% interval.",
        "combined_1sd": "Band shows the average within-sample combined 1 SD envelope.",
        "roi_fixed": "Band shows the average within-sample fixed-ROI 95% interval.",
        "roi_fixed_1sd": "Band shows the average within-sample fixed-ROI 1 SD envelope.",
    }
    return notes[mode]


def save_profile_figure(
    out_folder: str,
    out_name: str,
    subfolder: str,
    mode: str,
    samples_by_tracer: Dict[str, List[SampleData]],
    tracer_order: Sequence[str],
    target_times: Sequence[float],
    matched_indices: Dict[str, Dict[float, int]],
    tracer_styles: Dict[str, Dict[str, str]],
    figure_title: str,
    ylims: tuple[float, float],
    time_unit_label: str,
    concentration_label: str,
) -> str:
    n_panels = len(target_times)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, max(4.0 * n_panels, 7.0)), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    panel_names = build_panel_names(n_panels)

    for ax, panel_name, target_time in zip(axes, panel_names, target_times):
        tracer_panel_data = {}
        for tracer_name in tracer_order:
            tracer_samples = samples_by_tracer[tracer_name]
            tracer_label = tracer_samples[0].tracer_label
            style = tracer_styles[tracer_name]
            panel_data = aggregate_tracer_panel(tracer_samples, matched_indices, target_time, mode)
            tracer_panel_data[tracer_label] = panel_data

            for sample_curve in panel_data["sample_curves"]:
                ax.plot(
                    sample_curve["depth"],
                    sample_curve["fitted"],
                    color=style["color"],
                    linewidth=SAMPLE_LINEWIDTH,
                    alpha=SAMPLE_LINE_ALPHA,
                )

            if panel_data["band_low"] is not None and panel_data["band_high"] is not None:
                alpha = {
                    "fit_only": FIT_ONLY_FILL_ALPHA,
                    "roi_fixed": ROI_FIXED_FILL_ALPHA,
                    "roi_fixed_1sd": ROI_FIXED_1SD_FILL_ALPHA,
                    "combined_95ci": COMBINED_FILL_ALPHA,
                    "combined_1sd": COMBINED_1SD_FILL_ALPHA,
                }.get(mode, 0.0)
                ax.fill_between(panel_data["depth"], panel_data["band_low"], panel_data["band_high"], color=style["color"], alpha=alpha)

            ax.plot(
                panel_data["depth"],
                panel_data["fitted_mean"],
                color=style["color"],
                linewidth=CENTER_LINEWIDTH,
                label=f"{tracer_label} mean fit (n={panel_data['sample_count']})",
            )
            ax.scatter(
                panel_data["depth"],
                panel_data["measured_mean"],
                s=MEASURED_MARKER_SIZE,
                marker=style["marker"],
                color=style["color"],
                alpha=MEASURED_SAMPLE_ALPHA,
                label=f"{tracer_label} mean measured",
            )

        add_panel_title(ax, panel_name, target_time, build_panel_subtitle(tracer_panel_data, time_unit_label), time_unit_label)
        ax.set_ylabel(concentration_label)
        ax.grid(True)
        ax.set_ylim(*ylims)

    axes[-1].set_xlabel("Depth (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(4, max(2, len(labels))), bbox_to_anchor=(0.5, 0.995))
    fig.suptitle(f"{figure_title}\n{build_uncertainty_note(mode)}", y=1.02, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return save_plot(fig, out_folder, subfolder, out_name)


def save_audit_report(out_folder: str, payload: dict) -> None:
    out_dir = Path(out_folder) / "profile_fit_examples" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "comparison_audit.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for child_key, child_value in value.items():
                lines.append(f"  {child_key}: {child_value}")
        else:
            lines.append(f"{key}: {value}")
    with open(out_dir / "comparison_audit.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def value_at_index(series: Optional[np.ndarray], idx: int) -> float:
    if series is None or idx >= len(series):
        return float("nan")
    value = series[idx]
    return float(value) if np.isfinite(value) else float("nan")


def safe_profile_auc(depth: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(depth) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values[valid], depth[valid]))
    return float(np.trapz(values[valid], depth[valid]))


def safe_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def safe_first(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    if finite.size == 0 or not np.isfinite(finite[0]):
        return float("nan")
    return float(finite[0])


def penetration_depth(depth: np.ndarray, values: np.ndarray, threshold_fraction: float = 0.10) -> float:
    depth = np.asarray(depth, dtype=float)
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(depth) & np.isfinite(values)
    if np.count_nonzero(finite) == 0:
        return float("nan")
    values_valid = values[finite]
    peak = np.max(values_valid)
    if peak <= 0:
        return float("nan")
    threshold = threshold_fraction * peak
    passing = depth[finite][values_valid >= threshold]
    if passing.size == 0:
        return float("nan")
    return float(np.max(passing))


def compute_sample_metric_row(sample: SampleData, target_time: float, matched_index: int) -> Dict[str, Any]:
    measured_row = sample.measured[matched_index]
    fitted_row = sample.fitted[matched_index]
    row = {
        "tracer_name": sample.tracer_name,
        "tracer_label": sample.tracer_label,
        "sample_id": sample.sample_id,
        "target_time_min": float(target_time),
        "actual_time_min": float(sample.times[matched_index]),
        "time_offset_min": float(sample.times[matched_index] - target_time),
        "run_display": sample.run_display,
        "roi_folder": sample.roi_folder,
        "dx_mm": float(sample.dx_mm),
        "depth_points": int(sample.measured.shape[1]),
        "measured_profile_auc_concentration_x_mm": safe_profile_auc(sample.depth, measured_row),
        "fitted_profile_auc_concentration_x_mm": safe_profile_auc(sample.depth, fitted_row),
        "peak_measured_concentration": safe_max(measured_row),
        "peak_fitted_concentration": safe_max(fitted_row),
        "surface_measured_concentration": safe_first(measured_row),
        "surface_fitted_concentration": safe_first(fitted_row),
        "penetration_depth_fit_10pct_peak_mm": penetration_depth(sample.depth, fitted_row, threshold_fraction=0.10),
    }
    for metric_name in METRIC_COLUMN_CANDIDATES:
        row[metric_name] = value_at_index(sample.metric_series.get(metric_name), matched_index)
    return row


def holm_adjust(p_values: Sequence[float]) -> List[float]:
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = (n - rank) * p_values[idx]
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(1.0, running_max)
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted.tolist()


def build_group_summary(metrics_df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for target_time in sorted(metrics_df["target_time_min"].dropna().unique()):
        target_df = metrics_df[metrics_df["target_time_min"] == target_time]
        for metric in metrics:
            if metric not in target_df.columns:
                continue
            for tracer_name, tracer_df in target_df.groupby("tracer_name"):
                values = pd.to_numeric(tracer_df[metric], errors="coerce").dropna()
                if values.empty:
                    continue
                std_value = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
                sem_value = float(std_value / np.sqrt(len(values))) if len(values) > 1 else float("nan")
                rows.append(
                    {
                        "target_time_min": float(target_time),
                        "tracer_name": tracer_name,
                        "metric_name": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "n": int(len(values)),
                        "mean": float(values.mean()),
                        "std": std_value,
                        "sem": sem_value,
                        "min": float(values.min()),
                        "max": float(values.max()),
                    }
                )
    return pd.DataFrame(rows)


def build_statistics_tables(
    metrics_df: pd.DataFrame,
    metrics: Sequence[str],
    alpha: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_rows: List[Dict[str, Any]] = []
    omnibus_rows: List[Dict[str, Any]] = []

    for target_time in sorted(metrics_df["target_time_min"].dropna().unique()):
        target_df = metrics_df[metrics_df["target_time_min"] == target_time]
        for metric in metrics:
            if metric not in target_df.columns:
                continue

            tracer_groups: Dict[str, np.ndarray] = {}
            tracer_labels: Dict[str, str] = {}
            for tracer_name, tracer_df in target_df.groupby("tracer_name"):
                values = pd.to_numeric(tracer_df[metric], errors="coerce").dropna().to_numpy(dtype=float)
                if values.size:
                    tracer_groups[tracer_name] = values
                    tracer_labels[tracer_name] = tracer_df["tracer_label"].iloc[0]

            if len(tracer_groups) >= 3:
                eligible_groups = [values for values in tracer_groups.values() if values.size >= 2]
                if len(eligible_groups) >= 3:
                    statistic, p_value = stats.f_oneway(*eligible_groups)
                    omnibus_rows.append(
                        {
                            "target_time_min": float(target_time),
                            "metric_name": metric,
                            "metric_label": METRIC_LABELS.get(metric, metric),
                            "test_name": "one_way_anova",
                            "group_count": int(len(eligible_groups)),
                            "statistic": float(statistic),
                            "p_value": float(p_value),
                            "alpha": float(alpha),
                            "significant": bool(np.isfinite(p_value) and p_value < alpha),
                            "note": "",
                        }
                    )
                else:
                    omnibus_rows.append(
                        {
                            "target_time_min": float(target_time),
                            "metric_name": metric,
                            "metric_label": METRIC_LABELS.get(metric, metric),
                            "test_name": "one_way_anova",
                            "group_count": int(len(tracer_groups)),
                            "statistic": float("nan"),
                            "p_value": float("nan"),
                            "alpha": float(alpha),
                            "significant": False,
                            "note": "Need at least 3 tracer groups with n>=2 for ANOVA.",
                        }
                    )

            raw_p_values = []
            raw_row_indices = []
            for tracer_a, tracer_b in combinations(sorted(tracer_groups.keys()), 2):
                values_a = tracer_groups[tracer_a]
                values_b = tracer_groups[tracer_b]
                note = ""
                statistic = float("nan")
                p_value = float("nan")
                if values_a.size >= 2 and values_b.size >= 2:
                    statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
                    raw_p_values.append(float(p_value))
                    raw_row_indices.append(len(pairwise_rows))
                else:
                    note = "Need n>=2 in each tracer group for Welch t-test."
                pairwise_rows.append(
                    {
                        "target_time_min": float(target_time),
                        "metric_name": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "test_name": "welch_ttest",
                        "tracer_a": tracer_a,
                        "tracer_a_label": tracer_labels[tracer_a],
                        "tracer_b": tracer_b,
                        "tracer_b_label": tracer_labels[tracer_b],
                        "n_a": int(values_a.size),
                        "n_b": int(values_b.size),
                        "mean_a": float(np.mean(values_a)) if values_a.size else float("nan"),
                        "mean_b": float(np.mean(values_b)) if values_b.size else float("nan"),
                        "mean_diff_a_minus_b": float(np.mean(values_a) - np.mean(values_b)) if values_a.size and values_b.size else float("nan"),
                        "statistic": float(statistic) if np.isfinite(statistic) else float("nan"),
                        "p_value_raw": float(p_value) if np.isfinite(p_value) else float("nan"),
                        "p_value_holm": float("nan"),
                        "alpha": float(alpha),
                        "significant": False,
                        "note": note,
                    }
                )

            if raw_p_values:
                adjusted = holm_adjust(raw_p_values)
                for row_idx, adjusted_p in zip(raw_row_indices, adjusted):
                    pairwise_rows[row_idx]["p_value_holm"] = float(adjusted_p)
                    pairwise_rows[row_idx]["significant"] = bool(np.isfinite(adjusted_p) and adjusted_p < alpha)

    return pd.DataFrame(pairwise_rows), pd.DataFrame(omnibus_rows)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        config = json.load(handle)
    if "tracers" not in config or not config["tracers"]:
        raise ValueError("Comparison config must define a non-empty 'tracers' list.")
    return config


def validate_and_resolve_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    resolved = dict(config)
    resolved["results_source"] = _resolve_path_from_config(config.get("results_source"), config_path)
    resolved["out_folder"] = _resolve_path_from_config(config.get("out_folder"), config_path)
    if resolved["out_folder"] in (None, ""):
        raise ValueError("Comparison config must define 'out_folder'.")

    resolved["figure_title"] = config.get("figure_title", FIGURE_TITLE_DEFAULT)
    resolved["time_unit_label"] = config.get("time_unit_label", TIME_UNIT_LABEL_DEFAULT)
    resolved["concentration_label"] = config.get("concentration_label", CONCENTRATION_LABEL_DEFAULT)
    resolved["target_times_min"] = config.get("target_times_min")
    resolved["stats_metrics"] = config.get("stats_metrics", DEFAULT_STATS_METRICS)
    resolved["stats_alpha"] = float(config.get("stats_alpha", 0.05))

    tracers_resolved = []
    for tracer_index, tracer_cfg in enumerate(config["tracers"]):
        tracer_name = tracer_cfg.get("name")
        if tracer_name in (None, ""):
            raise ValueError("Each tracer entry needs a non-empty 'name'.")
        samples_cfg = tracer_cfg.get("samples", [])
        if not samples_cfg:
            raise ValueError(f"Tracer '{tracer_name}' must define at least one sample.")
        tracers_resolved.append(
            {
                "name": tracer_name,
                "label": tracer_cfg.get("label", tracer_name),
                "color": tracer_cfg.get("color", DEFAULT_COLORS[tracer_index % len(DEFAULT_COLORS)]),
                "marker": tracer_cfg.get("marker", MARKERS[tracer_index % len(MARKERS)]),
                "dx_mm": tracer_cfg.get("dx_mm"),
                "roi_folder": tracer_cfg.get("roi_folder"),
                "samples": samples_cfg,
            }
        )
    resolved["tracers"] = tracers_resolved
    return resolved


def load_sample_data(tracer_cfg: Dict[str, Any], sample_cfg: Dict[str, Any], results_source: Optional[str]) -> SampleData:
    roi_folder = sample_cfg.get("roi_folder", tracer_cfg.get("roi_folder"))
    if roi_folder in (None, ""):
        raise ValueError(f"Sample in tracer '{tracer_cfg['name']}' is missing 'roi_folder'.")

    dx_value = sample_cfg.get("dx_mm", tracer_cfg.get("dx_mm"))
    if dx_value in (None, ""):
        raise ValueError(f"Sample in tracer '{tracer_cfg['name']}' is missing 'dx_mm'.")

    run_path = sample_cfg.get("run_path")
    run_rel = sample_cfg.get("run_rel")
    if run_path not in (None, ""):
        resolved_run = str(Path(run_path).resolve())
        paths_used = build_profile_abs_paths(resolved_run, roi_folder)
        loader_required = load_csv_required_path
        loader_optional = load_csv_optional_path
        selection_mode = "run_path"
        run_display = resolved_run
    elif run_rel not in (None, ""):
        if results_source in (None, ""):
            raise ValueError(f"Sample '{sample_cfg.get('sample_id', '<unknown>')}' uses run_rel but config has no results_source.")
        resolved_run = _normalize_rel_path(run_rel)
        paths_used = build_profile_rel_paths(resolved_run, roi_folder)
        loader_required = lambda rel_path: load_csv_required_any(results_source, rel_path)
        loader_optional = lambda rel_path: load_csv_optional_any(results_source, rel_path)
        selection_mode = "run_rel"
        run_display = resolved_run
    else:
        raise ValueError(f"Sample in tracer '{tracer_cfg['name']}' must define either 'run_path' or 'run_rel'.")

    measured_df = loader_required(paths_used["measured_profiles_csv"])
    fitted_df = loader_required(paths_used["fitted_profiles_csv"])
    params_df = loader_required(paths_used["fit_parameters_csv"])
    measured = to_numeric_2d(measured_df)
    fitted = to_numeric_2d(fitted_df)
    if measured.shape != fitted.shape:
        raise ValueError(
            f"Measured/fitted shape mismatch for sample '{sample_cfg.get('sample_id', run_display)}': "
            f"{measured.shape} vs {fitted.shape}"
        )

    fit_std_df = loader_optional(paths_used["fit_std_csv"])
    hu_std_df = loader_optional(paths_used["hu_noise_std_csv"])
    calib_std_df = loader_optional(paths_used["calibration_std_csv"])
    combined_std_df = loader_optional(paths_used["combined_std_csv"])
    combined_ci_low_df = loader_optional(paths_used["combined_ci_low_csv"])
    combined_ci_high_df = loader_optional(paths_used["combined_ci_high_csv"])

    fit_std = to_numeric_2d(fit_std_df) if fit_std_df is not None else None
    hu_std = to_numeric_2d(hu_std_df) if hu_std_df is not None else None
    calib_std = to_numeric_2d(calib_std_df) if calib_std_df is not None else None
    combined_std = to_numeric_2d(combined_std_df) if combined_std_df is not None else None
    combined_ci_low = to_numeric_2d(combined_ci_low_df) if combined_ci_low_df is not None else None
    combined_ci_high = to_numeric_2d(combined_ci_high_df) if combined_ci_high_df is not None else None
    roi_fixed_std = build_roi_fixed_std(fit_std, hu_std, calib_std)

    time_col = find_time_column(params_df)
    times = pd.to_numeric(params_df[time_col], errors="coerce").to_numpy(dtype=float)
    if len(times) != measured.shape[0]:
        raise ValueError(
            f"Time vector length ({len(times)}) does not match profile rows ({measured.shape[0]}) "
            f"for sample '{sample_cfg.get('sample_id', run_display)}'."
        )

    metric_series = {
        metric_name: extract_optional_numeric_series(params_df, candidates)
        for metric_name, candidates in METRIC_COLUMN_CANDIDATES.items()
    }
    r2 = metric_series.get("profile_fit_r2")
    valid_rows = np.any(np.isfinite(fitted), axis=1)

    sample_id = sample_cfg.get("sample_id")
    if sample_id in (None, ""):
        sample_id = Path(run_display).name

    return SampleData(
        tracer_name=tracer_cfg["name"],
        tracer_label=tracer_cfg["label"],
        tracer_color=tracer_cfg["color"],
        tracer_marker=tracer_cfg["marker"],
        sample_id=str(sample_id),
        roi_folder=str(roi_folder),
        dx_mm=float(dx_value),
        run_display=run_display,
        selection_mode=selection_mode,
        paths_used=paths_used,
        measured=measured,
        fitted=fitted,
        fit_params_df=params_df,
        times=times,
        r2=r2,
        fit_std=fit_std,
        hu_std=hu_std,
        calib_std=calib_std,
        combined_std=combined_std,
        combined_ci_low=combined_ci_low,
        combined_ci_high=combined_ci_high,
        roi_fixed_std=roi_fixed_std,
        depth=build_depth_mm(measured.shape[1], float(dx_value)),
        valid_rows=valid_rows,
        metric_series=metric_series,
    )


def write_tables(
    out_folder: str,
    metrics_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
) -> List[str]:
    summary_dir = Path(out_folder) / "profile_fit_examples" / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    outputs = []

    metrics_path = summary_dir / "per_sample_target_time_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    outputs.append(str(metrics_path))

    group_path = summary_dir / "tracer_group_metric_summary.csv"
    group_summary_df.to_csv(group_path, index=False)
    outputs.append(str(group_path))

    pairwise_path = summary_dir / "pairwise_significance_tests.csv"
    pairwise_df.to_csv(pairwise_path, index=False)
    outputs.append(str(pairwise_path))

    omnibus_path = summary_dir / "omnibus_significance_tests.csv"
    omnibus_df.to_csv(omnibus_path, index=False)
    outputs.append(str(omnibus_path))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fitted concentration-depth profiles across multiple tracers and replicate samples."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON comparison config. Example: configs/comparison/profile_fit_examples_vis_vs_gad.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = validate_and_resolve_config(load_config(config_path), config_path)

    samples_by_tracer: Dict[str, List[SampleData]] = {}
    tracer_styles: Dict[str, Dict[str, str]] = {}
    all_samples: List[SampleData] = []

    for tracer_cfg in config["tracers"]:
        tracer_name = tracer_cfg["name"]
        tracer_styles[tracer_name] = {"color": tracer_cfg["color"], "marker": tracer_cfg["marker"]}
        tracer_samples = [
            load_sample_data(tracer_cfg, sample_cfg, config.get("results_source"))
            for sample_cfg in tracer_cfg["samples"]
        ]
        samples_by_tracer[tracer_name] = tracer_samples
        all_samples.extend(tracer_samples)

    target_times = choose_target_times(all_samples, config.get("target_times_min"))
    matched_indices: Dict[str, Dict[float, int]] = {}
    for sample in all_samples:
        matched_indices[sample.sample_id] = {
            target_time: nearest_index(sample.times, target_time, valid_mask=sample.valid_rows)
            for target_time in target_times
        }

    y_arrays = []
    for sample in all_samples:
        y_arrays.extend([sample.measured, sample.fitted])
        if sample.fit_std is not None:
            y_arrays.extend([sample.fitted - sample.fit_std, sample.fitted + sample.fit_std])
        if sample.roi_fixed_std is not None:
            y_arrays.extend([sample.fitted - Z95 * sample.roi_fixed_std, sample.fitted + Z95 * sample.roi_fixed_std])
        if sample.combined_std is not None:
            y_arrays.extend([sample.fitted - Z95 * sample.combined_std, sample.fitted + Z95 * sample.combined_std])
        if sample.combined_ci_low is not None and sample.combined_ci_high is not None:
            y_arrays.extend([sample.combined_ci_low, sample.combined_ci_high])
    ylims, _ = compute_axis_limits(*y_arrays)

    tracer_order = [tracer_cfg["name"] for tracer_cfg in config["tracers"]]
    figure_outputs = []
    for mode, subfolder, out_name in PLOT_MODE_SPECS:
        figure_outputs.append(
            save_profile_figure(
                out_folder=config["out_folder"],
                out_name=out_name,
                subfolder=subfolder,
                mode=mode,
                samples_by_tracer=samples_by_tracer,
                tracer_order=tracer_order,
                target_times=target_times,
                matched_indices=matched_indices,
                tracer_styles=tracer_styles,
                figure_title=config["figure_title"],
                ylims=ylims,
                time_unit_label=config["time_unit_label"],
                concentration_label=config["concentration_label"],
            )
        )

    metric_rows = []
    for sample in all_samples:
        for target_time in target_times:
            metric_rows.append(compute_sample_metric_row(sample, target_time, matched_indices[sample.sample_id][target_time]))
    metrics_df = pd.DataFrame(metric_rows)

    stats_metrics = [metric for metric in config["stats_metrics"] if metric in metrics_df.columns]
    group_summary_df = build_group_summary(metrics_df, stats_metrics)
    pairwise_df, omnibus_df = build_statistics_tables(metrics_df, stats_metrics, alpha=config["stats_alpha"])
    table_outputs = write_tables(config["out_folder"], metrics_df, group_summary_df, pairwise_df, omnibus_df)

    audit_payload = {
        "config_path": str(config_path),
        "results_source": config.get("results_source"),
        "out_folder": config["out_folder"],
        "figure_title": config["figure_title"],
        "target_times_requested_min": config.get("target_times_min"),
        "target_times_used_min": [float(x) for x in target_times],
        "stats_metrics": stats_metrics,
        "stats_alpha": float(config["stats_alpha"]),
        "ylims_used": [float(ylims[0]), float(ylims[1])],
        "tracer_count": len(samples_by_tracer),
        "sample_count": len(all_samples),
        "tracers": {
            tracer_name: {
                "label": samples_by_tracer[tracer_name][0].tracer_label,
                "sample_ids": [sample.sample_id for sample in tracer_samples],
                "sample_runs": [sample.run_display for sample in tracer_samples],
                "roi_folders": sorted({sample.roi_folder for sample in tracer_samples}),
            }
            for tracer_name, tracer_samples in samples_by_tracer.items()
        },
        "matched_times_by_sample": {
            sample.sample_id: {
                f"{target_time:.6f}": {
                    "matched_index": int(matched_indices[sample.sample_id][target_time]),
                    "actual_time_min": float(sample.times[matched_indices[sample.sample_id][target_time]]),
                }
                for target_time in target_times
            }
            for sample in all_samples
        },
        "figure_outputs": figure_outputs,
        "table_outputs": table_outputs,
    }
    save_audit_report(config["out_folder"], audit_payload)

    print("Saved profile-fit comparison outputs under:", str(Path(config["out_folder"]) / "profile_fit_examples"))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
