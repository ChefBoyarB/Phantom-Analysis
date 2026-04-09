
import io
import os
import json
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# USER INPUT
# ============================================================
# Point this to either:
#   1) the Results_Paper_1.zip file, or
#   2) an extracted Results_Paper_1 folder
RESULTS_SOURCE = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1"

# -----------------------------------------------------------------
# DATASET SELECTION
# -----------------------------------------------------------------
# BEST PRACTICE FOR FINAL PAPER OUTPUTS:
# set the four values below for the exact VIS/GAD run folder and ROI folder
# you want to compare. This is the most robust option.
#
# Examples relative to RESULTS_SOURCE:
# VIS_RUN_REL = "VIS320_Swollen_No_Pressure_135kvp"
# VIS_ROI_FOLDER = "VIS_320"
# GAD_RUN_REL = "GAD_Swollen_No_Pressure_135kvp"
# GAD_ROI_FOLDER = "GAD"
#
# This mirrors the time-course comparison workflow:
# - explicit run folders are the most robust choice for final paper outputs
# - if VIS_RUN_REL / GAD_RUN_REL are left as None, the script falls back to hint search.
VIS_RUN_REL = r"VIS320_Swollen_No_Pressure_135kvp"
GAD_RUN_REL = r"GAD_Swollen_No_Pressure_135kvp"
VIS_ROI_FOLDER = "VIS_320"
GAD_ROI_FOLDER = "GAD"

# Fallback hints used only when VIS_RUN_REL / GAD_RUN_REL are None.
VIS_FOLDER_HINT = "VIS320_Swollen_No_Pressure_135kvp"
GAD_FOLDER_HINT = "GAD_Swollen_No_Pressure_135kvp"
VIS_ROI_FOLDER_HINT = "VIS_320"
GAD_ROI_FOLDER_HINT = "GAD"

OUT_FOLDER = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\Paper_1_Comparisons"

# Depth spacing from run_metadata.json or your analysis settings
VIS_DX_MM = 0.166
GAD_DX_MM = 0.166

# Same approximate comparison times for both tracers
MANUAL_TARGET_TIMES_MIN = [0.20, 11.0, 22.0]   # set to None to auto-pick overlap start / mid / end

VIS_LABEL = "VIS 320"
GAD_LABEL = "GAD"
TIME_UNIT_LABEL = "min"
CONCENTRATION_LABEL = "Concentration (mg/mL)"

# Styling
FIT_ONLY_FILL_ALPHA = 0.10
ROI_FIXED_FILL_ALPHA = 0.08
ROI_FIXED_1SD_FILL_ALPHA = 0.10
COMBINED_FILL_ALPHA = 0.09
COMBINED_1SD_FILL_ALPHA = 0.10
ROI_FIXED_BOUND_ALPHA = 0.0    # retained for compatibility; no dashed bounds are drawn
ROI_FIXED_BOUND_LINEWIDTH = 1.3
ROI_FIXED_BOUND_LINESTYLE = "--"
CENTER_LINEWIDTH = 2.0
MEASURED_MARKER_SIZE = 18
Z95 = 1.96

# ============================================================
# FILE HELPERS
# ============================================================
def _is_zip_path(path: str) -> bool:
    return str(path).lower().endswith('.zip')


def _normalize_rel_path(rel_path: str) -> str:
    return str(rel_path).replace('\\', '/').strip('/')


def _list_paths_any(base: str) -> Sequence[str]:
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return zf.namelist()
    base_path = Path(base)
    out = []
    for p in base_path.rglob('*'):
        if p.is_file():
            out.append(str(p.relative_to(base_path)).replace('\\', '/'))
    return out


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


def find_run_folder(base: str, folder_hint: str) -> str:
    paths = _list_paths_any(base)
    candidates = sorted({p.split('/')[0] for p in paths if folder_hint in p})
    if not candidates:
        raise FileNotFoundError(f"Could not find a run folder matching hint: {folder_hint}")
    return candidates[0]


def resolve_run_folder(base: str, explicit_run_rel: Optional[str], folder_hint: str, tracer_label: str) -> Tuple[str, str]:
    explicit_run_rel = None if explicit_run_rel in (None, '') else _normalize_rel_path(explicit_run_rel)
    if explicit_run_rel is not None:
        # Accept either a directory path or absolute-like path under RESULTS_SOURCE.
        if _path_exists_any(base, explicit_run_rel):
            return explicit_run_rel, 'explicit'
        # If exact folder path doesn't exist, still allow it if files under it exist later.
        return explicit_run_rel, 'explicit'
    return find_run_folder(base, folder_hint), 'hint_search'


def build_profile_rel_paths(run_rel: str, roi_folder: str) -> dict:
    run_rel = _normalize_rel_path(run_rel)
    roi_folder = str(roi_folder).strip('/').strip('\\')
    base = f"{run_rel}/{roi_folder}"
    return {
        'measured_profiles_csv': f"{base}/CSVs_Profiles/measured_profiles_depth_vs_time.csv",
        'fitted_profiles_csv': f"{base}/CSVs_Profiles/fitted_profiles_depth_vs_time.csv",
        'fit_parameters_csv': f"{base}/CSVs_Summaries/fit_parameters_vs_time.csv",
        'fit_std_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_std_depth_vs_time.csv",
        'fit_ci_low_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_ci_low_depth_vs_time.csv",
        'fit_ci_high_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_ci_high_depth_vs_time.csv",
        'hu_noise_std_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_hu_noise_std_depth_vs_time.csv",
        'roi_sensitivity_std_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_roi_sensitivity_std_depth_vs_time.csv",
        'calibration_std_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_calibration_std_depth_vs_time.csv",
        'combined_std_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_combined_std_depth_vs_time.csv",
        'combined_ci_low_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_combined_ci_low_depth_vs_time.csv",
        'combined_ci_high_csv': f"{base}/CSVs_Uncertainty/fitted_profiles_combined_ci_high_depth_vs_time.csv",
    }



def save_audit_report(out_folder: str, payload: dict):
    out_dir = Path(out_folder) / 'profile_fit_examples' / 'audit'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'comparison_audit.json', 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f'{key}:')
            for item in value:
                lines.append(f'  - {item}')
        elif isinstance(value, dict):
            lines.append(f'{key}:')
            for k, v in value.items():
                lines.append(f'  {k}: {v}')
        else:
            lines.append(f'{key}: {value}')
    with open(out_dir / 'comparison_audit.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

# ============================================================
# DATA HELPERS
# ============================================================
def load_csv_required_any(base: str, relative_path: str) -> pd.DataFrame:
    if not _path_exists_any(base, relative_path):
        raise FileNotFoundError(f"Required file not found: {relative_path}")
    return _read_csv_any(base, relative_path)


def load_csv_optional_any(base: str, relative_path: Optional[str]) -> Optional[pd.DataFrame]:
    if relative_path is None or str(relative_path).strip() == '':
        return None
    if not _path_exists_any(base, relative_path):
        print(f"Optional file not found, skipping: {relative_path}")
        return None
    return _read_csv_any(base, relative_path)


def find_time_column(df: pd.DataFrame) -> str:
    candidates = ['time', 'time_plot', 'time_min', 'time_minutes']
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        'Could not find a time column. Checked: ' + ', '.join(candidates) +
        '\nAvailable columns:\n' + '\n'.join(df.columns.tolist())
    )


def find_r2_column(df: pd.DataFrame) -> Optional[str]:
    for col in ['profile_fit_r2', 'r2', 'R2']:
        if col in df.columns:
            return col
    return None


def to_numeric_2d(df: pd.DataFrame) -> np.ndarray:
    arr = df.apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
    if arr.ndim != 2:
        raise ValueError('Expected a 2D numeric array.')
    return arr


def nearest_index(values: np.ndarray, target: float, valid_mask: Optional[np.ndarray] = None) -> int:
    vals = np.asarray(values, dtype=float)
    if valid_mask is None:
        valid_mask = np.isfinite(vals)
    if not np.any(valid_mask):
        raise ValueError('No valid values available for nearest-index selection.')
    idx_valid = np.where(valid_mask)[0]
    nearest_pos = np.argmin(np.abs(vals[idx_valid] - float(target)))
    return int(idx_valid[nearest_pos])


def choose_target_times(vis_times: np.ndarray, gad_times: np.ndarray) -> List[float]:
    if MANUAL_TARGET_TIMES_MIN is not None:
        if len(MANUAL_TARGET_TIMES_MIN) != 3:
            raise ValueError('MANUAL_TARGET_TIMES_MIN must contain exactly 3 times.')
        return [float(x) for x in MANUAL_TARGET_TIMES_MIN]

    vis_valid = vis_times[np.isfinite(vis_times)]
    gad_valid = gad_times[np.isfinite(gad_times)]
    overlap_start = max(float(np.min(vis_valid)), float(np.min(gad_valid)))
    overlap_end = min(float(np.max(vis_valid)), float(np.max(gad_valid)))
    return [overlap_start, 0.5 * (overlap_start + overlap_end), overlap_end]


def build_depth_mm(n_depth: int, dx_mm: float) -> np.ndarray:
    return np.arange(n_depth, dtype=float) * float(dx_mm)


def build_roi_fixed_std(fit_std: Optional[np.ndarray],
                        hu_std: Optional[np.ndarray],
                        calib_std: Optional[np.ndarray]) -> Optional[np.ndarray]:
    parts = []
    for arr in [fit_std, hu_std, calib_std]:
        if arr is not None:
            parts.append(np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    if not parts:
        return None
    return np.sqrt(np.sum([p ** 2 for p in parts], axis=0))


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


def add_panel_title(ax,
                    panel_name: str,
                    target_t: float,
                    vis_t: float,
                    gad_t: float,
                    vis_r2: Optional[float],
                    gad_r2: Optional[float]):
    title = f"{panel_name} (~{target_t:.1f} {TIME_UNIT_LABEL})"
    subtitle = f"Actual frames: {VIS_LABEL} {vis_t:.2f} {TIME_UNIT_LABEL}, {GAD_LABEL} {gad_t:.2f} {TIME_UNIT_LABEL}"
    if vis_r2 is not None and gad_r2 is not None and np.isfinite(vis_r2) and np.isfinite(gad_r2):
        subtitle += f"\nR²: {VIS_LABEL}={vis_r2:.4f}, {GAD_LABEL}={gad_r2:.4f}"
    ax.set_title(title + '\n' + subtitle)


def save_plot(fig, subfolder: str, out_name: str):
    folder = Path(OUT_FOLDER) / 'profile_fit_examples' / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / out_name, dpi=300, bbox_inches='tight')
    plt.close(fig)

# ============================================================
# PLOTTING
# ============================================================
def save_profile_figure(out_name: str,
                        subfolder: str,
                        vis_measured: np.ndarray,
                        vis_fitted: np.ndarray,
                        gad_measured: np.ndarray,
                        gad_fitted: np.ndarray,
                        vis_times: np.ndarray,
                        gad_times: np.ndarray,
                        vis_indices: List[int],
                        gad_indices: List[int],
                        target_times: List[float],
                        vis_depth: np.ndarray,
                        gad_depth: np.ndarray,
                        ylims: tuple[float, float],
                        vis_r2: Optional[np.ndarray] = None,
                        gad_r2: Optional[np.ndarray] = None,
                        vis_fit_std: Optional[np.ndarray] = None,
                        gad_fit_std: Optional[np.ndarray] = None,
                        vis_roi_fixed_std: Optional[np.ndarray] = None,
                        gad_roi_fixed_std: Optional[np.ndarray] = None,
                        vis_combined_std: Optional[np.ndarray] = None,
                        gad_combined_std: Optional[np.ndarray] = None,
                        vis_combined_ci_low: Optional[np.ndarray] = None,
                        vis_combined_ci_high: Optional[np.ndarray] = None,
                        gad_combined_ci_low: Optional[np.ndarray] = None,
                        gad_combined_ci_high: Optional[np.ndarray] = None,
                        mode: str = 'none'):
    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True, sharey=True)
    panel_names = ['Beginning', 'Middle', 'End']

    for ax, panel_name, target_t, vis_idx, gad_idx in zip(axes, panel_names, target_times, vis_indices, gad_indices):
        vis_meas = vis_measured[vis_idx]
        vis_fit = vis_fitted[vis_idx]
        gad_meas = gad_measured[gad_idx]
        gad_fit = gad_fitted[gad_idx]

        vis_fit_line, = ax.plot(vis_depth, vis_fit, linewidth=CENTER_LINEWIDTH, label=f'{VIS_LABEL} fit')
        ax.scatter(vis_depth, vis_meas, s=MEASURED_MARKER_SIZE, marker='o', label=f'{VIS_LABEL} measured')
        gad_fit_line, = ax.plot(gad_depth, gad_fit, linewidth=CENTER_LINEWIDTH, label=f'{GAD_LABEL} fit')
        ax.scatter(gad_depth, gad_meas, s=MEASURED_MARKER_SIZE, marker='s', label=f'{GAD_LABEL} measured')

        if mode == 'fit_only':
            if vis_fit_std is not None:
                vstd = vis_fit_std[vis_idx]
                valid = np.isfinite(vis_depth) & np.isfinite(vis_fit) & np.isfinite(vstd)
                if np.any(valid):
                    ax.fill_between(vis_depth[valid], (vis_fit - vstd)[valid], (vis_fit + vstd)[valid],
                                    color=vis_fit_line.get_color(), alpha=FIT_ONLY_FILL_ALPHA)
            if gad_fit_std is not None:
                gstd = gad_fit_std[gad_idx]
                valid = np.isfinite(gad_depth) & np.isfinite(gad_fit) & np.isfinite(gstd)
                if np.any(valid):
                    ax.fill_between(gad_depth[valid], (gad_fit - gstd)[valid], (gad_fit + gstd)[valid],
                                    color=gad_fit_line.get_color(), alpha=FIT_ONLY_FILL_ALPHA)

        if mode == 'roi_fixed':
            if vis_roi_fixed_std is not None:
                vstd = vis_roi_fixed_std[vis_idx]
                valid = np.isfinite(vis_depth) & np.isfinite(vis_fit) & np.isfinite(vstd)
                if np.any(valid):
                    low = vis_fit - Z95 * vstd
                    high = vis_fit + Z95 * vstd
                    ax.fill_between(vis_depth[valid], low[valid], high[valid],
                                    color=vis_fit_line.get_color(), alpha=ROI_FIXED_FILL_ALPHA)
            if gad_roi_fixed_std is not None:
                gstd = gad_roi_fixed_std[gad_idx]
                valid = np.isfinite(gad_depth) & np.isfinite(gad_fit) & np.isfinite(gstd)
                if np.any(valid):
                    low = gad_fit - Z95 * gstd
                    high = gad_fit + Z95 * gstd
                    ax.fill_between(gad_depth[valid], low[valid], high[valid],
                                    color=gad_fit_line.get_color(), alpha=ROI_FIXED_FILL_ALPHA)

        if mode == 'roi_fixed_1sd':
            if vis_roi_fixed_std is not None:
                vstd = vis_roi_fixed_std[vis_idx]
                valid = np.isfinite(vis_depth) & np.isfinite(vis_fit) & np.isfinite(vstd)
                if np.any(valid):
                    ax.fill_between(vis_depth[valid], (vis_fit - vstd)[valid], (vis_fit + vstd)[valid],
                                    color=vis_fit_line.get_color(), alpha=ROI_FIXED_1SD_FILL_ALPHA)
            if gad_roi_fixed_std is not None:
                gstd = gad_roi_fixed_std[gad_idx]
                valid = np.isfinite(gad_depth) & np.isfinite(gad_fit) & np.isfinite(gstd)
                if np.any(valid):
                    ax.fill_between(gad_depth[valid], (gad_fit - gstd)[valid], (gad_fit + gstd)[valid],
                                    color=gad_fit_line.get_color(), alpha=ROI_FIXED_1SD_FILL_ALPHA)

        if mode == 'combined_95ci':
            if vis_combined_ci_low is not None and vis_combined_ci_high is not None:
                low = vis_combined_ci_low[vis_idx]
                high = vis_combined_ci_high[vis_idx]
                valid = np.isfinite(vis_depth) & np.isfinite(low) & np.isfinite(high)
                if np.any(valid):
                    ax.fill_between(vis_depth[valid], low[valid], high[valid],
                                    color=vis_fit_line.get_color(), alpha=COMBINED_FILL_ALPHA)
            elif vis_combined_std is not None:
                vstd = vis_combined_std[vis_idx]
                valid = np.isfinite(vis_depth) & np.isfinite(vis_fit) & np.isfinite(vstd)
                if np.any(valid):
                    low = vis_fit - Z95 * vstd
                    high = vis_fit + Z95 * vstd
                    ax.fill_between(vis_depth[valid], low[valid], high[valid],
                                    color=vis_fit_line.get_color(), alpha=COMBINED_FILL_ALPHA)
            if gad_combined_ci_low is not None and gad_combined_ci_high is not None:
                low = gad_combined_ci_low[gad_idx]
                high = gad_combined_ci_high[gad_idx]
                valid = np.isfinite(gad_depth) & np.isfinite(low) & np.isfinite(high)
                if np.any(valid):
                    ax.fill_between(gad_depth[valid], low[valid], high[valid],
                                    color=gad_fit_line.get_color(), alpha=COMBINED_FILL_ALPHA)
            elif gad_combined_std is not None:
                gstd = gad_combined_std[gad_idx]
                valid = np.isfinite(gad_depth) & np.isfinite(gad_fit) & np.isfinite(gstd)
                if np.any(valid):
                    low = gad_fit - Z95 * gstd
                    high = gad_fit + Z95 * gstd
                    ax.fill_between(gad_depth[valid], low[valid], high[valid],
                                    color=gad_fit_line.get_color(), alpha=COMBINED_FILL_ALPHA)

        if mode == 'combined_1sd':
            if vis_combined_std is not None:
                vstd = vis_combined_std[vis_idx]
                valid = np.isfinite(vis_depth) & np.isfinite(vis_fit) & np.isfinite(vstd)
                if np.any(valid):
                    ax.fill_between(vis_depth[valid], (vis_fit - vstd)[valid], (vis_fit + vstd)[valid],
                                    color=vis_fit_line.get_color(), alpha=COMBINED_1SD_FILL_ALPHA)
            if gad_combined_std is not None:
                gstd = gad_combined_std[gad_idx]
                valid = np.isfinite(gad_depth) & np.isfinite(gad_fit) & np.isfinite(gstd)
                if np.any(valid):
                    ax.fill_between(gad_depth[valid], (gad_fit - gstd)[valid], (gad_fit + gstd)[valid],
                                    color=gad_fit_line.get_color(), alpha=COMBINED_1SD_FILL_ALPHA)

        add_panel_title(
            ax,
            panel_name,
            target_t,
            vis_times[vis_idx],
            gad_times[gad_idx],
            vis_r2[vis_idx] if vis_r2 is not None else None,
            gad_r2[gad_idx] if gad_r2 is not None else None,
        )
        ax.set_ylabel(CONCENTRATION_LABEL)
        ax.grid(True)
        ax.set_xlim(0, max(float(np.max(vis_depth)), float(np.max(gad_depth))))
        ax.set_ylim(*ylims)

    axes[-1].set_xlabel('Depth (mm)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.995))
    fig.suptitle('Profile Fit Examples: VIS 320 and GAD', y=1.02, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_plot(fig, subfolder, out_name)

# ============================================================
# LOAD DATA
# ============================================================
vis_run_rel, vis_selection_mode = resolve_run_folder(RESULTS_SOURCE, VIS_RUN_REL, VIS_FOLDER_HINT, 'VIS')
gad_run_rel, gad_selection_mode = resolve_run_folder(RESULTS_SOURCE, GAD_RUN_REL, GAD_FOLDER_HINT, 'GAD')

vis_paths = build_profile_rel_paths(vis_run_rel, VIS_ROI_FOLDER)
gad_paths = build_profile_rel_paths(gad_run_rel, GAD_ROI_FOLDER)

vis_measured = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, vis_paths['measured_profiles_csv']))
vis_fitted = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, vis_paths['fitted_profiles_csv']))
vis_params_df = load_csv_required_any(RESULTS_SOURCE, vis_paths['fit_parameters_csv'])

gad_measured = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, gad_paths['measured_profiles_csv']))
gad_fitted = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, gad_paths['fitted_profiles_csv']))
gad_params_df = load_csv_required_any(RESULTS_SOURCE, gad_paths['fit_parameters_csv'])

vis_fit_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['fit_std_csv'])
vis_fit_ci_low_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['fit_ci_low_csv'])
vis_fit_ci_high_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['fit_ci_high_csv'])
vis_hu_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['hu_noise_std_csv'])
vis_calib_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['calibration_std_csv'])
vis_combined_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['combined_std_csv'])
vis_combined_ci_low_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['combined_ci_low_csv'])
vis_combined_ci_high_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths['combined_ci_high_csv'])

gad_fit_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['fit_std_csv'])
gad_fit_ci_low_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['fit_ci_low_csv'])
gad_fit_ci_high_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['fit_ci_high_csv'])
gad_hu_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['hu_noise_std_csv'])
gad_calib_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['calibration_std_csv'])
gad_combined_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['combined_std_csv'])
gad_combined_ci_low_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['combined_ci_low_csv'])
gad_combined_ci_high_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths['combined_ci_high_csv'])

if vis_measured.shape != vis_fitted.shape:
    raise ValueError(f'VIS measured/fitted shape mismatch: {vis_measured.shape} vs {vis_fitted.shape}')
if gad_measured.shape != gad_fitted.shape:
    raise ValueError(f'GAD measured/fitted shape mismatch: {gad_measured.shape} vs {gad_fitted.shape}')

vis_fit_std = to_numeric_2d(vis_fit_std_df) if vis_fit_std_df is not None else None
vis_fit_ci_low = to_numeric_2d(vis_fit_ci_low_df) if vis_fit_ci_low_df is not None else None
vis_fit_ci_high = to_numeric_2d(vis_fit_ci_high_df) if vis_fit_ci_high_df is not None else None
vis_hu_std = to_numeric_2d(vis_hu_std_df) if vis_hu_std_df is not None else None
vis_calib_std = to_numeric_2d(vis_calib_std_df) if vis_calib_std_df is not None else None
vis_combined_std = to_numeric_2d(vis_combined_std_df) if vis_combined_std_df is not None else None
vis_combined_ci_low = to_numeric_2d(vis_combined_ci_low_df) if vis_combined_ci_low_df is not None else None
vis_combined_ci_high = to_numeric_2d(vis_combined_ci_high_df) if vis_combined_ci_high_df is not None else None

gad_fit_std = to_numeric_2d(gad_fit_std_df) if gad_fit_std_df is not None else None
gad_fit_ci_low = to_numeric_2d(gad_fit_ci_low_df) if gad_fit_ci_low_df is not None else None
gad_fit_ci_high = to_numeric_2d(gad_fit_ci_high_df) if gad_fit_ci_high_df is not None else None
gad_hu_std = to_numeric_2d(gad_hu_std_df) if gad_hu_std_df is not None else None
gad_calib_std = to_numeric_2d(gad_calib_std_df) if gad_calib_std_df is not None else None
gad_combined_std = to_numeric_2d(gad_combined_std_df) if gad_combined_std_df is not None else None
gad_combined_ci_low = to_numeric_2d(gad_combined_ci_low_df) if gad_combined_ci_low_df is not None else None
gad_combined_ci_high = to_numeric_2d(gad_combined_ci_high_df) if gad_combined_ci_high_df is not None else None

vis_roi_fixed_std = build_roi_fixed_std(vis_fit_std, vis_hu_std, vis_calib_std)
gad_roi_fixed_std = build_roi_fixed_std(gad_fit_std, gad_hu_std, gad_calib_std)

vis_time_col = find_time_column(vis_params_df)
gad_time_col = find_time_column(gad_params_df)

vis_times = pd.to_numeric(vis_params_df[vis_time_col], errors='coerce').to_numpy(dtype=float)
gad_times = pd.to_numeric(gad_params_df[gad_time_col], errors='coerce').to_numpy(dtype=float)

if len(vis_times) != vis_measured.shape[0]:
    raise ValueError(f'VIS time vector length ({len(vis_times)}) does not match profile rows ({vis_measured.shape[0]}).')
if len(gad_times) != gad_measured.shape[0]:
    raise ValueError(f'GAD time vector length ({len(gad_times)}) does not match profile rows ({gad_measured.shape[0]}).')

vis_r2_col = find_r2_column(vis_params_df)
gad_r2_col = find_r2_column(gad_params_df)
vis_r2 = pd.to_numeric(vis_params_df[vis_r2_col], errors='coerce').to_numpy(dtype=float) if vis_r2_col else None
gad_r2 = pd.to_numeric(gad_params_df[gad_r2_col], errors='coerce').to_numpy(dtype=float) if gad_r2_col else None

vis_valid_rows = np.any(np.isfinite(vis_fitted), axis=1)
gad_valid_rows = np.any(np.isfinite(gad_fitted), axis=1)

target_times = choose_target_times(vis_times[vis_valid_rows], gad_times[gad_valid_rows])
vis_indices = [nearest_index(vis_times, t, valid_mask=vis_valid_rows) for t in target_times]
gad_indices = [nearest_index(gad_times, t, valid_mask=gad_valid_rows) for t in target_times]

vis_depth = build_depth_mm(vis_measured.shape[1], VIS_DX_MM)
gad_depth = build_depth_mm(gad_measured.shape[1], GAD_DX_MM)

ylims, _ = compute_axis_limits(
    vis_measured, vis_fitted, gad_measured, gad_fitted,
    vis_fitted - (vis_fit_std if vis_fit_std is not None else 0),
    vis_fitted + (vis_fit_std if vis_fit_std is not None else 0),
    gad_fitted - (gad_fit_std if gad_fit_std is not None else 0),
    gad_fitted + (gad_fit_std if gad_fit_std is not None else 0),
)

# ============================================================
# OUTPUTS
# ============================================================
save_profile_figure(
    out_name='vis_vs_gad_profile_fit_examples_no_uncertainty.png',
    subfolder='no_uncertainty',
    vis_measured=vis_measured,
    vis_fitted=vis_fitted,
    gad_measured=gad_measured,
    gad_fitted=gad_fitted,
    vis_times=vis_times,
    gad_times=gad_times,
    vis_indices=vis_indices,
    gad_indices=gad_indices,
    target_times=target_times,
    vis_depth=vis_depth,
    gad_depth=gad_depth,
    ylims=ylims,
    vis_r2=vis_r2,
    gad_r2=gad_r2,
    mode='none',
)

save_profile_figure(
    out_name='vis_vs_gad_profile_fit_examples_model_fit_only_uncertainty.png',
    subfolder='model_fit_only',
    vis_measured=vis_measured,
    vis_fitted=vis_fitted,
    gad_measured=gad_measured,
    gad_fitted=gad_fitted,
    vis_times=vis_times,
    gad_times=gad_times,
    vis_indices=vis_indices,
    gad_indices=gad_indices,
    target_times=target_times,
    vis_depth=vis_depth,
    gad_depth=gad_depth,
    ylims=ylims,
    vis_r2=vis_r2,
    gad_r2=gad_r2,
    vis_fit_std=vis_fit_std,
    gad_fit_std=gad_fit_std,
    mode='fit_only',
)

save_profile_figure(
    out_name='vis_vs_gad_profile_fit_examples_combined_95CI.png',
    subfolder='combined_95CI',
    vis_measured=vis_measured,
    vis_fitted=vis_fitted,
    gad_measured=gad_measured,
    gad_fitted=gad_fitted,
    vis_times=vis_times,
    gad_times=gad_times,
    vis_indices=vis_indices,
    gad_indices=gad_indices,
    target_times=target_times,
    vis_depth=vis_depth,
    gad_depth=gad_depth,
    ylims=ylims,
    vis_r2=vis_r2,
    gad_r2=gad_r2,
    vis_combined_std=vis_combined_std,
    gad_combined_std=gad_combined_std,
    vis_combined_ci_low=vis_combined_ci_low,
    vis_combined_ci_high=vis_combined_ci_high,
    gad_combined_ci_low=gad_combined_ci_low,
    gad_combined_ci_high=gad_combined_ci_high,
    mode='combined_95ci',
)

save_profile_figure(
    out_name='vis_vs_gad_profile_fit_examples_combined_1SD.png',
    subfolder='combined_1SD',
    vis_measured=vis_measured,
    vis_fitted=vis_fitted,
    gad_measured=gad_measured,
    gad_fitted=gad_fitted,
    vis_times=vis_times,
    gad_times=gad_times,
    vis_indices=vis_indices,
    gad_indices=gad_indices,
    target_times=target_times,
    vis_depth=vis_depth,
    gad_depth=gad_depth,
    ylims=ylims,
    vis_r2=vis_r2,
    gad_r2=gad_r2,
    vis_combined_std=vis_combined_std,
    gad_combined_std=gad_combined_std,
    mode='combined_1sd',
)

save_profile_figure(
    out_name='vis_vs_gad_profile_fit_examples_fixedROI_95CI.png',
    subfolder='fixedROI_95CI',
    vis_measured=vis_measured,
    vis_fitted=vis_fitted,
    gad_measured=gad_measured,
    gad_fitted=gad_fitted,
    vis_times=vis_times,
    gad_times=gad_times,
    vis_indices=vis_indices,
    gad_indices=gad_indices,
    target_times=target_times,
    vis_depth=vis_depth,
    gad_depth=gad_depth,
    ylims=ylims,
    vis_r2=vis_r2,
    gad_r2=gad_r2,
    vis_roi_fixed_std=vis_roi_fixed_std,
    gad_roi_fixed_std=gad_roi_fixed_std,
    mode='roi_fixed',
)

save_profile_figure(
    out_name='vis_vs_gad_profile_fit_examples_fixedROI_1SD.png',
    subfolder='fixedROI_1SD',
    vis_measured=vis_measured,
    vis_fitted=vis_fitted,
    gad_measured=gad_measured,
    gad_fitted=gad_fitted,
    vis_times=vis_times,
    gad_times=gad_times,
    vis_indices=vis_indices,
    gad_indices=gad_indices,
    target_times=target_times,
    vis_depth=vis_depth,
    gad_depth=gad_depth,
    ylims=ylims,
    vis_r2=vis_r2,
    gad_r2=gad_r2,
    vis_roi_fixed_std=vis_roi_fixed_std,
    gad_roi_fixed_std=gad_roi_fixed_std,
    mode='roi_fixed_1sd',
)

audit_payload = {
    'results_source': RESULTS_SOURCE,
    'vis_selection_mode': vis_selection_mode,
    'gad_selection_mode': gad_selection_mode,
    'vis_run_rel': vis_run_rel,
    'gad_run_rel': gad_run_rel,
    'vis_roi_folder': VIS_ROI_FOLDER,
    'gad_roi_folder': GAD_ROI_FOLDER,
    'vis_paths_used': vis_paths,
    'gad_paths_used': gad_paths,
    'vis_optional_files_found': {
        'fit_std_csv': vis_fit_std_df is not None,
        'fit_ci_low_csv': vis_fit_ci_low_df is not None,
        'fit_ci_high_csv': vis_fit_ci_high_df is not None,
        'hu_noise_std_csv': vis_hu_std_df is not None,
        'calibration_std_csv': vis_calib_std_df is not None,
        'combined_std_csv': vis_combined_std_df is not None,
        'combined_ci_low_csv': vis_combined_ci_low_df is not None,
        'combined_ci_high_csv': vis_combined_ci_high_df is not None,
    },
    'gad_optional_files_found': {
        'fit_std_csv': gad_fit_std_df is not None,
        'fit_ci_low_csv': gad_fit_ci_low_df is not None,
        'fit_ci_high_csv': gad_fit_ci_high_df is not None,
        'hu_noise_std_csv': gad_hu_std_df is not None,
        'calibration_std_csv': gad_calib_std_df is not None,
        'combined_std_csv': gad_combined_std_df is not None,
        'combined_ci_low_csv': gad_combined_ci_low_df is not None,
        'combined_ci_high_csv': gad_combined_ci_high_df is not None,
    },
    'manual_target_times_min': MANUAL_TARGET_TIMES_MIN,
    'target_times_used_min': [float(x) for x in target_times],
    'vis_indices_used': vis_indices,
    'gad_indices_used': gad_indices,
    'vis_actual_times_min': [float(vis_times[i]) for i in vis_indices],
    'gad_actual_times_min': [float(gad_times[i]) for i in gad_indices],
    'vis_r2_column': vis_r2_col,
    'gad_r2_column': gad_r2_col,
    'vis_time_column': vis_time_col,
    'gad_time_column': gad_time_col,
    'vis_measured_shape': list(vis_measured.shape),
    'vis_fitted_shape': list(vis_fitted.shape),
    'gad_measured_shape': list(gad_measured.shape),
    'gad_fitted_shape': list(gad_fitted.shape),
    'vis_depth_points': int(vis_measured.shape[1]),
    'gad_depth_points': int(gad_measured.shape[1]),
    'vis_dx_mm': float(VIS_DX_MM),
    'gad_dx_mm': float(GAD_DX_MM),
    'ylims_used': [float(ylims[0]), float(ylims[1])],
    'outputs_written': [
        'profile_fit_examples/no_uncertainty/vis_vs_gad_profile_fit_examples_no_uncertainty.png',
        'profile_fit_examples/model_fit_only/vis_vs_gad_profile_fit_examples_model_fit_only_uncertainty.png',
        'profile_fit_examples/combined_95CI/vis_vs_gad_profile_fit_examples_combined_95CI.png',
        'profile_fit_examples/combined_1SD/vis_vs_gad_profile_fit_examples_combined_1SD.png',
        'profile_fit_examples/fixedROI_95CI/vis_vs_gad_profile_fit_examples_fixedROI_95CI.png',
        'profile_fit_examples/fixedROI_1SD/vis_vs_gad_profile_fit_examples_fixedROI_1SD.png',
    ],
}
save_audit_report(OUT_FOLDER, audit_payload)

print('Saved audit files to:', str(Path(OUT_FOLDER) / 'profile_fit_examples' / 'audit'))
print('Saved profile-fit example outputs under:', str(Path(OUT_FOLDER) / 'profile_fit_examples'))
print('Done.')
