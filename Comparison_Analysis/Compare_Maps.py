import io
import os
import json
import zipfile
from pathlib import Path
from typing import List, Optional, Tuple, Sequence, Dict

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
# set the exact VIS/GAD run folders and ROI folders below.
#
# Examples relative to RESULTS_SOURCE:
# VIS_RUN_REL = "VIS320_Swollen_No_Pressure_135kvp"
# VIS_ROI_FOLDER = "VIS_320"
# GAD_RUN_REL = "GAD_Swollen_No_Pressure_135kvp"
# GAD_ROI_FOLDER = "GAD"
#
# If VIS_RUN_REL / GAD_RUN_REL are left as None, the script falls back to hint search.
VIS_RUN_REL = r"VIS320_Swollen_No_Pressure_135kvp"
GAD_RUN_REL = r"GAD_Swollen_No_Pressure_135kvp"
VIS_ROI_FOLDER = "VIS_320"
GAD_ROI_FOLDER = "GAD"

# Fallback hints used only when VIS_RUN_REL / GAD_RUN_REL are None.
VIS_FOLDER_HINT = "VIS320_Swollen_No_Pressure_135kvp"
GAD_FOLDER_HINT = "GAD_Swollen_No_Pressure_135kvp"

OUT_FOLDER = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\Paper_1_Comparisons"

VIS_DX_MM = 0.166
GAD_DX_MM = 0.166

MANUAL_TARGET_TIMES_MIN = [0.75, 11.0, 21.5]  # set to None to auto-pick overlap start / mid / end

# Shared x-axis for VIS and GAD heatmaps. Set to None to use the union of both datasets.
SHARED_TIME_AXIS_MIN = None
SHARED_TIME_AXIS_MAX = None

VIS_LABEL = "VIS 320"
GAD_LABEL = "GAD"

# Beginning / middle / end colors used both on the heatmaps and the matched depth-profile plots
TIMEPOINT_COLORS = ["#9652F0", "#00FFFF", "#FF0000"]
TIMEPOINT_NAMES = ["Beginning", "Middle", "End"]

PLOT_DEPTH_ZERO_AT_TOP = True   # True = 0 mm shown at the top of the y-axis for map and line-comparison plots

ROBUST_COLOR_LIMITS = True
LOWER_PERCENTILE = 2.0
UPPER_PERCENTILE = 98.0

PROFILE_Y_PAD_MM = 0.15
PROFILE_FIXED_ROI_FILL_ALPHA = 0.14
PROFILE_FIXED_ROI_Z = 1.96

# ============================================================
# FILE HELPERS
# ============================================================
def _is_zip_path(path: str) -> bool:
    return str(path).lower().endswith(".zip")


def _normalize_rel_path(rel_path: str) -> str:
    return str(rel_path).replace("\\", "/").strip("/")


def _list_paths_any(base: str) -> Sequence[str]:
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return zf.namelist()
    base_path = Path(base)
    out = []
    for p in base_path.rglob("*"):
        if p.is_file():
            out.append(str(p.relative_to(base_path)).replace("\\", "/"))
    return out


def _path_exists_any(base: str, relative_path: str) -> bool:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return rel in zf.namelist()
    return (Path(base) / rel).exists()


def _read_csv_any(base: str, relative_path: str) -> pd.DataFrame:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return pd.read_csv(io.BytesIO(zf.read(rel)))
    return pd.read_csv(Path(base) / rel)


def find_run_folder(base: str, folder_hint: str) -> str:
    paths = _list_paths_any(base)
    candidates = sorted({p.split("/")[0] for p in paths if folder_hint in p})
    if not candidates:
        raise FileNotFoundError(f"Could not find a run folder matching hint: {folder_hint}")
    return candidates[0]


def resolve_run_folder(base: str, explicit_run_rel: Optional[str], folder_hint: str, tracer_label: str) -> Tuple[str, str]:
    explicit_run_rel = None if explicit_run_rel in (None, "") else _normalize_rel_path(explicit_run_rel)
    if explicit_run_rel is not None:
        return explicit_run_rel, "explicit"
    return find_run_folder(base, folder_hint), "hint_search"


def build_map_rel_paths(run_rel: str, roi_folder: str) -> Dict[str, str]:
    run_rel = _normalize_rel_path(run_rel)
    roi_folder = str(roi_folder).strip("/").strip("\\")
    return {
        "fit_parameters_csv": f"{run_rel}/{roi_folder}/CSVs_Summaries/fit_parameters_vs_time.csv",

        "per_timepoint_fitted_profiles_csv": f"{run_rel}/{roi_folder}/CSVs_Profiles/fitted_profiles_depth_vs_time.csv",
        "per_timepoint_flux_magnitude_csv": f"{run_rel}/{roi_folder}/CSVs_Diffusion/diffusive_flux_magnitude_map.csv",

        "temporally_regularized_fitted_profiles_csv": f"{run_rel}/{roi_folder}/CSVs_Profiles/temporally_regularized_fitted_profiles_depth_vs_time.csv",
        "temporally_regularized_flux_magnitude_csv": f"{run_rel}/{roi_folder}/CSVs_Diffusion/temporally_regularized_diffusive_flux_magnitude_map.csv",

        "local_effective_diffusivity_csv": f"{run_rel}/{roi_folder}/CSVs_Diffusion/local_effective_diffusivity_map.csv",

        # Fixed-ROI uncertainty inputs for fitted concentration profile comparisons
        "fitted_profiles_fit_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/fitted_profiles_std_depth_vs_time.csv",
        "fitted_profiles_hu_noise_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/fitted_profiles_hu_noise_std_depth_vs_time.csv",
        "fitted_profiles_calibration_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/fitted_profiles_calibration_std_depth_vs_time.csv",

        # Optional temporally-regularized fitted-profile uncertainty inputs
        "temporally_regularized_fitted_profiles_fit_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_hu_noise_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_hu_noise_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_calibration_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_calibration_std_depth_vs_time.csv",

        # Fixed-ROI uncertainty inputs for per-timepoint diffusive flux magnitude comparisons
        "per_timepoint_flux_magnitude_model_fit_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/diffusive_flux_magnitude_model_fit_std_map.csv",
        "per_timepoint_flux_magnitude_hu_noise_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/diffusive_flux_magnitude_hu_noise_std_map.csv",
        "per_timepoint_flux_magnitude_calibration_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/diffusive_flux_magnitude_calibration_std_map.csv",
        "per_timepoint_flux_magnitude_fixed_roi_std_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/diffusive_flux_magnitude_fixed_roi_std_map.csv",
        "per_timepoint_flux_magnitude_fixed_roi_ci_width_csv": f"{run_rel}/{roi_folder}/CSVs_Uncertainty/diffusive_flux_magnitude_fixed_roi_ci_width_map.csv",
        "per_timepoint_flux_magnitude_fixed_roi_relative_percent_csv": f"{run_rel}/{roi_folder}/CSVs_Diffusion/diffusive_flux_magnitude_fixed_roi_relative_percent_map.csv",
    }


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_audit_report(out_folder: str, payload: dict):
    out_dir = Path(out_folder) / "map_comparisons" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison_audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        elif isinstance(value, dict):
            lines.append(f"{key}:")
            for k, v in value.items():
                lines.append(f"  {k}: {v}")
        else:
            lines.append(f"{key}: {value}")
    with open(out_dir / "comparison_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================
# DATA HELPERS
# ============================================================
def load_csv_required_any(base: str, relative_path: str) -> pd.DataFrame:
    if not _path_exists_any(base, relative_path):
        raise FileNotFoundError(f"Required file not found: {relative_path}")
    return _read_csv_any(base, relative_path)


def load_csv_optional_any(base: str, relative_path: str) -> Optional[pd.DataFrame]:
    if not _path_exists_any(base, relative_path):
        print(f"Optional file not found, skipping: {relative_path}")
        return None
    return _read_csv_any(base, relative_path)


def to_numeric_2d(df: pd.DataFrame) -> np.ndarray:
    arr = df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D numeric array.")
    return arr


def find_time_column(df: pd.DataFrame) -> str:
    candidates = ["time", "time_plot", "time_min", "time_minutes"]
    for col in candidates:
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find time column. Checked: " + ", ".join(candidates) +
        "\nAvailable columns:\n" + "\n".join(df.columns.tolist())
    )


def nearest_index(values: np.ndarray, target: float, valid_mask: Optional[np.ndarray] = None) -> int:
    vals = np.asarray(values, dtype=float)
    if valid_mask is None:
        valid_mask = np.isfinite(vals)
    if not np.any(valid_mask):
        raise ValueError("No valid values available for nearest-index selection.")
    idx_valid = np.where(valid_mask)[0]
    nearest_pos = np.argmin(np.abs(vals[idx_valid] - float(target)))
    return int(idx_valid[nearest_pos])


def choose_target_times(vis_times: np.ndarray, gad_times: np.ndarray) -> List[float]:
    if MANUAL_TARGET_TIMES_MIN is not None:
        if len(MANUAL_TARGET_TIMES_MIN) != 3:
            raise ValueError("MANUAL_TARGET_TIMES_MIN must contain exactly 3 values.")
        return [float(x) for x in MANUAL_TARGET_TIMES_MIN]

    vis_valid = vis_times[np.isfinite(vis_times)]
    gad_valid = gad_times[np.isfinite(gad_times)]
    overlap_start = max(float(np.min(vis_valid)), float(np.min(gad_valid)))
    overlap_end = min(float(np.max(vis_valid)), float(np.max(gad_valid)))
    return [overlap_start, 0.5 * (overlap_start + overlap_end), overlap_end]


def build_depth_mm(n_depth: int, dx_mm: float) -> np.ndarray:
    return np.arange(n_depth, dtype=float) * float(dx_mm)


def get_global_vmin_vmax(a: np.ndarray, b: np.ndarray,
                         robust: bool = True,
                         lower_pct: float = LOWER_PERCENTILE,
                         upper_pct: float = UPPER_PERCENTILE) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    if robust:
        vmin = float(np.percentile(vals, lower_pct))
        vmax = float(np.percentile(vals, upper_pct))
    else:
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))

    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def get_nonnegative_vmin_vmax(a: np.ndarray, b: np.ndarray,
                              robust: bool = True,
                              upper_pct: float = UPPER_PERCENTILE) -> Tuple[float, float]:
    vals = np.concatenate([np.asarray(a, dtype=float).ravel(), np.asarray(b, dtype=float).ravel()])
    vals = vals[np.isfinite(vals)]
    vals = vals[vals >= 0]
    if vals.size == 0:
        return 0.0, 1.0

    vmin = 0.0
    vmax = float(np.percentile(vals, upper_pct)) if robust else float(np.max(vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax


def save_figure(fig, out_path: str, dpi: int = 300):
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def combine_available_std_maps(*maps: Optional[np.ndarray]) -> Optional[np.ndarray]:
    valid_maps = [np.asarray(m, dtype=float) for m in maps if m is not None]
    if not valid_maps:
        return None
    ref_shape = valid_maps[0].shape
    for m in valid_maps[1:]:
        if m.shape != ref_shape:
            raise ValueError(f"Uncertainty map shape mismatch: expected {ref_shape}, got {m.shape}")
    stack = np.stack(valid_maps, axis=0)
    finite_mask = np.isfinite(stack)
    any_finite = np.any(finite_mask, axis=0)
    rss = np.sqrt(np.nansum(np.where(finite_mask, stack, 0.0) ** 2, axis=0))
    rss[~any_finite] = np.nan
    return rss


# ============================================================
# PLOTTING
# ============================================================
def save_heatmap_pair_figure(
    vis_map: np.ndarray,
    gad_map: np.ndarray,
    vis_times: np.ndarray,
    gad_times: np.ndarray,
    vis_depth_mm: np.ndarray,
    gad_depth_mm: np.ndarray,
    title: str,
    cbar_label: str,
    out_path: str,
    nonnegative: bool = False,
    target_times: Optional[List[float]] = None,
    shared_tmin: Optional[float] = None,
    shared_tmax: Optional[float] = None,
):
    vmin, vmax = (
        get_nonnegative_vmin_vmax(vis_map, gad_map, robust=ROBUST_COLOR_LIMITS)
        if nonnegative else
        get_global_vmin_vmax(vis_map, gad_map, robust=ROBUST_COLOR_LIMITS)
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    if shared_tmin is None:
        shared_tmin = float(min(np.nanmin(vis_times), np.nanmin(gad_times)))
    if shared_tmax is None:
        shared_tmax = float(max(np.nanmax(vis_times), np.nanmax(gad_times)))

    vis_extent = [
        float(shared_tmin),
        float(shared_tmax),
        float(np.min(vis_depth_mm)),
        float(np.max(vis_depth_mm))
    ]

    gad_extent = [
        float(shared_tmin),
        float(shared_tmax),
        float(np.min(gad_depth_mm)),
        float(np.max(gad_depth_mm))
    ]

    im0 = axes[0].imshow(
        vis_map.T,
        aspect="auto",
        extent=vis_extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )
    axes[0].set_title(VIS_LABEL)
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Depth (mm)")

    im1 = axes[1].imshow(
        gad_map.T,
        aspect="auto",
        extent=gad_extent,
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )
    axes[1].set_title(GAD_LABEL)
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Depth (mm)")

    axes[0].set_xlim(float(shared_tmin), float(shared_tmax))
    axes[1].set_xlim(float(shared_tmin), float(shared_tmax))

    if PLOT_DEPTH_ZERO_AT_TOP:
        axes[0].invert_yaxis()
        axes[1].invert_yaxis()

    if target_times is not None:
        for idx, target_t in enumerate(target_times[:len(TIMEPOINT_COLORS)]):
            color = TIMEPOINT_COLORS[idx]
            axes[0].axvline(float(target_t), color=color, linestyle="--", linewidth=1.8, alpha=0.95)
            axes[1].axvline(float(target_t), color=color, linestyle="--", linewidth=1.8, alpha=0.95)

    cbar = fig.colorbar(im1, ax=axes.ravel().tolist(), shrink=0.95)
    cbar.set_label(cbar_label)

    fig.suptitle(title, fontsize=14)
    save_figure(fig, out_path)


def save_three_timepoint_comparison(
    vis_map: np.ndarray,
    gad_map: np.ndarray,
    vis_times: np.ndarray,
    gad_times: np.ndarray,
    vis_depth_mm: np.ndarray,
    gad_depth_mm: np.ndarray,
    target_times: List[float],
    title_prefix: str,
    x_label: str,
    out_path: str,
    nonnegative: bool = False,
    vis_std_map: Optional[np.ndarray] = None,
    gad_std_map: Optional[np.ndarray] = None,
    show_fixed_roi_uncertainty: bool = False,
    y_pad_mm: float = PROFILE_Y_PAD_MM,
):
    vis_valid = np.any(np.isfinite(vis_map), axis=1)
    gad_valid = np.any(np.isfinite(gad_map), axis=1)

    vis_indices = [nearest_index(vis_times, t, vis_valid) for t in target_times]
    gad_indices = [nearest_index(gad_times, t, gad_valid) for t in target_times]

    vis_slices = np.vstack([vis_map[i] for i in vis_indices])
    gad_slices = np.vstack([gad_map[i] for i in gad_indices])

    vis_limits = vis_slices.copy()
    gad_limits = gad_slices.copy()
    if show_fixed_roi_uncertainty and vis_std_map is not None:
        vis_std_slices = np.vstack([vis_std_map[i] for i in vis_indices])
        vis_limits = np.vstack([vis_slices - PROFILE_FIXED_ROI_Z * vis_std_slices,
                                vis_slices + PROFILE_FIXED_ROI_Z * vis_std_slices])
    if show_fixed_roi_uncertainty and gad_std_map is not None:
        gad_std_slices = np.vstack([gad_std_map[i] for i in gad_indices])
        gad_limits = np.vstack([gad_slices - PROFILE_FIXED_ROI_Z * gad_std_slices,
                                gad_slices + PROFILE_FIXED_ROI_Z * gad_std_slices])

    vmin, vmax = (
        get_nonnegative_vmin_vmax(vis_limits, gad_limits, robust=ROBUST_COLOR_LIMITS)
        if nonnegative else
        get_global_vmin_vmax(vis_limits, gad_limits, robust=ROBUST_COLOR_LIMITS)
    )

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), sharex=False, sharey=True)
    row_names = TIMEPOINT_NAMES

    for r, (panel_name, target_t, vis_idx, gad_idx) in enumerate(zip(row_names, target_times, vis_indices, gad_indices)):
        vis_vec = vis_map[vis_idx]
        gad_vec = gad_map[gad_idx]

        ax_vis = axes[r, 0]
        ax_gad = axes[r, 1]

        row_color = TIMEPOINT_COLORS[r % len(TIMEPOINT_COLORS)]

        if show_fixed_roi_uncertainty and vis_std_map is not None:
            vis_std_vec = vis_std_map[vis_idx]
            valid = np.isfinite(vis_depth_mm) & np.isfinite(vis_vec) & np.isfinite(vis_std_vec)
            if np.any(valid):
                ax_vis.fill_betweenx(
                    vis_depth_mm[valid],
                    (vis_vec - PROFILE_FIXED_ROI_Z * vis_std_vec)[valid],
                    (vis_vec + PROFILE_FIXED_ROI_Z * vis_std_vec)[valid],
                    color=row_color,
                    alpha=PROFILE_FIXED_ROI_FILL_ALPHA,
                )
        if show_fixed_roi_uncertainty and gad_std_map is not None:
            gad_std_vec = gad_std_map[gad_idx]
            valid = np.isfinite(gad_depth_mm) & np.isfinite(gad_vec) & np.isfinite(gad_std_vec)
            if np.any(valid):
                ax_gad.fill_betweenx(
                    gad_depth_mm[valid],
                    (gad_vec - PROFILE_FIXED_ROI_Z * gad_std_vec)[valid],
                    (gad_vec + PROFILE_FIXED_ROI_Z * gad_std_vec)[valid],
                    color=row_color,
                    alpha=PROFILE_FIXED_ROI_FILL_ALPHA,
                )

        ax_vis.plot(
            vis_vec,
            vis_depth_mm,
            linewidth=2.5,
            color=row_color,
            marker="o",
            markersize=3.5,
            markevery=max(1, len(vis_depth_mm) // 8),
        )
        ax_gad.plot(
            gad_vec,
            gad_depth_mm,
            linewidth=2.5,
            color=row_color,
            marker="s",
            markersize=3.5,
            markevery=max(1, len(gad_depth_mm) // 8),
        )

        ax_vis.set_title(f"{panel_name}: {VIS_LABEL}\nTarget {target_t:.2f} min | Actual {vis_times[vis_idx]:.2f} min", color=row_color)
        ax_gad.set_title(f"{panel_name}: {GAD_LABEL}\nTarget {target_t:.2f} min | Actual {gad_times[gad_idx]:.2f} min", color=row_color)

        ax_vis.set_xlabel(x_label)
        ax_gad.set_xlabel(x_label)
        ax_vis.set_ylabel("Depth (mm)")
        ax_gad.set_ylabel("Depth (mm)")

        ax_vis.grid(True)
        ax_gad.grid(True)

        ax_vis.set_xlim(vmin, vmax)
        ax_gad.set_xlim(vmin, vmax)

        if PLOT_DEPTH_ZERO_AT_TOP:
            ax_vis.set_ylim(float(np.max(vis_depth_mm) + y_pad_mm), float(np.min(vis_depth_mm)))
            ax_gad.set_ylim(float(np.max(gad_depth_mm) + y_pad_mm), float(np.min(gad_depth_mm)))
        else:
            ax_vis.set_ylim(float(np.min(vis_depth_mm)), float(np.max(vis_depth_mm) + y_pad_mm))
            ax_gad.set_ylim(float(np.min(gad_depth_mm)), float(np.max(gad_depth_mm) + y_pad_mm))

    fig.suptitle(title_prefix, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, out_path)


# ============================================================
# COMPARISON DRIVER
# ============================================================
MAP_CONFIGS = [
    {
        "key": "per_timepoint_fitted_profiles",
        "subfolder": "per_timepoint",
        "title": "Per-Timepoint Fitted Concentration Map",
        "cbar_label": "Concentration (mg/mL)",
        "line_x_label": "Concentration (mg/mL)",
        "out_stem": "vis_vs_gad_per_timepoint_fitted_concentration",
        "nonnegative": True,
        "supports_fixed_roi_uncertainty": True,
        "uncertainty_prefix": "fitted_profiles",
        "uncertainty_strategy": "combine_components",
    },
    {
        "key": "per_timepoint_flux_magnitude",
        "subfolder": "per_timepoint",
        "title": "Per-Timepoint Diffusive Flux Magnitude Map",
        "cbar_label": r"|J_diff|",
        "line_x_label": r"|J_diff|",
        "out_stem": "vis_vs_gad_per_timepoint_diffusive_flux_magnitude",
        "nonnegative": True,
        "supports_fixed_roi_uncertainty": True,
        "uncertainty_prefix": "per_timepoint_flux_magnitude",
        "uncertainty_strategy": "direct_fixed_roi_or_components",
    },
    {
        "key": "temporally_regularized_fitted_profiles",
        "subfolder": "temporally_regularized",
        "title": "Temporally Regularized Fitted Concentration Map",
        "cbar_label": "Concentration (mg/mL)",
        "line_x_label": "Concentration (mg/mL)",
        "out_stem": "vis_vs_gad_temporally_regularized_fitted_concentration",
        "nonnegative": True,
        "supports_fixed_roi_uncertainty": True,
        "uncertainty_prefix": "temporally_regularized_fitted_profiles",
        "uncertainty_strategy": "combine_components",
    },
    {
        "key": "temporally_regularized_flux_magnitude",
        "subfolder": "temporally_regularized",
        "title": "Temporally Regularized Diffusive Flux Magnitude Map",
        "cbar_label": r"|J_diff|",
        "line_x_label": r"|J_diff|",
        "out_stem": "vis_vs_gad_temporally_regularized_diffusive_flux_magnitude",
        "nonnegative": True,
    },
    {
        "key": "local_effective_diffusivity",
        "subfolder": "secondary",
        "title": "Local Effective Diffusivity Map",
        "cbar_label": r"Effective diffusivity (mm$^2$/s)",
        "line_x_label": r"Effective diffusivity (mm$^2$/s)",
        "out_stem": "vis_vs_gad_local_effective_diffusivity",
        "nonnegative": True,
    },
]


def run_map_comparison(
    vis_map_csv: str,
    gad_map_csv: str,
    vis_times: np.ndarray,
    gad_times: np.ndarray,
    vis_depth_mm: np.ndarray,
    gad_depth_mm: np.ndarray,
    target_times: List[float],
    map_title: str,
    cbar_label: str,
    line_x_label: str,
    out_dir: str,
    out_stem: str,
    nonnegative: bool = False,
    shared_tmin: Optional[float] = None,
    shared_tmax: Optional[float] = None,
    vis_fixed_roi_std_map: Optional[np.ndarray] = None,
    gad_fixed_roi_std_map: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    vis_map = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, vis_map_csv))
    gad_map = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, gad_map_csv))

    if vis_map.shape[0] != len(vis_times):
        raise ValueError(f"VIS map '{vis_map_csv}' has {vis_map.shape[0]} rows but VIS time vector has {len(vis_times)}.")
    if gad_map.shape[0] != len(gad_times):
        raise ValueError(f"GAD map '{gad_map_csv}' has {gad_map.shape[0]} rows but GAD time vector has {len(gad_times)}.")

    out_dir = Path(out_dir)
    ensure_dir(str(out_dir))

    full_map_path = str(out_dir / f"{out_stem}_full_map_side_by_side.png")
    line_path = str(out_dir / f"{out_stem}_early_mid_late_line_comparison.png")
    fixed_roi_line_path = str(out_dir / f"{out_stem}_early_mid_late_line_comparison_fixedROI_95CI.png")

    save_heatmap_pair_figure(
        vis_map=vis_map,
        gad_map=gad_map,
        vis_times=vis_times,
        gad_times=gad_times,
        vis_depth_mm=vis_depth_mm,
        gad_depth_mm=gad_depth_mm,
        title=f"{map_title}: {VIS_LABEL} vs {GAD_LABEL}",
        cbar_label=cbar_label,
        out_path=full_map_path,
        nonnegative=nonnegative,
        target_times=target_times,
        shared_tmin=shared_tmin,
        shared_tmax=shared_tmax,
    )

    save_three_timepoint_comparison(
        vis_map=vis_map,
        gad_map=gad_map,
        vis_times=vis_times,
        gad_times=gad_times,
        vis_depth_mm=vis_depth_mm,
        gad_depth_mm=gad_depth_mm,
        target_times=target_times,
        title_prefix=f"{map_title}: early / mid / late comparison",
        x_label=line_x_label,
        out_path=line_path,
        nonnegative=nonnegative,
    )

    wrote_fixed_roi = False
    if vis_fixed_roi_std_map is not None and gad_fixed_roi_std_map is not None:
        save_three_timepoint_comparison(
            vis_map=vis_map,
            gad_map=gad_map,
            vis_times=vis_times,
            gad_times=gad_times,
            vis_depth_mm=vis_depth_mm,
            gad_depth_mm=gad_depth_mm,
            target_times=target_times,
            title_prefix=f"{map_title}: early / mid / late comparison (fixed ROI 95% CI)",
            x_label=line_x_label,
            out_path=fixed_roi_line_path,
            nonnegative=nonnegative,
            vis_std_map=vis_fixed_roi_std_map,
            gad_std_map=gad_fixed_roi_std_map,
            show_fixed_roi_uncertainty=True,
        )
        wrote_fixed_roi = True

    vis_valid = np.any(np.isfinite(vis_map), axis=1)
    gad_valid = np.any(np.isfinite(gad_map), axis=1)
    vis_indices = [nearest_index(vis_times, t, vis_valid) for t in target_times]
    gad_indices = [nearest_index(gad_times, t, gad_valid) for t in target_times]

    return {
        "vis_map_csv": vis_map_csv,
        "gad_map_csv": gad_map_csv,
        "vis_shape": list(vis_map.shape),
        "gad_shape": list(gad_map.shape),
        "target_times_used_min": [float(x) for x in target_times],
        "vis_indices_used": vis_indices,
        "gad_indices_used": gad_indices,
        "vis_actual_times_min": [float(vis_times[i]) for i in vis_indices],
        "gad_actual_times_min": [float(gad_times[i]) for i in gad_indices],
        "shared_time_axis_min": float(shared_tmin) if shared_tmin is not None else None,
        "shared_time_axis_max": float(shared_tmax) if shared_tmax is not None else None,
        "fixed_roi_uncertainty_requested": bool(vis_fixed_roi_std_map is not None or gad_fixed_roi_std_map is not None),
        "fixed_roi_uncertainty_used": bool(wrote_fixed_roi),
        "outputs_written": [
            str(Path("map_comparisons") / out_dir.name / f"{out_stem}_full_map_side_by_side.png"),
            str(Path("map_comparisons") / out_dir.name / f"{out_stem}_early_mid_late_line_comparison.png"),
        ] + (
            [str(Path("map_comparisons") / out_dir.name / f"{out_stem}_early_mid_late_line_comparison_fixedROI_95CI.png")]
            if wrote_fixed_roi else []
        ),
    }


# ============================================================
# MAIN
# ============================================================
def main():
    root_out = Path(OUT_FOLDER) / "map_comparisons"
    ensure_dir(str(root_out))

    vis_run_rel, vis_selection_mode = resolve_run_folder(RESULTS_SOURCE, VIS_RUN_REL, VIS_FOLDER_HINT, "VIS")
    gad_run_rel, gad_selection_mode = resolve_run_folder(RESULTS_SOURCE, GAD_RUN_REL, GAD_FOLDER_HINT, "GAD")

    vis_paths = build_map_rel_paths(vis_run_rel, VIS_ROI_FOLDER)
    gad_paths = build_map_rel_paths(gad_run_rel, GAD_ROI_FOLDER)

    vis_params_df = load_csv_required_any(RESULTS_SOURCE, vis_paths["fit_parameters_csv"])
    gad_params_df = load_csv_required_any(RESULTS_SOURCE, gad_paths["fit_parameters_csv"])

    vis_time_col = find_time_column(vis_params_df)
    gad_time_col = find_time_column(gad_params_df)

    vis_times = pd.to_numeric(vis_params_df[vis_time_col], errors="coerce").to_numpy(dtype=float)
    gad_times = pd.to_numeric(gad_params_df[gad_time_col], errors="coerce").to_numpy(dtype=float)

    map_key_to_path_vis = {
        "per_timepoint_fitted_profiles": vis_paths["per_timepoint_fitted_profiles_csv"],
        "per_timepoint_flux_magnitude": vis_paths["per_timepoint_flux_magnitude_csv"],
        "temporally_regularized_fitted_profiles": vis_paths["temporally_regularized_fitted_profiles_csv"],
        "temporally_regularized_flux_magnitude": vis_paths["temporally_regularized_flux_magnitude_csv"],
        "local_effective_diffusivity": vis_paths["local_effective_diffusivity_csv"],
    }
    map_key_to_path_gad = {
        "per_timepoint_fitted_profiles": gad_paths["per_timepoint_fitted_profiles_csv"],
        "per_timepoint_flux_magnitude": gad_paths["per_timepoint_flux_magnitude_csv"],
        "temporally_regularized_fitted_profiles": gad_paths["temporally_regularized_fitted_profiles_csv"],
        "temporally_regularized_flux_magnitude": gad_paths["temporally_regularized_flux_magnitude_csv"],
        "local_effective_diffusivity": gad_paths["local_effective_diffusivity_csv"],
    }

    # Determine common target times once from the per-timepoint fitted concentration maps.
    vis_map_for_time_pick = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, vis_paths["per_timepoint_fitted_profiles_csv"]))
    gad_map_for_time_pick = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, gad_paths["per_timepoint_fitted_profiles_csv"]))

    target_times = choose_target_times(
        vis_times[np.any(np.isfinite(vis_map_for_time_pick), axis=1)],
        gad_times[np.any(np.isfinite(gad_map_for_time_pick), axis=1)]
    )

    shared_tmin = float(SHARED_TIME_AXIS_MIN) if SHARED_TIME_AXIS_MIN is not None else float(min(np.nanmin(vis_times), np.nanmin(gad_times)))
    shared_tmax = float(SHARED_TIME_AXIS_MAX) if SHARED_TIME_AXIS_MAX is not None else float(max(np.nanmax(vis_times), np.nanmax(gad_times)))

    vis_depth_reference = build_depth_mm(vis_map_for_time_pick.shape[1], VIS_DX_MM)
    gad_depth_reference = build_depth_mm(gad_map_for_time_pick.shape[1], GAD_DX_MM)

    audit_map_details = {}
    all_outputs_written = []

    for cfg in MAP_CONFIGS:
        vis_map_csv = map_key_to_path_vis[cfg["key"]]
        gad_map_csv = map_key_to_path_gad[cfg["key"]]

        # Depth vectors are derived per-map in case VIS/GAD depth lengths differ by map type.
        vis_map_tmp = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, vis_map_csv))
        gad_map_tmp = to_numeric_2d(load_csv_required_any(RESULTS_SOURCE, gad_map_csv))
        vis_depth_mm = build_depth_mm(vis_map_tmp.shape[1], VIS_DX_MM)
        gad_depth_mm = build_depth_mm(gad_map_tmp.shape[1], GAD_DX_MM)

        vis_fixed_roi_std_map = None
        gad_fixed_roi_std_map = None
        uncertainty_status = {"supported_by_script": bool(cfg.get("supports_fixed_roi_uncertainty", False))}
        if cfg.get("supports_fixed_roi_uncertainty", False):
            unc_prefix = cfg.get("uncertainty_prefix", "fitted_profiles")
            unc_strategy = cfg.get("uncertainty_strategy", "combine_components")
            uncertainty_status.update({
                "prefix": unc_prefix,
                "strategy": unc_strategy,
            })

            if unc_strategy == "combine_components":
                vis_fit_key = f"{unc_prefix}_fit_std_csv"
                vis_hu_key = f"{unc_prefix}_hu_noise_std_csv"
                vis_cal_key = f"{unc_prefix}_calibration_std_csv"
                gad_fit_key = f"{unc_prefix}_fit_std_csv"
                gad_hu_key = f"{unc_prefix}_hu_noise_std_csv"
                gad_cal_key = f"{unc_prefix}_calibration_std_csv"

                vis_fit_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_fit_key])
                vis_hu_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_hu_key])
                vis_calib_std_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_cal_key])
                gad_fit_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_fit_key])
                gad_hu_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_hu_key])
                gad_calib_std_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_cal_key])

                vis_fixed_roi_std_map = combine_available_std_maps(
                    to_numeric_2d(vis_fit_std_df) if vis_fit_std_df is not None else None,
                    to_numeric_2d(vis_hu_std_df) if vis_hu_std_df is not None else None,
                    to_numeric_2d(vis_calib_std_df) if vis_calib_std_df is not None else None,
                )
                gad_fixed_roi_std_map = combine_available_std_maps(
                    to_numeric_2d(gad_fit_std_df) if gad_fit_std_df is not None else None,
                    to_numeric_2d(gad_hu_std_df) if gad_hu_std_df is not None else None,
                    to_numeric_2d(gad_calib_std_df) if gad_calib_std_df is not None else None,
                )

                uncertainty_status.update({
                    "vis_fit_std_found": vis_fit_std_df is not None,
                    "vis_hu_noise_std_found": vis_hu_std_df is not None,
                    "vis_calibration_std_found": vis_calib_std_df is not None,
                    "gad_fit_std_found": gad_fit_std_df is not None,
                    "gad_hu_noise_std_found": gad_hu_std_df is not None,
                    "gad_calibration_std_found": gad_calib_std_df is not None,
                })

            elif unc_strategy == "direct_fixed_roi_or_components":
                vis_fixed_key = f"{unc_prefix}_fixed_roi_std_csv"
                gad_fixed_key = f"{unc_prefix}_fixed_roi_std_csv"
                vis_fixed_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_fixed_key])
                gad_fixed_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_fixed_key])

                if vis_fixed_df is not None:
                    vis_fixed_roi_std_map = to_numeric_2d(vis_fixed_df)
                if gad_fixed_df is not None:
                    gad_fixed_roi_std_map = to_numeric_2d(gad_fixed_df)

                uncertainty_status.update({
                    "vis_fixed_roi_std_found": vis_fixed_df is not None,
                    "gad_fixed_roi_std_found": gad_fixed_df is not None,
                })

                if vis_fixed_roi_std_map is None or gad_fixed_roi_std_map is None:
                    vis_model_key = f"{unc_prefix}_model_fit_std_csv"
                    vis_hu_key = f"{unc_prefix}_hu_noise_std_csv"
                    vis_cal_key = f"{unc_prefix}_calibration_std_csv"
                    gad_model_key = f"{unc_prefix}_model_fit_std_csv"
                    gad_hu_key = f"{unc_prefix}_hu_noise_std_csv"
                    gad_cal_key = f"{unc_prefix}_calibration_std_csv"

                    vis_model_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_model_key])
                    vis_hu_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_hu_key])
                    vis_cal_df = load_csv_optional_any(RESULTS_SOURCE, vis_paths[vis_cal_key])
                    gad_model_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_model_key])
                    gad_hu_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_hu_key])
                    gad_cal_df = load_csv_optional_any(RESULTS_SOURCE, gad_paths[gad_cal_key])

                    if vis_fixed_roi_std_map is None:
                        vis_fixed_roi_std_map = combine_available_std_maps(
                            to_numeric_2d(vis_model_df) if vis_model_df is not None else None,
                            to_numeric_2d(vis_hu_df) if vis_hu_df is not None else None,
                            to_numeric_2d(vis_cal_df) if vis_cal_df is not None else None,
                        )
                    if gad_fixed_roi_std_map is None:
                        gad_fixed_roi_std_map = combine_available_std_maps(
                            to_numeric_2d(gad_model_df) if gad_model_df is not None else None,
                            to_numeric_2d(gad_hu_df) if gad_hu_df is not None else None,
                            to_numeric_2d(gad_cal_df) if gad_cal_df is not None else None,
                        )

                    uncertainty_status.update({
                        "vis_model_fit_std_found": vis_model_df is not None,
                        "vis_hu_noise_std_found": vis_hu_df is not None,
                        "vis_calibration_std_found": vis_cal_df is not None,
                        "gad_model_fit_std_found": gad_model_df is not None,
                        "gad_hu_noise_std_found": gad_hu_df is not None,
                        "gad_calibration_std_found": gad_cal_df is not None,
                    })
            else:
                uncertainty_status["reason"] = f"Unsupported uncertainty strategy: {unc_strategy}"
        else:
            uncertainty_status["reason"] = "No matching fixed-ROI uncertainty CSVs are defined for this map type in the current workflow."

        subfolder_out = root_out / cfg["subfolder"]
        details = run_map_comparison(
            vis_map_csv=vis_map_csv,
            gad_map_csv=gad_map_csv,
            vis_times=vis_times,
            gad_times=gad_times,
            vis_depth_mm=vis_depth_mm,
            gad_depth_mm=gad_depth_mm,
            target_times=target_times,
            map_title=cfg["title"],
            cbar_label=cfg["cbar_label"],
            line_x_label=cfg["line_x_label"],
            out_dir=str(subfolder_out),
            out_stem=cfg["out_stem"],
            nonnegative=cfg["nonnegative"],
            shared_tmin=shared_tmin,
            shared_tmax=shared_tmax,
            vis_fixed_roi_std_map=vis_fixed_roi_std_map,
            gad_fixed_roi_std_map=gad_fixed_roi_std_map,
        )
        details["fixed_roi_uncertainty_status"] = uncertainty_status
        audit_map_details[cfg["key"]] = details
        all_outputs_written.extend(details["outputs_written"])

    audit_payload = {
        "results_source": RESULTS_SOURCE,
        "vis_selection_mode": vis_selection_mode,
        "gad_selection_mode": gad_selection_mode,
        "vis_run_rel": vis_run_rel,
        "gad_run_rel": gad_run_rel,
        "vis_roi_folder": VIS_ROI_FOLDER,
        "gad_roi_folder": GAD_ROI_FOLDER,
        "vis_paths_used": vis_paths,
        "gad_paths_used": gad_paths,
        "vis_time_column": vis_time_col,
        "gad_time_column": gad_time_col,
        "manual_target_times_min": MANUAL_TARGET_TIMES_MIN,
        "target_times_used_min": [float(x) for x in target_times],
        "shared_time_axis_min": float(shared_tmin),
        "shared_time_axis_max": float(shared_tmax),
        "vis_dx_mm": float(VIS_DX_MM),
        "gad_dx_mm": float(GAD_DX_MM),
        "profile_y_pad_mm": float(PROFILE_Y_PAD_MM),
        "profile_fixed_roi_fill_alpha": float(PROFILE_FIXED_ROI_FILL_ALPHA),
        "profile_fixed_roi_z": float(PROFILE_FIXED_ROI_Z),
        "plot_depth_zero_at_top": bool(PLOT_DEPTH_ZERO_AT_TOP),
        "fixed_roi_uncertainty_note": "Fixed-ROI uncertainty overlays are used when matching CSVs exist. Per-timepoint fitted concentration uses combined model-fit + HU-noise + calibration profile uncertainty, and per-timepoint diffusive flux magnitude can now use direct saved fixed-ROI flux uncertainty CSVs or fall back to component combination if needed.",
        "robust_color_limits": bool(ROBUST_COLOR_LIMITS),
        "lower_percentile": float(LOWER_PERCENTILE),
        "upper_percentile": float(UPPER_PERCENTILE),
        "map_details": audit_map_details,
        "outputs_written": all_outputs_written,
    }
    save_audit_report(OUT_FOLDER, audit_payload)

    print("Saved audit files to:", str(root_out / "audit"))
    print("Saved map comparison outputs under:", str(root_out))
    print("Done.")


if __name__ == "__main__":
    main()
