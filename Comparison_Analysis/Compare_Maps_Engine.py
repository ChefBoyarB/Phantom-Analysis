import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURE_TITLE_DEFAULT = "Map Comparison"
TIME_UNIT_LABEL_DEFAULT = "min"

PLOT_DEPTH_ZERO_AT_TOP_DEFAULT = True
ROBUST_COLOR_LIMITS_DEFAULT = True
LOWER_PERCENTILE_DEFAULT = 2.0
UPPER_PERCENTILE_DEFAULT = 98.0
PROFILE_Y_PAD_MM_DEFAULT = 0.15
PROFILE_FIXED_ROI_FILL_ALPHA_DEFAULT = 0.14
PROFILE_FIXED_ROI_Z_DEFAULT = 1.96
FIGURE_LEGEND_Y = 0.975
FIGURE_SUPTITLE_Y = 1.01
FIGURE_TOP_RECT = 0.84
LEGEND_BORDER_AXES_PAD = 1.2
LOCAL_D_MAP_KEY = "local_effective_diffusivity"
LOCAL_D_RMSE_PATH_KEY = "local_effective_fit_rmse_csv"
LOCAL_D_BOUND_LO = 1e-6
LOCAL_D_BOUND_HI = 0.01
LOCAL_D_BOUND_RTOL = 1e-6
LOCAL_D_BOUND_ATOL = 1e-10
LOCAL_D_RMSE_IQR_MULT = 1.5
LOCAL_D_DEFAULT_WINDOW_MIN = 5.0
LOCAL_D_RATIO_EPS = 1e-12

CENTER_LINEWIDTH = 2.3
SAMPLE_LINEWIDTH = 1.1
SAMPLE_LINE_ALPHA = 0.20

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

DEFAULT_REPORT_WINDOWS = [
    {
        "name": "post_5min",
        "min_time_min": 5.0,
        "max_time_min": None,
    }
]

DEFAULT_STATS_METRICS = [
    "depth_mean",
    "depth_integral_x_mm",
    "depth_max",
    "surface_value",
    "penetration_depth_10pct_peak_mm",
]

TIMEPOINT_COLORS = ["#9652F0", "#00FFFF", "#FF0000"]

LINE_MODE_SPECS = [
    ("none", "target_profiles", "target_time_profiles.png"),
    ("fixed_roi_95ci", "target_profiles_fixedROI_95CI", "target_time_profiles_fixedROI_95CI.png"),
    ("combined_95ci", "target_profiles_combined_95CI", "target_time_profiles_combined_95CI.png"),
]

MAP_METRIC_LABELS = {
    "depth_mean": "Depth Mean",
    "depth_median": "Depth Median",
    "depth_max": "Depth Max",
    "surface_value": "Surface Value",
    "depth_integral_x_mm": "Depth Integral (value x mm)",
    "penetration_depth_10pct_peak_mm": "Penetration Depth at 10% Peak (mm)",
    "fixed_roi_std_depth_mean": "Fixed-ROI SD, Depth Mean",
    "fixed_roi_ci_width_depth_mean": "Fixed-ROI 95% CI Width, Depth Mean",
    "combined_std_depth_mean": "Combined SD, Depth Mean",
    "combined_ci_width_depth_mean": "Combined 95% CI Width, Depth Mean",
}

MAP_CONFIGS = [
    {
        "key": "per_timepoint_fitted_profiles",
        "map_path_key": "per_timepoint_fitted_profiles_csv",
        "subfolder": "per_timepoint",
        "title": "Per-Timepoint Fitted Concentration Map",
        "cbar_label": "Concentration (mg/mL)",
        "line_value_label": "Concentration (mg/mL)",
        "out_stem": "per_timepoint_fitted_concentration",
        "nonnegative": True,
        "fixed_roi_uncertainty": {
            "std_key": "fitted_profiles_fixed_roi_std_csv",
            "ci_low_key": "fitted_profiles_fixed_roi_ci_low_csv",
            "ci_high_key": "fitted_profiles_fixed_roi_ci_high_csv",
            "component_keys": [
                "fitted_profiles_fit_std_csv",
                "fitted_profiles_hu_noise_std_csv",
                "fitted_profiles_calibration_std_csv",
            ],
        },
        "combined_uncertainty": {
            "std_key": "fitted_profiles_combined_std_csv",
            "ci_low_key": "fitted_profiles_combined_ci_low_csv",
            "ci_high_key": "fitted_profiles_combined_ci_high_csv",
            "component_keys": [
                "fitted_profiles_fit_std_csv",
                "fitted_profiles_hu_noise_std_csv",
                "fitted_profiles_roi_sensitivity_std_csv",
                "fitted_profiles_calibration_std_csv",
            ],
        },
    },
    {
        "key": "per_timepoint_flux_magnitude",
        "map_path_key": "per_timepoint_flux_magnitude_csv",
        "subfolder": "per_timepoint",
        "title": "Per-Timepoint Diffusive Flux Magnitude Map",
        "cbar_label": r"|J_diff|",
        "line_value_label": r"|J_diff|",
        "out_stem": "per_timepoint_diffusive_flux_magnitude",
        "nonnegative": True,
        "fixed_roi_uncertainty": {
            "std_key": "per_timepoint_flux_magnitude_fixed_roi_std_csv",
            "ci_low_key": "per_timepoint_flux_magnitude_fixed_roi_ci_low_csv",
            "ci_high_key": "per_timepoint_flux_magnitude_fixed_roi_ci_high_csv",
            "component_keys": [
                "per_timepoint_flux_magnitude_model_fit_std_csv",
                "per_timepoint_flux_magnitude_hu_noise_std_csv",
                "per_timepoint_flux_magnitude_calibration_std_csv",
            ],
        },
        "combined_uncertainty": {
            "std_key": "per_timepoint_flux_magnitude_combined_std_csv",
            "ci_low_key": "per_timepoint_flux_magnitude_combined_ci_low_csv",
            "ci_high_key": "per_timepoint_flux_magnitude_combined_ci_high_csv",
            "component_keys": [
                "per_timepoint_flux_magnitude_model_fit_std_csv",
                "per_timepoint_flux_magnitude_hu_noise_std_csv",
                "per_timepoint_flux_magnitude_roi_sensitivity_std_csv",
                "per_timepoint_flux_magnitude_calibration_std_csv",
            ],
        },
    },
    {
        "key": "temporally_regularized_fitted_profiles",
        "map_path_key": "temporally_regularized_fitted_profiles_csv",
        "subfolder": "temporally_regularized",
        "title": "Temporally Regularized Fitted Concentration Map",
        "cbar_label": "Concentration (mg/mL)",
        "line_value_label": "Concentration (mg/mL)",
        "out_stem": "temporally_regularized_fitted_concentration",
        "nonnegative": True,
        "fixed_roi_uncertainty": {
            "std_key": "temporally_regularized_fitted_profiles_fixed_roi_std_csv",
            "ci_low_key": "temporally_regularized_fitted_profiles_fixed_roi_ci_low_csv",
            "ci_high_key": "temporally_regularized_fitted_profiles_fixed_roi_ci_high_csv",
            "component_keys": [
                "temporally_regularized_fitted_profiles_fit_std_csv",
                "temporally_regularized_fitted_profiles_hu_noise_std_csv",
                "temporally_regularized_fitted_profiles_calibration_std_csv",
            ],
        },
        "combined_uncertainty": {
            "std_key": "temporally_regularized_fitted_profiles_combined_std_csv",
            "ci_low_key": "temporally_regularized_fitted_profiles_combined_ci_low_csv",
            "ci_high_key": "temporally_regularized_fitted_profiles_combined_ci_high_csv",
            "component_keys": [
                "temporally_regularized_fitted_profiles_fit_std_csv",
                "temporally_regularized_fitted_profiles_hu_noise_std_csv",
                "temporally_regularized_fitted_profiles_roi_sensitivity_std_csv",
                "temporally_regularized_fitted_profiles_calibration_std_csv",
            ],
        },
    },
    {
        "key": "temporally_regularized_flux_magnitude",
        "map_path_key": "temporally_regularized_flux_magnitude_csv",
        "subfolder": "temporally_regularized",
        "title": "Temporally Regularized Diffusive Flux Magnitude Map",
        "cbar_label": r"|J_diff|",
        "line_value_label": r"|J_diff|",
        "out_stem": "temporally_regularized_diffusive_flux_magnitude",
        "nonnegative": True,
    },
    {
        "key": "local_effective_diffusivity",
        "map_path_key": "local_effective_diffusivity_csv",
        "subfolder": "secondary",
        "title": "Local Effective Diffusivity",
        "cbar_label": r"Local effective diffusivity, $D_{eff}$ (mm$^2$/s)",
        "line_value_label": r"Local effective diffusivity, $D_{eff}$ (mm$^2$/s)",
        "out_stem": "local_effective_diffusivity",
        "nonnegative": True,
    },
]

_MISSING_OPTIONAL_PATHS_LOGGED: set[str] = set()


@dataclass
class MapData:
    key: str
    label: str
    values: np.ndarray
    valid_rows: np.ndarray
    fixed_roi_std: Optional[np.ndarray]
    fixed_roi_ci_low: Optional[np.ndarray]
    fixed_roi_ci_high: Optional[np.ndarray]
    combined_std: Optional[np.ndarray]
    combined_ci_low: Optional[np.ndarray]
    combined_ci_high: Optional[np.ndarray]
    uncertainty_audit: Dict[str, Any]


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
    time_column: str
    analysis_config_path: Optional[str]
    run_metadata_path: Optional[str]
    paths_used: Dict[str, str]
    times: np.ndarray
    depth: np.ndarray
    maps: Dict[str, MapData]
    local_effective_fit_rmse_map: Optional[np.ndarray]
    audit_info: Dict[str, Any]


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


def _read_json_any(base: str, relative_path: str) -> Dict[str, Any]:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return json.loads(zf.read(rel).decode("utf-8-sig"))
    with open(Path(base) / rel, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


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
        if relative_path not in _MISSING_OPTIONAL_PATHS_LOGGED:
            print(f"Optional file not found, skipping: {relative_path}")
            _MISSING_OPTIONAL_PATHS_LOGGED.add(str(relative_path))
        return None
    return _read_csv_any(base, relative_path)


def load_json_optional_any(base: str, relative_path: Optional[str]) -> Optional[Dict[str, Any]]:
    if relative_path in (None, ""):
        return None
    if not _path_exists_any(base, relative_path):
        return None
    return _read_json_any(base, relative_path)


def load_csv_required_path(path_value: str) -> pd.DataFrame:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def load_csv_optional_path(path_value: Optional[str]) -> Optional[pd.DataFrame]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.exists():
        path_text = str(path)
        if path_text not in _MISSING_OPTIONAL_PATHS_LOGGED:
            print(f"Optional file not found, skipping: {path}")
            _MISSING_OPTIONAL_PATHS_LOGGED.add(path_text)
        return None
    return pd.read_csv(path)


def load_json_required_path(path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_json_optional_path(path_value: Optional[str]) -> Optional[Dict[str, Any]]:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


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


def choose_target_times(
    samples: Sequence[SampleData],
    target_map_key: str,
    manual_target_times: Optional[Sequence[float]],
) -> List[float]:
    if manual_target_times is not None:
        if len(manual_target_times) == 0:
            raise ValueError("target_times_min cannot be empty.")
        return [float(x) for x in manual_target_times]

    min_times = []
    max_times = []
    for sample in samples:
        map_data = sample.maps[target_map_key]
        valid_times = sample.times[map_data.valid_rows & np.isfinite(sample.times)]
        if valid_times.size == 0:
            raise ValueError(f"Sample {sample.sample_id} has no valid times for map '{target_map_key}'.")
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


def interp_series(x: np.ndarray, y: np.ndarray, target_x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    target_x = np.asarray(target_x, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(valid) < 2:
        return np.full(target_x.shape, np.nan, dtype=float)
    x_valid = x[valid]
    y_valid = y[valid]
    order = np.argsort(x_valid)
    x_valid = x_valid[order]
    y_valid = y_valid[order]
    x_unique, unique_idx = np.unique(x_valid, return_index=True)
    y_unique = y_valid[unique_idx]
    if x_unique.size < 2:
        return np.full(target_x.shape, np.nan, dtype=float)
    return np.interp(target_x, x_unique, y_unique, left=np.nan, right=np.nan)


def interp_profile(depth: np.ndarray, values: np.ndarray, target_depth: np.ndarray) -> np.ndarray:
    return interp_series(depth, values, target_depth)


def regrid_map(
    times: np.ndarray,
    depth: np.ndarray,
    values: np.ndarray,
    target_times: np.ndarray,
    target_depth: np.ndarray,
) -> np.ndarray:
    depth_interp_rows = np.vstack([interp_profile(depth, row, target_depth) for row in np.asarray(values, dtype=float)])
    out = np.full((len(target_times), len(target_depth)), np.nan, dtype=float)
    for depth_idx in range(len(target_depth)):
        out[:, depth_idx] = interp_series(times, depth_interp_rows[:, depth_idx], target_times)
    return out


def nanmean_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=0)
    totals = np.sum(np.where(finite, arr, 0.0), axis=0)
    out = np.full(arr.shape[1:], np.nan, dtype=float)
    valid = counts > 0
    out[valid] = totals[valid] / counts[valid]
    return out


def combine_uncertainty_terms(*arrays: Optional[np.ndarray]) -> Optional[np.ndarray]:
    parts = []
    for arr in arrays:
        if arr is not None:
            parts.append(np.nan_to_num(np.asarray(arr, dtype=float), nan=0.0, posinf=0.0, neginf=0.0))
    if not parts:
        return None
    return np.sqrt(np.sum([part ** 2 for part in parts], axis=0))


def approx_ci_from_std(
    center: Optional[np.ndarray],
    std_values: Optional[np.ndarray],
    z: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if center is None or std_values is None:
        return None, None
    center = np.asarray(center, dtype=float)
    std_values = np.asarray(std_values, dtype=float)
    return center - z * std_values, center + z * std_values


def arrays_match(left: Optional[np.ndarray], right: Optional[np.ndarray], atol: float = 1e-10, rtol: float = 1e-6) -> Optional[bool]:
    if left is None or right is None:
        return None
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    finite_mask = np.isfinite(left_arr) & np.isfinite(right_arr)
    if not np.any(finite_mask):
        return None
    return bool(np.allclose(left_arr[finite_mask], right_arr[finite_mask], atol=atol, rtol=rtol))


def compute_axis_limits(*arrays: np.ndarray) -> Tuple[Tuple[float, float], float]:
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


def get_color_limits(
    arrays: Sequence[np.ndarray],
    nonnegative: bool,
    robust: bool,
    lower_pct: float,
    upper_pct: float,
) -> Tuple[float, float]:
    values = []
    for arr in arrays:
        finite = np.asarray(arr, dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            values.append(finite)
    if not values:
        return 0.0, 1.0
    merged = np.concatenate(values)
    if nonnegative:
        merged = merged[merged >= 0]
    if merged.size == 0:
        return 0.0, 1.0
    if robust:
        vmin = 0.0 if nonnegative else float(np.percentile(merged, lower_pct))
        vmax = float(np.percentile(merged, upper_pct))
    else:
        vmin = 0.0 if nonnegative else float(np.min(merged))
        vmax = float(np.max(merged))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def safe_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def safe_median(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def safe_max(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def safe_first(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or not np.isfinite(arr[0]):
        return float("nan")
    return float(arr[0])


def safe_integral(depth: np.ndarray, values: np.ndarray) -> float:
    valid = np.isfinite(depth) & np.isfinite(values)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values[valid], depth[valid]))
    return float(np.trapz(values[valid], depth[valid]))


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


def finite_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "sem": float("nan"),
            "median": float("nan"),
            "iqr": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    std_value = float(np.std(arr, ddof=1)) if arr.size > 1 else float("nan")
    sem_value = float(std_value / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
    q75, q25 = np.percentile(arr, [75, 25]) if arr.size else (float("nan"), float("nan"))
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": std_value,
        "sem": sem_value,
        "median": float(np.median(arr)),
        "iqr": float(q75 - q25),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


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


def build_map_rel_paths(run_rel: str, roi_folder: str) -> Dict[str, str]:
    run_rel = _normalize_rel_path(run_rel)
    roi_folder = str(roi_folder).strip("/").strip("\\")
    base = f"{run_rel}/{roi_folder}"
    return {
        "run_metadata_json": f"{run_rel}/run_metadata.json",
        "fit_parameters_csv": f"{base}/CSVs_Summaries/fit_parameters_vs_time.csv",
        "per_timepoint_fitted_profiles_csv": f"{base}/CSVs_Profiles/fitted_profiles_depth_vs_time.csv",
        "per_timepoint_flux_magnitude_csv": f"{base}/CSVs_Diffusion/diffusive_flux_magnitude_map.csv",
        "temporally_regularized_fitted_profiles_csv": f"{base}/CSVs_Profiles/temporally_regularized_fitted_profiles_depth_vs_time.csv",
        "temporally_regularized_flux_magnitude_csv": f"{base}/CSVs_Diffusion/temporally_regularized_diffusive_flux_magnitude_map.csv",
        "local_effective_diffusivity_csv": f"{base}/CSVs_Diffusion/local_effective_diffusivity_map.csv",
        "local_effective_fit_rmse_csv": f"{base}/CSVs_Diffusion/local_effective_fit_rmse_map.csv",
        "fitted_profiles_fit_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_std_depth_vs_time.csv",
        "fitted_profiles_hu_noise_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_hu_noise_std_depth_vs_time.csv",
        "fitted_profiles_roi_sensitivity_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_roi_sensitivity_std_depth_vs_time.csv",
        "fitted_profiles_calibration_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_calibration_std_depth_vs_time.csv",
        "fitted_profiles_fixed_roi_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_fixed_roi_std_depth_vs_time.csv",
        "fitted_profiles_fixed_roi_ci_low_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_fixed_roi_ci_low_depth_vs_time.csv",
        "fitted_profiles_fixed_roi_ci_high_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_fixed_roi_ci_high_depth_vs_time.csv",
        "fitted_profiles_combined_std_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_combined_std_depth_vs_time.csv",
        "fitted_profiles_combined_ci_low_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_combined_ci_low_depth_vs_time.csv",
        "fitted_profiles_combined_ci_high_csv": f"{base}/CSVs_Uncertainty/fitted_profiles_combined_ci_high_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_fit_std_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_hu_noise_std_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_hu_noise_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_roi_sensitivity_std_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_roi_sensitivity_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_calibration_std_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_calibration_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_fixed_roi_std_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_fixed_roi_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_fixed_roi_ci_low_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_fixed_roi_ci_low_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_fixed_roi_ci_high_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_fixed_roi_ci_high_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_combined_std_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_combined_std_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_combined_ci_low_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_combined_ci_low_depth_vs_time.csv",
        "temporally_regularized_fitted_profiles_combined_ci_high_csv": f"{base}/CSVs_Uncertainty/temporally_regularized_fitted_profiles_combined_ci_high_depth_vs_time.csv",
        "per_timepoint_flux_magnitude_model_fit_std_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_model_fit_std_map.csv",
        "per_timepoint_flux_magnitude_hu_noise_std_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_hu_noise_std_map.csv",
        "per_timepoint_flux_magnitude_roi_sensitivity_std_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_roi_sensitivity_std_map.csv",
        "per_timepoint_flux_magnitude_calibration_std_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_calibration_std_map.csv",
        "per_timepoint_flux_magnitude_fixed_roi_std_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_fixed_roi_std_map.csv",
        "per_timepoint_flux_magnitude_fixed_roi_ci_low_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_fixed_roi_ci_low_map.csv",
        "per_timepoint_flux_magnitude_fixed_roi_ci_high_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_fixed_roi_ci_high_map.csv",
        "per_timepoint_flux_magnitude_combined_std_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_combined_std_map.csv",
        "per_timepoint_flux_magnitude_combined_ci_low_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_combined_ci_low_map.csv",
        "per_timepoint_flux_magnitude_combined_ci_high_csv": f"{base}/CSVs_Uncertainty/diffusive_flux_magnitude_combined_ci_high_map.csv",
    }


def build_map_abs_paths(run_path: str, roi_folder: str) -> Dict[str, str]:
    run_dir = Path(run_path)
    roi_dir = run_dir / str(roi_folder)
    return {
        "run_metadata_json": str(run_dir / "run_metadata.json"),
        "fit_parameters_csv": str(roi_dir / "CSVs_Summaries" / "fit_parameters_vs_time.csv"),
        "per_timepoint_fitted_profiles_csv": str(roi_dir / "CSVs_Profiles" / "fitted_profiles_depth_vs_time.csv"),
        "per_timepoint_flux_magnitude_csv": str(roi_dir / "CSVs_Diffusion" / "diffusive_flux_magnitude_map.csv"),
        "temporally_regularized_fitted_profiles_csv": str(roi_dir / "CSVs_Profiles" / "temporally_regularized_fitted_profiles_depth_vs_time.csv"),
        "temporally_regularized_flux_magnitude_csv": str(roi_dir / "CSVs_Diffusion" / "temporally_regularized_diffusive_flux_magnitude_map.csv"),
        "local_effective_diffusivity_csv": str(roi_dir / "CSVs_Diffusion" / "local_effective_diffusivity_map.csv"),
        "local_effective_fit_rmse_csv": str(roi_dir / "CSVs_Diffusion" / "local_effective_fit_rmse_map.csv"),
        "fitted_profiles_fit_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_std_depth_vs_time.csv"),
        "fitted_profiles_hu_noise_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_hu_noise_std_depth_vs_time.csv"),
        "fitted_profiles_roi_sensitivity_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_roi_sensitivity_std_depth_vs_time.csv"),
        "fitted_profiles_calibration_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_calibration_std_depth_vs_time.csv"),
        "fitted_profiles_fixed_roi_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_fixed_roi_std_depth_vs_time.csv"),
        "fitted_profiles_fixed_roi_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_fixed_roi_ci_low_depth_vs_time.csv"),
        "fitted_profiles_fixed_roi_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_fixed_roi_ci_high_depth_vs_time.csv"),
        "fitted_profiles_combined_std_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_combined_std_depth_vs_time.csv"),
        "fitted_profiles_combined_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_combined_ci_low_depth_vs_time.csv"),
        "fitted_profiles_combined_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "fitted_profiles_combined_ci_high_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_fit_std_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_std_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_hu_noise_std_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_hu_noise_std_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_roi_sensitivity_std_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_roi_sensitivity_std_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_calibration_std_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_calibration_std_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_fixed_roi_std_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_fixed_roi_std_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_fixed_roi_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_fixed_roi_ci_low_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_fixed_roi_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_fixed_roi_ci_high_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_combined_std_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_combined_std_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_combined_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_combined_ci_low_depth_vs_time.csv"),
        "temporally_regularized_fitted_profiles_combined_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "temporally_regularized_fitted_profiles_combined_ci_high_depth_vs_time.csv"),
        "per_timepoint_flux_magnitude_model_fit_std_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_model_fit_std_map.csv"),
        "per_timepoint_flux_magnitude_hu_noise_std_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_hu_noise_std_map.csv"),
        "per_timepoint_flux_magnitude_roi_sensitivity_std_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_roi_sensitivity_std_map.csv"),
        "per_timepoint_flux_magnitude_calibration_std_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_calibration_std_map.csv"),
        "per_timepoint_flux_magnitude_fixed_roi_std_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_fixed_roi_std_map.csv"),
        "per_timepoint_flux_magnitude_fixed_roi_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_fixed_roi_ci_low_map.csv"),
        "per_timepoint_flux_magnitude_fixed_roi_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_fixed_roi_ci_high_map.csv"),
        "per_timepoint_flux_magnitude_combined_std_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_combined_std_map.csv"),
        "per_timepoint_flux_magnitude_combined_ci_low_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_combined_ci_low_map.csv"),
        "per_timepoint_flux_magnitude_combined_ci_high_csv": str(roi_dir / "CSVs_Uncertainty" / "diffusive_flux_magnitude_combined_ci_high_map.csv"),
    }


def load_numeric_array_optional(
    loader_optional,
    paths_used: Dict[str, str],
    path_key: Optional[str],
    expected_shape: Tuple[int, int],
) -> Optional[np.ndarray]:
    if path_key in (None, ""):
        return None
    path_value = paths_used.get(path_key)
    if path_value in (None, ""):
        return None
    df = loader_optional(path_value)
    if df is None:
        return None
    arr = to_numeric_2d(df)
    if arr.shape != expected_shape:
        raise ValueError(f"Unexpected shape for {path_key}: got {arr.shape}, expected {expected_shape}.")
    return arr


def resolve_uncertainty_bundle(
    center: np.ndarray,
    loader_optional,
    paths_used: Dict[str, str],
    spec: Optional[Dict[str, Any]],
    z: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    if not spec:
        return None, None, None, {"supported": False}

    std_direct = load_numeric_array_optional(loader_optional, paths_used, spec.get("std_key"), center.shape)
    ci_low_direct = load_numeric_array_optional(loader_optional, paths_used, spec.get("ci_low_key"), center.shape)
    ci_high_direct = load_numeric_array_optional(loader_optional, paths_used, spec.get("ci_high_key"), center.shape)

    component_arrays = []
    component_presence: Dict[str, bool] = {}
    for key in spec.get("component_keys", []):
        arr = load_numeric_array_optional(loader_optional, paths_used, key, center.shape)
        component_arrays.append(arr)
        component_presence[key] = arr is not None
    std_from_components = combine_uncertainty_terms(*component_arrays)

    std_final = std_direct if std_direct is not None else std_from_components
    ci_low_final = ci_low_direct
    ci_high_final = ci_high_direct
    if (ci_low_final is None or ci_high_final is None) and std_final is not None:
        ci_low_final, ci_high_final = approx_ci_from_std(center, std_final, z=z)

    audit = {
        "supported": True,
        "std_direct_present": std_direct is not None,
        "ci_low_direct_present": ci_low_direct is not None,
        "ci_high_direct_present": ci_high_direct is not None,
        "component_presence": component_presence,
        "std_source": "direct_csv" if std_direct is not None else ("component_combination" if std_from_components is not None else "missing"),
        "ci_source": "direct_csv"
        if ci_low_direct is not None and ci_high_direct is not None
        else ("approximated_from_std" if std_final is not None else "missing"),
        "component_match_direct": arrays_match(std_direct, std_from_components),
    }
    return std_final, ci_low_final, ci_high_final, audit


def first_roi_name_from_run_metadata(run_metadata: Optional[Dict[str, Any]]) -> Optional[str]:
    if not run_metadata:
        return None
    selected_rois = run_metadata.get("selected_rois")
    if isinstance(selected_rois, list) and len(selected_rois) == 1:
        roi_name = selected_rois[0].get("roi_name")
        if roi_name not in (None, ""):
            return str(roi_name)
    return None


def first_roi_name_from_analysis_config(analysis_cfg: Optional[Dict[str, Any]]) -> Optional[str]:
    if not analysis_cfg:
        return None
    settings = analysis_cfg.get("settings", {})
    manual_named_rois = settings.get("manual_named_rois")
    if isinstance(manual_named_rois, list) and len(manual_named_rois) == 1 and len(manual_named_rois[0]) >= 1:
        roi_name = manual_named_rois[0][0]
        if roi_name not in (None, ""):
            return str(roi_name)
    return None


def depth_spacing_from_run_metadata(run_metadata: Optional[Dict[str, Any]]) -> Optional[float]:
    if not run_metadata:
        return None
    frame_geometry = run_metadata.get("frame_geometry", {})
    dx_value = frame_geometry.get("depth_spacing_mm")
    if dx_value in (None, ""):
        return None
    return float(dx_value)


def resolve_analysis_output_folder(analysis_config_doc: Dict[str, Any], analysis_config_path: Path) -> str:
    output_folder = analysis_config_doc.get("settings", {}).get("output_folder")
    if output_folder in (None, ""):
        raise ValueError(f"Analysis config '{analysis_config_path}' is missing settings.output_folder.")
    output_path = Path(str(output_folder))
    if output_path.is_absolute():
        return str(output_path)
    return str((analysis_config_path.parent / output_path).resolve())


def validate_and_resolve_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    if "tracers" not in config or not config["tracers"]:
        raise ValueError("Comparison config must define a non-empty 'tracers' list.")

    resolved: Dict[str, Any] = {}
    resolved["results_source"] = _resolve_path_from_config(config.get("results_source"), config_path)
    resolved["out_folder"] = _resolve_path_from_config(config.get("out_folder"), config_path)
    if resolved["out_folder"] in (None, ""):
        raise ValueError("Comparison config requires a non-empty 'out_folder'.")

    resolved["figure_title"] = config.get("figure_title", FIGURE_TITLE_DEFAULT)
    resolved["time_unit_label"] = config.get("time_unit_label", TIME_UNIT_LABEL_DEFAULT)
    resolved["target_times_min"] = config.get("target_times_min")
    resolved["target_time_map_key"] = config.get("target_time_map_key", "per_timepoint_fitted_profiles")
    resolved["shared_time_axis_min"] = (
        float(config["shared_time_axis_min"]) if config.get("shared_time_axis_min") is not None else None
    )
    resolved["shared_time_axis_max"] = (
        float(config["shared_time_axis_max"]) if config.get("shared_time_axis_max") is not None else None
    )
    resolved["plot_depth_zero_at_top"] = bool(config.get("plot_depth_zero_at_top", PLOT_DEPTH_ZERO_AT_TOP_DEFAULT))
    resolved["robust_color_limits"] = bool(config.get("robust_color_limits", ROBUST_COLOR_LIMITS_DEFAULT))
    resolved["lower_percentile"] = float(config.get("lower_percentile", LOWER_PERCENTILE_DEFAULT))
    resolved["upper_percentile"] = float(config.get("upper_percentile", UPPER_PERCENTILE_DEFAULT))
    raw_map_color_limits = config.get("map_color_limits", {}) or {}
    if not isinstance(raw_map_color_limits, dict):
        raise ValueError("map_color_limits must be a JSON object mapping map keys to {vmin, vmax} objects.")
    raw_map_display_trim_rows = config.get("map_display_trim_rows", {}) or {}
    if not isinstance(raw_map_display_trim_rows, dict):
        raise ValueError("map_display_trim_rows must be a JSON object mapping map keys to {bottom_rows} objects.")
    resolved["profile_y_pad_mm"] = float(config.get("profile_y_pad_mm", PROFILE_Y_PAD_MM_DEFAULT))
    resolved["profile_fixed_roi_fill_alpha"] = float(
        config.get("profile_fixed_roi_fill_alpha", PROFILE_FIXED_ROI_FILL_ALPHA_DEFAULT)
    )
    resolved["profile_fixed_roi_z"] = float(config.get("profile_fixed_roi_z", PROFILE_FIXED_ROI_Z_DEFAULT))
    resolved["stats_alpha"] = float(config.get("stats_alpha", 0.05))
    resolved["stats_metrics"] = list(config.get("stats_metrics", DEFAULT_STATS_METRICS))
    resolved["window_stats_metrics"] = config.get("window_stats_metrics")
    resolved["report_windows"] = config.get("report_windows", DEFAULT_REPORT_WINDOWS)

    map_spec_by_key = {spec["key"]: spec for spec in MAP_CONFIGS}
    requested_map_keys = config.get("map_keys", [spec["key"] for spec in MAP_CONFIGS])
    invalid_map_keys = [key for key in requested_map_keys if key not in map_spec_by_key]
    if invalid_map_keys:
        raise ValueError(f"Unknown map_keys in config: {invalid_map_keys}")
    if resolved["target_time_map_key"] not in requested_map_keys:
        raise ValueError("target_time_map_key must also be included in map_keys.")
    resolved["map_specs"] = [map_spec_by_key[key] for key in requested_map_keys]
    invalid_color_limit_keys = [key for key in raw_map_color_limits.keys() if key not in map_spec_by_key]
    if invalid_color_limit_keys:
        raise ValueError(f"Unknown map_color_limits keys in config: {invalid_color_limit_keys}")
    invalid_trim_keys = [key for key in raw_map_display_trim_rows.keys() if key not in map_spec_by_key]
    if invalid_trim_keys:
        raise ValueError(f"Unknown map_display_trim_rows keys in config: {invalid_trim_keys}")
    resolved_map_color_limits: Dict[str, Dict[str, float]] = {}
    for map_key, limits in raw_map_color_limits.items():
        if limits in (None, {}):
            continue
        if not isinstance(limits, dict):
            raise ValueError(f"map_color_limits['{map_key}'] must be an object with optional vmin/vmax.")
        parsed_limits: Dict[str, float] = {}
        if limits.get("vmin") is not None:
            parsed_limits["vmin"] = float(limits["vmin"])
        if limits.get("vmax") is not None:
            parsed_limits["vmax"] = float(limits["vmax"])
        if "vmin" in parsed_limits and "vmax" in parsed_limits and parsed_limits["vmin"] >= parsed_limits["vmax"]:
            raise ValueError(f"map_color_limits['{map_key}'] must satisfy vmin < vmax.")
        if parsed_limits:
            resolved_map_color_limits[map_key] = parsed_limits
    resolved["map_color_limits"] = resolved_map_color_limits
    resolved_map_display_trim_rows: Dict[str, Dict[str, int]] = {}
    for map_key, trim_spec in raw_map_display_trim_rows.items():
        if trim_spec in (None, {}):
            continue
        if not isinstance(trim_spec, dict):
            raise ValueError(f"map_display_trim_rows['{map_key}'] must be an object with optional bottom_rows.")
        parsed_trim: Dict[str, int] = {}
        if trim_spec.get("bottom_rows") is not None:
            bottom_rows = int(trim_spec["bottom_rows"])
            if bottom_rows < 0:
                raise ValueError(f"map_display_trim_rows['{map_key}'].bottom_rows must be >= 0.")
            parsed_trim["bottom_rows"] = bottom_rows
        if parsed_trim:
            resolved_map_display_trim_rows[map_key] = parsed_trim
    resolved["map_display_trim_rows"] = resolved_map_display_trim_rows

    tracers_resolved = []
    for tracer_index, tracer_cfg in enumerate(config["tracers"]):
        tracer_name = tracer_cfg.get("name")
        if tracer_name in (None, ""):
            raise ValueError("Each tracer entry needs a non-empty 'name'.")
        samples_cfg = tracer_cfg.get("samples", [])
        if not samples_cfg:
            raise ValueError(f"Tracer '{tracer_name}' must define at least one sample.")

        samples_resolved = []
        for sample_cfg in samples_cfg:
            sample_entry = dict(sample_cfg)
            sample_entry["run_path"] = _resolve_path_from_config(sample_cfg.get("run_path"), config_path)
            sample_entry["analysis_config"] = _resolve_path_from_config(sample_cfg.get("analysis_config"), config_path)
            if (
                sample_cfg.get("run_rel") in (None, "")
                and sample_entry["run_path"] in (None, "")
                and sample_entry["analysis_config"] in (None, "")
            ):
                raise ValueError(
                    f"Each sample for tracer '{tracer_name}' must define one of: run_rel, run_path, analysis_config."
                )
            samples_resolved.append(sample_entry)

        tracers_resolved.append(
            {
                "name": tracer_name,
                "label": tracer_cfg.get("label", tracer_name),
                "color": tracer_cfg.get("color", DEFAULT_COLORS[tracer_index % len(DEFAULT_COLORS)]),
                "marker": tracer_cfg.get("marker", MARKERS[tracer_index % len(MARKERS)]),
                "roi_folder": tracer_cfg.get("roi_folder"),
                "dx_mm": tracer_cfg.get("dx_mm"),
                "samples": samples_resolved,
            }
        )
    resolved["tracers"] = tracers_resolved
    return resolved


def load_sample_data(
    tracer_cfg: Dict[str, Any],
    sample_cfg: Dict[str, Any],
    results_source: Optional[str],
    map_specs: Sequence[Dict[str, Any]],
    z_value: float,
) -> SampleData:
    analysis_config_path = sample_cfg.get("analysis_config")
    analysis_config_doc = load_json_required_path(analysis_config_path) if analysis_config_path not in (None, "") else None

    run_path = sample_cfg.get("run_path")
    run_rel = sample_cfg.get("run_rel")
    selection_mode: str
    run_display: str
    run_metadata_doc: Optional[Dict[str, Any]] = None
    run_metadata_path: Optional[str] = None

    if analysis_config_doc is not None:
        resolved_run = resolve_analysis_output_folder(analysis_config_doc, Path(analysis_config_path))
        selection_mode = "analysis_config"
        run_display = resolved_run
        run_metadata_path = str(Path(resolved_run) / "run_metadata.json")
        run_metadata_doc = load_json_optional_path(run_metadata_path)
    elif run_path not in (None, ""):
        resolved_run = str(Path(str(run_path)).resolve())
        selection_mode = "run_path"
        run_display = resolved_run
        run_metadata_path = str(Path(resolved_run) / "run_metadata.json")
        run_metadata_doc = load_json_optional_path(run_metadata_path)
    elif run_rel not in (None, ""):
        if results_source in (None, ""):
            raise ValueError(
                f"Sample '{sample_cfg.get('sample_id', '<unknown>')}' uses run_rel but config has no results_source."
            )
        resolved_run = _normalize_rel_path(run_rel)
        selection_mode = "run_rel"
        run_display = resolved_run
        run_metadata_path = f"{resolved_run}/run_metadata.json"
        run_metadata_doc = load_json_optional_any(results_source, run_metadata_path)
    else:
        raise ValueError(f"Sample in tracer '{tracer_cfg['name']}' must define run_rel, run_path, or analysis_config.")

    roi_folder = (
        sample_cfg.get("roi_folder")
        or tracer_cfg.get("roi_folder")
        or first_roi_name_from_run_metadata(run_metadata_doc)
        or first_roi_name_from_analysis_config(analysis_config_doc)
    )
    if roi_folder in (None, ""):
        raise ValueError(
            f"Could not infer roi_folder for sample '{sample_cfg.get('sample_id', run_display)}'. "
            "Set roi_folder in the comparison config or use an analysis run with exactly one selected ROI."
        )

    dx_value = sample_cfg.get("dx_mm")
    if dx_value in (None, ""):
        dx_value = tracer_cfg.get("dx_mm")
    if dx_value in (None, ""):
        dx_value = depth_spacing_from_run_metadata(run_metadata_doc)
    if dx_value in (None, ""):
        raise ValueError(
            f"Could not infer dx_mm for sample '{sample_cfg.get('sample_id', run_display)}'. "
            "Set dx_mm in the comparison config or provide run metadata with frame_geometry.depth_spacing_mm."
        )

    if selection_mode in {"analysis_config", "run_path"}:
        paths_used = build_map_abs_paths(resolved_run, str(roi_folder))
        loader_required = load_csv_required_path
        loader_optional = load_csv_optional_path
    else:
        paths_used = build_map_rel_paths(resolved_run, str(roi_folder))
        loader_required = lambda relative_path: load_csv_required_any(results_source, relative_path)
        loader_optional = lambda relative_path: load_csv_optional_any(results_source, relative_path)

    fit_params_df = loader_required(paths_used["fit_parameters_csv"])
    time_column = find_time_column(fit_params_df)
    times = pd.to_numeric(fit_params_df[time_column], errors="coerce").to_numpy(dtype=float)

    maps: Dict[str, MapData] = {}
    map_audit: Dict[str, Any] = {}
    for map_spec in map_specs:
        map_path_key = map_spec["map_path_key"]
        values = to_numeric_2d(loader_required(paths_used[map_path_key]))
        if values.shape[0] != len(times):
            raise ValueError(
                f"Map '{map_spec['key']}' for sample '{sample_cfg.get('sample_id', run_display)}' has "
                f"{values.shape[0]} rows but the time vector has {len(times)}."
            )

        fixed_roi_std, fixed_roi_ci_low, fixed_roi_ci_high, fixed_roi_audit = resolve_uncertainty_bundle(
            values,
            loader_optional,
            paths_used,
            map_spec.get("fixed_roi_uncertainty"),
            z=z_value,
        )
        combined_std, combined_ci_low, combined_ci_high, combined_audit = resolve_uncertainty_bundle(
            values,
            loader_optional,
            paths_used,
            map_spec.get("combined_uncertainty"),
            z=z_value,
        )

        valid_rows = np.any(np.isfinite(values), axis=1)
        maps[map_spec["key"]] = MapData(
            key=map_spec["key"],
            label=map_spec["title"],
            values=values,
            valid_rows=valid_rows,
            fixed_roi_std=fixed_roi_std,
            fixed_roi_ci_low=fixed_roi_ci_low,
            fixed_roi_ci_high=fixed_roi_ci_high,
            combined_std=combined_std,
            combined_ci_low=combined_ci_low,
            combined_ci_high=combined_ci_high,
            uncertainty_audit={
                "fixed_roi": fixed_roi_audit,
                "combined": combined_audit,
            },
        )
        map_audit[map_spec["key"]] = {
            "shape": list(values.shape),
            "valid_row_count": int(np.count_nonzero(valid_rows)),
            "uncertainty": maps[map_spec["key"]].uncertainty_audit,
        }

    local_effective_fit_rmse_map = None
    local_effective_rmse_df = loader_optional(paths_used.get(LOCAL_D_RMSE_PATH_KEY))
    if local_effective_rmse_df is not None:
        local_effective_fit_rmse_map = to_numeric_2d(local_effective_rmse_df)
        local_map_shape = maps[LOCAL_D_MAP_KEY].values.shape if LOCAL_D_MAP_KEY in maps else None
        if local_map_shape is not None and local_effective_fit_rmse_map.shape != local_map_shape:
            raise ValueError(
                f"Local effective fit RMSE map for sample '{sample_cfg.get('sample_id', run_display)}' has shape "
                f"{list(local_effective_fit_rmse_map.shape)} but expected {list(local_map_shape)}."
            )
    if LOCAL_D_MAP_KEY in map_audit:
        map_audit[LOCAL_D_MAP_KEY]["has_local_fit_rmse_map"] = bool(local_effective_fit_rmse_map is not None)

    sample_id = sample_cfg.get("sample_id")
    if sample_id in (None, ""):
        sample_id = Path(run_display).name

    audit_info = {
        "analysis_config_path": analysis_config_path,
        "run_metadata_path": run_metadata_path,
        "time_column": time_column,
        "roi_inferred_from_run_metadata": first_roi_name_from_run_metadata(run_metadata_doc),
        "roi_inferred_from_analysis_config": first_roi_name_from_analysis_config(analysis_config_doc),
        "dx_inferred_from_run_metadata": depth_spacing_from_run_metadata(run_metadata_doc),
        "paths_used": paths_used,
        "maps": map_audit,
    }

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
        time_column=time_column,
        analysis_config_path=analysis_config_path,
        run_metadata_path=run_metadata_path,
        paths_used=paths_used,
        times=times,
        depth=build_depth_mm(next(iter(maps.values())).values.shape[1], float(dx_value)),
        maps=maps,
        local_effective_fit_rmse_map=local_effective_fit_rmse_map,
        audit_info=audit_info,
    )


def build_common_depth_grid(samples: Sequence[SampleData]) -> np.ndarray:
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


def build_common_time_grid(
    samples: Sequence[SampleData],
    shared_tmin: Optional[float],
    shared_tmax: Optional[float],
) -> np.ndarray:
    min_time = float(shared_tmin) if shared_tmin is not None else min(float(np.nanmin(sample.times)) for sample in samples)
    max_time = float(shared_tmax) if shared_tmax is not None else max(float(np.nanmax(sample.times)) for sample in samples)
    dt_candidates = []
    for sample in samples:
        finite_times = sample.times[np.isfinite(sample.times)]
        if finite_times.size > 1:
            diffs = np.diff(finite_times)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                dt_candidates.append(float(np.min(diffs)))
    dt = min(dt_candidates) if dt_candidates else 1.0
    point_count = int(np.floor((max_time - min_time) / dt)) + 1
    grid = min_time + np.arange(point_count, dtype=float) * dt
    if grid.size == 0 or grid[-1] < max_time - 0.25 * dt:
        grid = np.append(grid, max_time)
    return grid


def aggregate_tracer_map(
    samples: Sequence[SampleData],
    map_key: str,
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
) -> Dict[str, Any]:
    regridded = []
    time_ranges = []
    for sample in samples:
        map_data = sample.maps[map_key]
        regridded.append(regrid_map(sample.times, sample.depth, map_data.values, time_grid, depth_grid))
        valid_times = sample.times[map_data.valid_rows & np.isfinite(sample.times)]
        if valid_times.size:
            time_ranges.append((float(np.min(valid_times)), float(np.max(valid_times))))
    aggregate = nanmean_stack(np.stack(regridded, axis=0))
    return {
        "map": aggregate,
        "sample_count": len(samples),
        "time_ranges": time_ranges,
    }


def resolve_local_d_window(report_windows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    for window in report_windows:
        if str(window.get("name", "")).strip().lower() == "post_5min":
            return {
                "name": str(window.get("name", "post_5min")),
                "min_time_min": float(window.get("min_time_min")) if window.get("min_time_min") is not None else None,
                "max_time_min": float(window.get("max_time_min")) if window.get("max_time_min") is not None else None,
            }
    return {
        "name": "post_5min",
        "min_time_min": float(LOCAL_D_DEFAULT_WINDOW_MIN),
        "max_time_min": None,
    }


def time_mask_for_window(times: np.ndarray, min_time_min: Optional[float], max_time_min: Optional[float]) -> np.ndarray:
    mask = np.isfinite(times)
    if min_time_min is not None:
        mask &= np.asarray(times, dtype=float) >= float(min_time_min)
    if max_time_min is not None:
        mask &= np.asarray(times, dtype=float) <= float(max_time_min)
    return mask


def local_d_bound_hits(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    return finite & (
        np.isclose(arr, LOCAL_D_BOUND_LO, rtol=LOCAL_D_BOUND_RTOL, atol=LOCAL_D_BOUND_ATOL)
        | np.isclose(arr, LOCAL_D_BOUND_HI, rtol=LOCAL_D_BOUND_RTOL, atol=LOCAL_D_BOUND_ATOL)
    )


def compute_local_d_rmse_threshold(
    samples: Sequence[SampleData],
    min_time_min: Optional[float],
    max_time_min: Optional[float],
) -> float:
    rmse_values: List[np.ndarray] = []
    for sample in samples:
        if sample.local_effective_fit_rmse_map is None or LOCAL_D_MAP_KEY not in sample.maps:
            continue
        time_mask = time_mask_for_window(sample.times, min_time_min, max_time_min)
        row_mask = time_mask & sample.maps[LOCAL_D_MAP_KEY].valid_rows
        if not np.any(row_mask):
            continue
        finite = np.asarray(sample.local_effective_fit_rmse_map[row_mask], dtype=float).ravel()
        finite = finite[np.isfinite(finite)]
        if finite.size:
            rmse_values.append(finite)
    if not rmse_values:
        return float("nan")
    merged = np.concatenate(rmse_values)
    median = float(np.nanmedian(merged))
    q1, q3 = np.nanpercentile(merged, [25.0, 75.0])
    iqr = float(q3 - q1)
    if np.isfinite(iqr) and iqr > 0:
        return float(median + LOCAL_D_RMSE_IQR_MULT * iqr)
    return float(np.nanpercentile(merged, 90.0))


def save_matrix_csv(
    out_folder: str,
    subfolder: str,
    filename: str,
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
    values: np.ndarray,
) -> str:
    folder = Path(out_folder) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(np.asarray(values, dtype=float), columns=[f"depth_mm_{depth:.3f}" for depth in np.asarray(depth_grid, dtype=float)])
    df.insert(0, "time_min", np.asarray(time_grid, dtype=float))
    out_path = folder / filename
    df.to_csv(out_path, index=False)
    return str(out_path)


def apply_bottom_row_trim_2d(values: np.ndarray, bottom_rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if bottom_rows > 0 and arr.ndim == 2:
        n_depth = arr.shape[1]
        trim = min(bottom_rows, n_depth)
        if trim > 0:
            arr[:, -trim:] = np.nan
    return arr


def apply_bottom_row_trim_1d(values: np.ndarray, bottom_rows: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    if bottom_rows > 0 and arr.ndim == 1:
        trim = min(bottom_rows, arr.size)
        if trim > 0:
            arr[-trim:] = np.nan
    return arr


def build_local_d_post_window_products(
    samples: Sequence[SampleData],
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
    min_time_min: Optional[float],
    max_time_min: Optional[float],
    rmse_threshold: float,
) -> Dict[str, Any]:
    time_window_mask = time_mask_for_window(time_grid, min_time_min, max_time_min)
    time_grid_window = np.asarray(time_grid, dtype=float)[time_window_mask]

    masked_local_d_rows = []
    rmse_rows = []
    reliable_rows = []
    bound_hit_rows = []

    for sample in samples:
        map_data = sample.maps[LOCAL_D_MAP_KEY]
        local_d_grid = regrid_map(sample.times, sample.depth, map_data.values, time_grid, depth_grid)
        if sample.local_effective_fit_rmse_map is not None:
            rmse_grid = regrid_map(sample.times, sample.depth, sample.local_effective_fit_rmse_map, time_grid, depth_grid)
        else:
            rmse_grid = np.full_like(local_d_grid, np.nan, dtype=float)

        finite_local_d = np.isfinite(local_d_grid)
        bound_hits = local_d_bound_hits(local_d_grid)
        rmse_bad = (
            np.isfinite(rmse_grid)
            & np.isfinite(rmse_threshold)
            & (rmse_grid > float(rmse_threshold))
        )
        reliable = finite_local_d & ~bound_hits & ~rmse_bad

        masked_local_d_rows.append(np.where(reliable, local_d_grid, np.nan)[time_window_mask])
        rmse_rows.append(np.where(np.isfinite(rmse_grid), rmse_grid, np.nan)[time_window_mask])
        reliable_rows.append(reliable[time_window_mask].astype(float))
        bound_hit_rows.append(bound_hits[time_window_mask].astype(float))

    masked_local_d_stack = np.stack(masked_local_d_rows, axis=0)
    rmse_stack = np.stack(rmse_rows, axis=0)
    reliable_stack = np.stack(reliable_rows, axis=0)
    bound_hit_stack = np.stack(bound_hit_rows, axis=0)

    masked_mean_map = nanmean_stack(masked_local_d_stack)
    mean_rmse_map = nanmean_stack(rmse_stack)
    support_fraction_map = np.sum(reliable_stack, axis=0) / float(len(samples))
    bound_hit_fraction_map = np.sum(bound_hit_stack, axis=0) / float(len(samples))
    strict_support_mask = np.sum(reliable_stack, axis=0) >= int(len(samples))
    strict_masked_mean_map = np.where(strict_support_mask, masked_mean_map, np.nan)

    return {
        "time_grid_window": time_grid_window,
        "masked_mean_map": strict_masked_mean_map,
        "support_fraction_map": support_fraction_map,
        "mean_rmse_map": mean_rmse_map,
        "bound_hit_fraction_map": bound_hit_fraction_map,
        "strict_support_mask": strict_support_mask,
    }


def save_local_d_supplemental_outputs(
    out_folder: str,
    figure_title: str,
    tracer_order: Sequence[str],
    tracer_labels: Dict[str, str],
    products_by_tracer: Dict[str, Dict[str, Any]],
    depth_grid: np.ndarray,
    plot_depth_zero_at_top: bool,
    rmse_threshold: float,
    window_info: Dict[str, Any],
    color_limit_override: Optional[Dict[str, float]] = None,
    display_trim_override: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    output_paths: Dict[str, Any] = {"figures": [], "csvs": []}
    subfolder = "secondary"
    bottom_rows = int((display_trim_override or {}).get("bottom_rows", 0))
    arrays = [
        apply_bottom_row_trim_2d(products_by_tracer[tracer_name]["masked_mean_map"], bottom_rows)
        for tracer_name in tracer_order
    ]
    vmin, vmax = get_color_limits(arrays, nonnegative=True, robust=True, lower_pct=2.0, upper_pct=98.0)
    if color_limit_override:
        if color_limit_override.get("vmin") is not None:
            vmin = float(color_limit_override["vmin"])
        if color_limit_override.get("vmax") is not None:
            vmax = float(color_limit_override["vmax"])
    time_grid_window = products_by_tracer[tracer_order[0]]["time_grid_window"]

    fig, axes = plt.subplots(1, len(tracer_order), figsize=(5.9 * len(tracer_order), 5.6), constrained_layout=True)
    if len(tracer_order) == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0.0)
    last_im = None
    for ax, tracer_name in zip(axes, tracer_order):
        panel = products_by_tracer[tracer_name]
        display_masked_mean = apply_bottom_row_trim_2d(panel["masked_mean_map"], bottom_rows)
        masked_map = np.ma.masked_invalid(display_masked_mean)
        last_im = ax.imshow(
            masked_map.T,
            aspect="auto",
            extent=[float(time_grid_window[0]), float(time_grid_window[-1]), float(depth_grid[0]), float(depth_grid[-1])],
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(f"{tracer_labels[tracer_name]}\npost-5 min mean")
        ax.set_xlabel("Time (min)")
        ax.set_ylabel("Depth (mm)")
        if plot_depth_zero_at_top:
            ax.invert_yaxis()
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=np.ravel(axes).tolist(), shrink=0.95)
        cbar.set_label(r"Local effective diffusivity, $D_{eff}$ (mm$^2$/s)")
    fig.suptitle(
        f"{figure_title}: Local Effective Diffusivity\n"
        f"Post-5 min mean over the reliable depth range",
        fontsize=14,
    )
    out_path = save_plot(fig, out_folder, subfolder, "local_effective_diffusivity_post5_masked_mean_by_tracer.png")
    output_paths["figures"].append(out_path)

    support_arrays = [products_by_tracer[tracer_name]["support_fraction_map"] for tracer_name in tracer_order]
    rmse_arrays = [products_by_tracer[tracer_name]["mean_rmse_map"] for tracer_name in tracer_order]
    support_arrays = [apply_bottom_row_trim_2d(arr, bottom_rows) for arr in support_arrays]
    rmse_arrays = [apply_bottom_row_trim_2d(arr, bottom_rows) for arr in rmse_arrays]
    rmse_vmin, rmse_vmax = get_color_limits(rmse_arrays, nonnegative=True, robust=True, lower_pct=2.0, upper_pct=98.0)
    fig, axes = plt.subplots(2, len(tracer_order), figsize=(5.9 * len(tracer_order), 8.2), constrained_layout=True)
    if len(tracer_order) == 1:
        axes = np.asarray([[axes[0]], [axes[1]]], dtype=object)
    support_im = None
    rmse_im = None
    for col_idx, tracer_name in enumerate(tracer_order):
        panel = products_by_tracer[tracer_name]
        display_support = apply_bottom_row_trim_2d(panel["support_fraction_map"], bottom_rows)
        display_rmse = apply_bottom_row_trim_2d(panel["mean_rmse_map"], bottom_rows)
        ax_support = axes[0, col_idx]
        ax_rmse = axes[1, col_idx]

        support_im = ax_support.imshow(
            np.asarray(display_support, dtype=float).T,
            aspect="auto",
            extent=[float(time_grid_window[0]), float(time_grid_window[-1]), float(depth_grid[0]), float(depth_grid[-1])],
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="viridis",
        )
        ax_support.set_title(f"{tracer_labels[tracer_name]}\nsupport fraction")
        ax_support.set_xlabel("Time (min)")
        ax_support.set_ylabel("Depth (mm)")

        rmse_im = ax_rmse.imshow(
            np.ma.masked_invalid(display_rmse).T,
            aspect="auto",
            extent=[float(time_grid_window[0]), float(time_grid_window[-1]), float(depth_grid[0]), float(depth_grid[-1])],
            origin="lower",
            vmin=rmse_vmin,
            vmax=rmse_vmax,
            cmap="magma",
        )
        ax_rmse.set_title(f"{tracer_labels[tracer_name]}\nmean local-fit RMSE")
        ax_rmse.set_xlabel("Time (min)")
        ax_rmse.set_ylabel("Depth (mm)")

        if plot_depth_zero_at_top:
            ax_support.invert_yaxis()
            ax_rmse.invert_yaxis()

    if support_im is not None:
        cbar_support = fig.colorbar(support_im, ax=axes[0, :].ravel().tolist(), shrink=0.88)
        cbar_support.set_label("Replicate support fraction")
    if rmse_im is not None:
        cbar_rmse = fig.colorbar(rmse_im, ax=axes[1, :].ravel().tolist(), shrink=0.88)
        cbar_rmse.set_label("Local fit RMSE (mg/mL)")
    fig.suptitle(
        f"{figure_title}: Local Effective Diffusivity Supplementary QC\n"
        f"Top: replicate support fraction; bottom: mean local-fit RMSE",
        fontsize=14,
    )
    out_path = save_plot(fig, out_folder, subfolder, "local_effective_diffusivity_post5_qc_by_tracer.png")
    output_paths["figures"].append(out_path)

    ratio_tracer_a = "GAD" if "GAD" in products_by_tracer else tracer_order[0]
    ratio_tracer_b = "VIS320" if "VIS320" in products_by_tracer else tracer_order[1]
    map_a = apply_bottom_row_trim_2d(products_by_tracer[ratio_tracer_a]["masked_mean_map"], bottom_rows)
    map_b = apply_bottom_row_trim_2d(products_by_tracer[ratio_tracer_b]["masked_mean_map"], bottom_rows)
    with np.errstate(divide="ignore", invalid="ignore"):
        log2_ratio = np.log2(map_a / np.maximum(map_b, LOCAL_D_RATIO_EPS))
    finite_ratio = log2_ratio[np.isfinite(log2_ratio)]
    ratio_limit = float(np.nanpercentile(np.abs(finite_ratio), 98.0)) if finite_ratio.size else 1.0
    if not np.isfinite(ratio_limit) or np.isclose(ratio_limit, 0.0):
        ratio_limit = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 5.6), constrained_layout=True)
    im = ax.imshow(
        np.ma.masked_invalid(log2_ratio).T,
        aspect="auto",
        extent=[float(time_grid_window[0]), float(time_grid_window[-1]), float(depth_grid[0]), float(depth_grid[-1])],
        origin="lower",
        cmap="coolwarm",
        norm=mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-ratio_limit, vmax=ratio_limit),
    )
    ax.set_title(f"log2({tracer_labels[ratio_tracer_a]} / {tracer_labels[ratio_tracer_b]})")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Depth (mm)")
    if plot_depth_zero_at_top:
        ax.invert_yaxis()
    cbar = fig.colorbar(im, ax=ax, shrink=0.92)
    cbar.set_label(f"log2({tracer_labels[ratio_tracer_a]} / {tracer_labels[ratio_tracer_b]})")
    fig.suptitle(
        f"{figure_title}: Relative Local Effective Diffusivity\n"
        f"log2({tracer_labels[ratio_tracer_a]} / {tracer_labels[ratio_tracer_b]}); positive values indicate higher $D_{{eff}}$ for {tracer_labels[ratio_tracer_a]}",
        fontsize=14,
    )
    out_path = save_plot(fig, out_folder, subfolder, "local_effective_diffusivity_post5_log2_ratio_gad_over_vis.png")
    output_paths["figures"].append(out_path)

    for tracer_name in tracer_order:
        panel = products_by_tracer[tracer_name]
        safe_tracer = str(tracer_name).replace(" ", "_")
        output_paths["csvs"].append(
            save_matrix_csv(
                out_folder,
                subfolder,
                f"local_effective_diffusivity_post5_masked_mean_{safe_tracer}.csv",
                time_grid_window,
                depth_grid,
                panel["masked_mean_map"],
            )
        )
        output_paths["csvs"].append(
            save_matrix_csv(
                out_folder,
                subfolder,
                f"local_effective_diffusivity_post5_support_fraction_{safe_tracer}.csv",
                time_grid_window,
                depth_grid,
                panel["support_fraction_map"],
            )
        )
        output_paths["csvs"].append(
            save_matrix_csv(
                out_folder,
                subfolder,
                f"local_effective_diffusivity_post5_mean_rmse_{safe_tracer}.csv",
                time_grid_window,
                depth_grid,
                panel["mean_rmse_map"],
            )
        )
        output_paths["csvs"].append(
            save_matrix_csv(
                out_folder,
                subfolder,
                f"local_effective_diffusivity_post5_bound_hit_fraction_{safe_tracer}.csv",
                time_grid_window,
                depth_grid,
                panel["bound_hit_fraction_map"],
            )
        )

    output_paths["csvs"].append(
        save_matrix_csv(
            out_folder,
            subfolder,
            "local_effective_diffusivity_post5_log2_ratio_gad_over_vis.csv",
            time_grid_window,
            depth_grid,
            log2_ratio,
        )
    )

    output_paths["audit"] = {
        "window_name": window_info["name"],
        "window_min_time_min": window_info["min_time_min"],
        "window_max_time_min": window_info["max_time_min"],
        "rmse_threshold": float(rmse_threshold) if np.isfinite(rmse_threshold) else float("nan"),
        "strict_support_requires_all_replicates": True,
        "local_d_bounds": [LOCAL_D_BOUND_LO, LOCAL_D_BOUND_HI],
        "ratio_tracer_numerator": ratio_tracer_a,
        "ratio_tracer_denominator": ratio_tracer_b,
    }
    return output_paths


def compute_map_band(map_data: MapData, idx: int, mode: str, z_value: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    center = map_data.values[idx]
    if mode == "fixed_roi_95ci":
        if map_data.fixed_roi_ci_low is not None and map_data.fixed_roi_ci_high is not None:
            return map_data.fixed_roi_ci_low[idx], map_data.fixed_roi_ci_high[idx]
        if map_data.fixed_roi_std is not None:
            band = z_value * map_data.fixed_roi_std[idx]
            return center - band, center + band
    if mode == "combined_95ci":
        if map_data.combined_ci_low is not None and map_data.combined_ci_high is not None:
            return map_data.combined_ci_low[idx], map_data.combined_ci_high[idx]
        if map_data.combined_std is not None:
            band = z_value * map_data.combined_std[idx]
            return center - band, center + band
    return None, None


def aggregate_tracer_target_profile(
    samples: Sequence[SampleData],
    map_key: str,
    target_time: float,
    matched_indices: Dict[str, Dict[str, Dict[float, int]]],
    depth_grid: np.ndarray,
    mode: str,
    z_value: float,
) -> Dict[str, Any]:
    rows = []
    band_low_rows = []
    band_high_rows = []
    actual_times = []
    sample_curves = []

    for sample in samples:
        idx = matched_indices[sample.sample_id][map_key][target_time]
        map_data = sample.maps[map_key]
        row = interp_profile(sample.depth, map_data.values[idx], depth_grid)
        rows.append(row)
        actual_times.append(float(sample.times[idx]))
        sample_curves.append({"sample_id": sample.sample_id, "depth": depth_grid, "values": row})

        low, high = compute_map_band(map_data, idx, mode, z_value)
        if low is not None and high is not None:
            band_low_rows.append(interp_profile(sample.depth, low, depth_grid))
            band_high_rows.append(interp_profile(sample.depth, high, depth_grid))

    values_stack = np.vstack(rows)
    aggregate = {
        "depth": depth_grid,
        "mean_values": nanmean_stack(values_stack),
        "sample_curves": sample_curves,
        "sample_count": len(samples),
        "actual_times": actual_times,
        "band_low": None,
        "band_high": None,
    }
    if band_low_rows and band_high_rows:
        aggregate["band_low"] = nanmean_stack(np.vstack(band_low_rows))
        aggregate["band_high"] = nanmean_stack(np.vstack(band_high_rows))
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


def save_plot(fig: plt.Figure, out_folder: str, subfolder: str, filename: str) -> str:
    folder = Path(out_folder) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def line_mode_note(mode: str) -> str:
    notes = {
        "none": "Bold lines show tracer means; faint lines show individual samples.",
        "fixed_roi_95ci": "Band shows the average within-sample fixed-ROI 95% interval.",
        "combined_95ci": "Band shows the average within-sample combined 95% interval.",
    }
    return notes[mode]


def save_heatmap_figure(
    out_folder: str,
    map_spec: Dict[str, Any],
    tracer_order: Sequence[str],
    aggregated_maps: Dict[str, Dict[str, Any]],
    tracer_labels: Dict[str, str],
    figure_title: str,
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
    target_times: Sequence[float],
    time_unit_label: str,
    plot_depth_zero_at_top: bool,
    robust_color_limits: bool,
    lower_percentile: float,
    upper_percentile: float,
    color_limit_override: Optional[Dict[str, float]] = None,
    display_trim_override: Optional[Dict[str, int]] = None,
) -> str:
    bottom_rows = int((display_trim_override or {}).get("bottom_rows", 0))
    arrays = [apply_bottom_row_trim_2d(aggregated_maps[tracer_name]["map"], bottom_rows) for tracer_name in tracer_order]
    vmin, vmax = get_color_limits(
        arrays,
        nonnegative=bool(map_spec.get("nonnegative", False)),
        robust=robust_color_limits,
        lower_pct=lower_percentile,
        upper_pct=upper_percentile,
    )
    if color_limit_override:
        if color_limit_override.get("vmin") is not None:
            vmin = float(color_limit_override["vmin"])
        if color_limit_override.get("vmax") is not None:
            vmax = float(color_limit_override["vmax"])

    fig, axes = plt.subplots(1, len(tracer_order), figsize=(5.6 * len(tracer_order), 5.5), constrained_layout=True)
    if len(tracer_order) == 1:
        axes = [axes]
    cmap = plt.cm.get_cmap("viridis").copy()
    cmap.set_bad(color="white", alpha=0.0)

    last_im = None
    for ax, tracer_name in zip(axes, tracer_order):
        panel = aggregated_maps[tracer_name]
        display_map = apply_bottom_row_trim_2d(panel["map"], bottom_rows)
        masked_map = np.ma.masked_invalid(display_map)
        last_im = ax.imshow(
            masked_map.T,
            aspect="auto",
            extent=[float(time_grid[0]), float(time_grid[-1]), float(depth_grid[0]), float(depth_grid[-1])],
            origin="lower",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        ax.set_title(f"{tracer_labels[tracer_name]} (n={panel['sample_count']})")
        ax.set_xlabel(f"Time ({time_unit_label})")
        ax.set_ylabel("Depth (mm)")
        for idx, target_time in enumerate(target_times[: len(TIMEPOINT_COLORS)]):
            ax.axvline(float(target_time), color=TIMEPOINT_COLORS[idx], linestyle="--", linewidth=1.8, alpha=0.95)
        if plot_depth_zero_at_top:
            ax.invert_yaxis()

    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=np.ravel(axes).tolist(), shrink=0.95)
        cbar.set_label(map_spec["cbar_label"])
    fig.suptitle(f"{figure_title}: {map_spec['title']}\nMean map within each tracer", fontsize=14)
    return save_plot(
        fig,
        out_folder,
        map_spec["subfolder"],
        f"{map_spec['out_stem']}_mean_map_by_tracer.png",
    )


def save_target_profile_figure(
    out_folder: str,
    map_spec: Dict[str, Any],
    tracer_order: Sequence[str],
    samples_by_tracer: Dict[str, List[SampleData]],
    tracer_labels: Dict[str, str],
    tracer_styles: Dict[str, Dict[str, str]],
    target_times: Sequence[float],
    matched_indices: Dict[str, Dict[str, Dict[float, int]]],
    depth_grid: np.ndarray,
    figure_title: str,
    time_unit_label: str,
    mode: str,
    ylims: Tuple[float, float],
    z_value: float,
    fixed_roi_fill_alpha: float,
    display_trim_override: Optional[Dict[str, int]] = None,
) -> Optional[str]:
    n_panels = len(target_times)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, max(4.0 * n_panels, 7.0)), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]
    panel_names = build_panel_names(n_panels)
    any_band = False
    bottom_rows = int((display_trim_override or {}).get("bottom_rows", 0))

    for ax, panel_name, target_time in zip(axes, panel_names, target_times):
        tracer_panel_data = {}
        for tracer_name in tracer_order:
            tracer_samples = samples_by_tracer[tracer_name]
            panel_data = aggregate_tracer_target_profile(
                tracer_samples,
                map_spec["key"],
                target_time,
                matched_indices,
                depth_grid,
                mode,
                z_value,
            )
            tracer_panel_data[tracer_labels[tracer_name]] = panel_data
            style = tracer_styles[tracer_name]

            for sample_curve in panel_data["sample_curves"]:
                sample_values = apply_bottom_row_trim_1d(sample_curve["values"], bottom_rows)
                ax.plot(
                    sample_curve["depth"],
                    sample_values,
                    color=style["color"],
                    linewidth=SAMPLE_LINEWIDTH,
                    alpha=SAMPLE_LINE_ALPHA,
                )

            if panel_data["band_low"] is not None and panel_data["band_high"] is not None:
                any_band = True
                band_low = apply_bottom_row_trim_1d(panel_data["band_low"], bottom_rows)
                band_high = apply_bottom_row_trim_1d(panel_data["band_high"], bottom_rows)
                ax.fill_between(
                    panel_data["depth"],
                    band_low,
                    band_high,
                    color=style["color"],
                    alpha=fixed_roi_fill_alpha if mode == "fixed_roi_95ci" else 0.10,
                )

            mean_values = apply_bottom_row_trim_1d(panel_data["mean_values"], bottom_rows)
            ax.plot(
                panel_data["depth"],
                mean_values,
                color=style["color"],
                linewidth=CENTER_LINEWIDTH,
                label=f"{tracer_labels[tracer_name]} mean (n={panel_data['sample_count']})",
            )

        title = f"{panel_name} (~{target_time:.2f} {time_unit_label})"
        subtitle = build_panel_subtitle(tracer_panel_data, time_unit_label)
        if subtitle:
            title += "\nActual matched frames: " + subtitle
        ax.set_title(title)
        ax.set_ylabel(map_spec["line_value_label"])
        ax.grid(True)
        ax.set_ylim(*ylims)

    if mode != "none" and not any_band:
        plt.close(fig)
        return None

    axes[-1].set_xlabel("Depth (mm)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(4, max(2, len(labels))),
        bbox_to_anchor=(0.5, FIGURE_LEGEND_Y),
        borderaxespad=LEGEND_BORDER_AXES_PAD,
    )
    fig.suptitle(f"{figure_title}: {map_spec['title']}\n{line_mode_note(mode)}", y=FIGURE_SUPTITLE_Y, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, FIGURE_TOP_RECT])
    mode_suffix = next(spec[2] for spec in LINE_MODE_SPECS if spec[0] == mode)
    return save_plot(fig, out_folder, map_spec["subfolder"], f"{map_spec['out_stem']}_{mode_suffix}")


def build_map_profile_ylims(
    samples: Sequence[SampleData],
    map_key: str,
    z_value: float,
) -> Tuple[float, float]:
    arrays: List[np.ndarray] = []
    for sample in samples:
        map_data = sample.maps[map_key]
        arrays.append(map_data.values)
        if map_data.fixed_roi_std is not None:
            arrays.append(map_data.values - z_value * map_data.fixed_roi_std)
            arrays.append(map_data.values + z_value * map_data.fixed_roi_std)
        if map_data.combined_std is not None:
            arrays.append(map_data.values - z_value * map_data.combined_std)
            arrays.append(map_data.values + z_value * map_data.combined_std)
        if map_data.fixed_roi_ci_low is not None and map_data.fixed_roi_ci_high is not None:
            arrays.extend([map_data.fixed_roi_ci_low, map_data.fixed_roi_ci_high])
        if map_data.combined_ci_low is not None and map_data.combined_ci_high is not None:
            arrays.extend([map_data.combined_ci_low, map_data.combined_ci_high])
    ylims, _ = compute_axis_limits(*arrays)
    return ylims


def compute_metric_rows_for_index(
    sample: SampleData,
    map_spec: Dict[str, Any],
    row_index: int,
    target_time: Optional[float] = None,
) -> List[Dict[str, Any]]:
    map_data = sample.maps[map_spec["key"]]
    values = map_data.values[row_index]
    actual_time = float(sample.times[row_index])

    fixed_roi_ci_width = None
    if map_data.fixed_roi_ci_low is not None and map_data.fixed_roi_ci_high is not None:
        fixed_roi_ci_width = map_data.fixed_roi_ci_high[row_index] - map_data.fixed_roi_ci_low[row_index]

    combined_ci_width = None
    if map_data.combined_ci_low is not None and map_data.combined_ci_high is not None:
        combined_ci_width = map_data.combined_ci_high[row_index] - map_data.combined_ci_low[row_index]

    metric_values = {
        "depth_mean": safe_mean(values),
        "depth_median": safe_median(values),
        "depth_max": safe_max(values),
        "surface_value": safe_first(values),
        "depth_integral_x_mm": safe_integral(sample.depth, values),
        "penetration_depth_10pct_peak_mm": penetration_depth(sample.depth, values, threshold_fraction=0.10),
        "fixed_roi_std_depth_mean": safe_mean(map_data.fixed_roi_std[row_index]) if map_data.fixed_roi_std is not None else float("nan"),
        "fixed_roi_ci_width_depth_mean": safe_mean(fixed_roi_ci_width) if fixed_roi_ci_width is not None else float("nan"),
        "combined_std_depth_mean": safe_mean(map_data.combined_std[row_index]) if map_data.combined_std is not None else float("nan"),
        "combined_ci_width_depth_mean": safe_mean(combined_ci_width) if combined_ci_width is not None else float("nan"),
    }

    base_row = {
        "tracer_name": sample.tracer_name,
        "tracer_label": sample.tracer_label,
        "sample_id": sample.sample_id,
        "run_display": sample.run_display,
        "roi_folder": sample.roi_folder,
        "dx_mm": float(sample.dx_mm),
        "depth_points": int(len(sample.depth)),
        "finite_depth_points": int(np.count_nonzero(np.isfinite(values))),
        "map_key": map_spec["key"],
        "map_label": map_spec["title"],
        "actual_time_min": actual_time,
    }
    if target_time is not None:
        base_row["target_time_min"] = float(target_time)
        base_row["time_offset_min"] = float(actual_time - float(target_time))

    rows = []
    for metric_name, metric_value in metric_values.items():
        row = dict(base_row)
        row["metric_name"] = metric_name
        row["metric_label"] = MAP_METRIC_LABELS.get(metric_name, metric_name)
        row["metric_value"] = float(metric_value) if np.isfinite(metric_value) else float("nan")
        rows.append(row)
    return rows


def build_group_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "target_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "tracer_name",
        "tracer_label",
        "n",
        "mean",
        "std",
        "sem",
        "min",
        "max",
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = ["target_time_min", "map_key", "map_label", "metric_name", "metric_label", "tracer_name", "tracer_label"]
    for group_values, group_df in metrics_df.groupby(group_cols, dropna=False):
        values = pd.to_numeric(group_df["metric_value"], errors="coerce").dropna()
        if values.empty:
            continue
        std_value = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        sem_value = float(std_value / np.sqrt(len(values))) if len(values) > 1 else float("nan")
        row = dict(zip(group_cols, group_values))
        row.update(
            {
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": std_value,
                "sem": sem_value,
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def build_statistics_tables(
    metrics_df: pd.DataFrame,
    metric_names: Sequence[str],
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_columns = [
        "target_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "test_name",
        "tracer_a",
        "tracer_a_label",
        "tracer_b",
        "tracer_b_label",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "mean_diff_a_minus_b",
        "statistic",
        "p_value_raw",
        "p_value_holm",
        "alpha",
        "significant",
        "note",
    ]
    omnibus_columns = [
        "target_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "test_name",
        "group_count",
        "statistic",
        "p_value",
        "alpha",
        "significant",
        "note",
    ]
    pairwise_rows: List[Dict[str, Any]] = []
    omnibus_rows: List[Dict[str, Any]] = []
    group_cols = ["target_time_min", "map_key", "map_label", "metric_name", "metric_label"]

    filtered_df = metrics_df[metrics_df["metric_name"].isin(metric_names)].copy()
    for group_values, group_df in filtered_df.groupby(group_cols, dropna=False):
        target_time, map_key, map_label, metric_name, metric_label = group_values
        tracer_groups: Dict[str, np.ndarray] = {}
        tracer_labels: Dict[str, str] = {}
        for tracer_name, tracer_df in group_df.groupby("tracer_name"):
            values = pd.to_numeric(tracer_df["metric_value"], errors="coerce").dropna().to_numpy(dtype=float)
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
                        "map_key": map_key,
                        "map_label": map_label,
                        "metric_name": metric_name,
                        "metric_label": metric_label,
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
                        "map_key": map_key,
                        "map_label": map_label,
                        "metric_name": metric_name,
                        "metric_label": metric_label,
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
                if np.isfinite(p_value):
                    raw_p_values.append(float(p_value))
                    raw_row_indices.append(len(pairwise_rows))
                else:
                    note = "Welch t-test returned NaN; groups may be identical or nearly identical."
            else:
                note = "Need n>=2 in each tracer group for Welch t-test."
            pairwise_rows.append(
                {
                    "target_time_min": float(target_time),
                    "map_key": map_key,
                    "map_label": map_label,
                    "metric_name": metric_name,
                    "metric_label": metric_label,
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

    return pd.DataFrame(pairwise_rows, columns=pairwise_columns), pd.DataFrame(omnibus_rows, columns=omnibus_columns)


def build_window_sample_summary(
    all_time_metrics_df: pd.DataFrame,
    report_windows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    output_columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "tracer_name",
        "tracer_label",
        "sample_id",
        "run_display",
        "roi_folder",
        "dx_mm",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "timepoint_count",
        "mean",
        "std",
        "median",
        "iqr",
        "min",
        "max",
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "tracer_name",
        "tracer_label",
        "sample_id",
        "run_display",
        "roi_folder",
        "dx_mm",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
    ]
    for window in report_windows:
        window_name = str(window["name"])
        min_time = float(window["min_time_min"]) if window.get("min_time_min") is not None else None
        max_time = float(window["max_time_min"]) if window.get("max_time_min") is not None else None
        window_df = all_time_metrics_df.copy()
        if min_time is not None:
            window_df = window_df[window_df["actual_time_min"] >= min_time]
        if max_time is not None:
            window_df = window_df[window_df["actual_time_min"] <= max_time]

        for group_values, sample_df in window_df.groupby(group_cols, dropna=False):
            stats_map = finite_stats(sample_df["metric_value"])
            row = dict(zip(group_cols, group_values))
            row.update(
                {
                    "window_name": window_name,
                    "window_min_time_min": min_time,
                    "window_max_time_min": max_time,
                    "timepoint_count": int(stats_map["n"]),
                    "mean": stats_map["mean"],
                    "std": stats_map["std"],
                    "median": stats_map["median"],
                    "iqr": stats_map["iqr"],
                    "min": stats_map["min"],
                    "max": stats_map["max"],
                }
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def build_window_group_summary(window_sample_summary_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "tracer_name",
        "tracer_label",
        "n_samples",
        "mean_of_sample_means",
        "std_of_sample_means",
        "sem_of_sample_means",
        "median_of_sample_means",
        "iqr_of_sample_means",
        "min_of_sample_means",
        "max_of_sample_means",
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "tracer_name",
        "tracer_label",
    ]
    for group_values, group_df in window_sample_summary_df.groupby(group_cols, dropna=False):
        stats_map = finite_stats(group_df["mean"])
        row = dict(zip(group_cols, group_values))
        row.update(
            {
                "n_samples": stats_map["n"],
                "mean_of_sample_means": stats_map["mean"],
                "std_of_sample_means": stats_map["std"],
                "sem_of_sample_means": stats_map["sem"],
                "median_of_sample_means": stats_map["median"],
                "iqr_of_sample_means": stats_map["iqr"],
                "min_of_sample_means": stats_map["min"],
                "max_of_sample_means": stats_map["max"],
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=output_columns)


def build_window_statistics_tables(
    window_sample_summary_df: pd.DataFrame,
    metric_names: Sequence[str],
    alpha: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pairwise_columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "test_name",
        "tracer_a",
        "tracer_a_label",
        "tracer_b",
        "tracer_b_label",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "mean_diff_a_minus_b",
        "statistic",
        "p_value_raw",
        "p_value_holm",
        "alpha",
        "significant",
        "note",
    ]
    omnibus_columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "map_key",
        "map_label",
        "metric_name",
        "metric_label",
        "test_name",
        "group_count",
        "statistic",
        "p_value",
        "alpha",
        "significant",
        "note",
    ]
    pairwise_rows: List[Dict[str, Any]] = []
    omnibus_rows: List[Dict[str, Any]] = []
    group_cols = ["window_name", "window_min_time_min", "window_max_time_min", "map_key", "map_label", "metric_name", "metric_label"]

    filtered_df = window_sample_summary_df[window_sample_summary_df["metric_name"].isin(metric_names)].copy()
    for group_values, metric_df in filtered_df.groupby(group_cols, dropna=False):
        (
            window_name,
            window_min_time_min,
            window_max_time_min,
            map_key,
            map_label,
            metric_name,
            metric_label,
        ) = group_values

        tracer_groups: Dict[str, np.ndarray] = {}
        tracer_labels: Dict[str, str] = {}
        for tracer_name, tracer_df in metric_df.groupby("tracer_name"):
            values = pd.to_numeric(tracer_df["mean"], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                tracer_groups[tracer_name] = values
                tracer_labels[tracer_name] = tracer_df["tracer_label"].iloc[0]

        if len(tracer_groups) >= 3:
            eligible_groups = [values for values in tracer_groups.values() if values.size >= 2]
            if len(eligible_groups) >= 3:
                statistic, p_value = stats.f_oneway(*eligible_groups)
                omnibus_rows.append(
                    {
                        "window_name": window_name,
                        "window_min_time_min": window_min_time_min,
                        "window_max_time_min": window_max_time_min,
                        "map_key": map_key,
                        "map_label": map_label,
                        "metric_name": metric_name,
                        "metric_label": metric_label,
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
                        "window_name": window_name,
                        "window_min_time_min": window_min_time_min,
                        "window_max_time_min": window_max_time_min,
                        "map_key": map_key,
                        "map_label": map_label,
                        "metric_name": metric_name,
                        "metric_label": metric_label,
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
                if np.isfinite(p_value):
                    raw_p_values.append(float(p_value))
                    raw_row_indices.append(len(pairwise_rows))
                else:
                    note = "Welch t-test returned NaN; groups may be identical or nearly identical."
            else:
                note = "Need n>=2 in each tracer group for Welch t-test."
            pairwise_rows.append(
                {
                    "window_name": window_name,
                    "window_min_time_min": window_min_time_min,
                    "window_max_time_min": window_max_time_min,
                    "map_key": map_key,
                    "map_label": map_label,
                    "metric_name": metric_name,
                    "metric_label": metric_label,
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

    return pd.DataFrame(pairwise_rows, columns=pairwise_columns), pd.DataFrame(omnibus_rows, columns=omnibus_columns)


def write_tables(
    out_folder: str,
    target_metrics_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    omnibus_df: pd.DataFrame,
    extra_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> List[str]:
    summary_dir = Path(out_folder) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    table_specs = [
        ("per_sample_target_time_map_metrics.csv", target_metrics_df),
        ("tracer_group_map_metric_summary.csv", group_summary_df),
        ("pairwise_map_significance_tests.csv", pairwise_df),
        ("omnibus_map_significance_tests.csv", omnibus_df),
    ]
    for filename, df in table_specs:
        out_path = summary_dir / filename
        df.to_csv(out_path, index=False)
        outputs.append(str(out_path))

    if extra_tables:
        for filename, df in extra_tables.items():
            out_path = summary_dir / filename
            df.to_csv(out_path, index=False)
            outputs.append(str(out_path))
    return outputs


def save_audit_report(out_folder: str, payload: Dict[str, Any]) -> None:
    out_dir = Path(out_folder) / "audit"
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
            for sub_key, sub_value in value.items():
                lines.append(f"  {sub_key}: {sub_value}")
        else:
            lines.append(f"{key}: {value}")
    with open(out_dir / "comparison_audit.txt", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare concentration, flux, and diffusivity maps across multiple tracers and replicate samples."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON map-comparison config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = validate_and_resolve_config(load_config(config_path), config_path)

    map_specs = config["map_specs"]
    tracer_order = [tracer_cfg["name"] for tracer_cfg in config["tracers"]]
    tracer_labels = {tracer_cfg["name"]: tracer_cfg["label"] for tracer_cfg in config["tracers"]}
    tracer_styles = {
        tracer_cfg["name"]: {"color": tracer_cfg["color"], "marker": tracer_cfg["marker"]}
        for tracer_cfg in config["tracers"]
    }

    samples_by_tracer: Dict[str, List[SampleData]] = {}
    all_samples: List[SampleData] = []
    for tracer_cfg in config["tracers"]:
        tracer_samples = [
            load_sample_data(
                tracer_cfg,
                sample_cfg,
                config.get("results_source"),
                map_specs,
                z_value=config["profile_fixed_roi_z"],
            )
            for sample_cfg in tracer_cfg["samples"]
        ]
        samples_by_tracer[tracer_cfg["name"]] = tracer_samples
        all_samples.extend(tracer_samples)

    target_times = choose_target_times(all_samples, config["target_time_map_key"], config.get("target_times_min"))
    depth_grid = build_common_depth_grid(all_samples)
    time_grid = build_common_time_grid(all_samples, config["shared_time_axis_min"], config["shared_time_axis_max"])

    matched_indices: Dict[str, Dict[str, Dict[float, int]]] = {}
    for sample in all_samples:
        matched_indices[sample.sample_id] = {}
        for map_spec in map_specs:
            map_key = map_spec["key"]
            map_data = sample.maps[map_key]
            matched_indices[sample.sample_id][map_key] = {
                target_time: nearest_index(sample.times, target_time, valid_mask=map_data.valid_rows)
                for target_time in target_times
            }

    figure_outputs: Dict[str, Dict[str, Any]] = {}
    for map_spec in map_specs:
        aggregated_maps = {
            tracer_name: aggregate_tracer_map(samples_by_tracer[tracer_name], map_spec["key"], time_grid, depth_grid)
            for tracer_name in tracer_order
        }
        heatmap_output = save_heatmap_figure(
            out_folder=config["out_folder"],
            map_spec=map_spec,
            tracer_order=tracer_order,
            aggregated_maps=aggregated_maps,
            tracer_labels=tracer_labels,
            figure_title=config["figure_title"],
            time_grid=time_grid,
            depth_grid=depth_grid,
            target_times=target_times,
            time_unit_label=config["time_unit_label"],
            plot_depth_zero_at_top=config["plot_depth_zero_at_top"],
            robust_color_limits=config["robust_color_limits"],
            lower_percentile=config["lower_percentile"],
            upper_percentile=config["upper_percentile"],
            color_limit_override=config.get("map_color_limits", {}).get(map_spec["key"]),
            display_trim_override=config.get("map_display_trim_rows", {}).get(map_spec["key"]),
        )
        ylims = build_map_profile_ylims(all_samples, map_spec["key"], z_value=config["profile_fixed_roi_z"])
        line_outputs = []
        for mode, _, _ in LINE_MODE_SPECS:
            out_path = save_target_profile_figure(
                out_folder=config["out_folder"],
                map_spec=map_spec,
                tracer_order=tracer_order,
                samples_by_tracer=samples_by_tracer,
                tracer_labels=tracer_labels,
                tracer_styles=tracer_styles,
                target_times=target_times,
                matched_indices=matched_indices,
                depth_grid=depth_grid,
                figure_title=config["figure_title"],
                time_unit_label=config["time_unit_label"],
                mode=mode,
                ylims=ylims,
                z_value=config["profile_fixed_roi_z"],
                fixed_roi_fill_alpha=config["profile_fixed_roi_fill_alpha"],
                display_trim_override=config.get("map_display_trim_rows", {}).get(map_spec["key"]),
            )
            if out_path is not None:
                line_outputs.append(out_path)
        figure_outputs[map_spec["key"]] = {
            "heatmap_output": heatmap_output,
            "line_outputs": line_outputs,
            "ylims_used": [float(ylims[0]), float(ylims[1])],
        }

    local_d_supplement_outputs: Dict[str, Any] = {}
    if LOCAL_D_MAP_KEY in [spec["key"] for spec in map_specs]:
        local_d_window = resolve_local_d_window(config["report_windows"])
        local_d_rmse_threshold = compute_local_d_rmse_threshold(
            all_samples,
            local_d_window["min_time_min"],
            local_d_window["max_time_min"],
        )
        local_d_products_by_tracer = {
            tracer_name: build_local_d_post_window_products(
                samples_by_tracer[tracer_name],
                time_grid,
                depth_grid,
                local_d_window["min_time_min"],
                local_d_window["max_time_min"],
                local_d_rmse_threshold,
            )
            for tracer_name in tracer_order
        }
        first_tracer = tracer_order[0] if tracer_order else None
        if first_tracer is not None and local_d_products_by_tracer[first_tracer]["time_grid_window"].size > 0:
            local_d_supplement_outputs = save_local_d_supplemental_outputs(
                out_folder=config["out_folder"],
                figure_title=config["figure_title"],
                tracer_order=tracer_order,
                tracer_labels=tracer_labels,
                products_by_tracer=local_d_products_by_tracer,
                depth_grid=depth_grid,
                plot_depth_zero_at_top=config["plot_depth_zero_at_top"],
                rmse_threshold=local_d_rmse_threshold,
                window_info=local_d_window,
                color_limit_override=config.get("map_color_limits", {}).get(LOCAL_D_MAP_KEY),
                display_trim_override=config.get("map_display_trim_rows", {}).get(LOCAL_D_MAP_KEY),
            )
            figure_outputs.setdefault(LOCAL_D_MAP_KEY, {})["supplemental_outputs"] = local_d_supplement_outputs

    target_metric_rows = []
    for sample in all_samples:
        for map_spec in map_specs:
            map_key = map_spec["key"]
            for target_time in target_times:
                matched_idx = matched_indices[sample.sample_id][map_key][target_time]
                target_metric_rows.extend(
                    compute_metric_rows_for_index(sample, map_spec, matched_idx, target_time=target_time)
                )
    target_metrics_df = pd.DataFrame(target_metric_rows)

    all_time_metric_rows = []
    for sample in all_samples:
        for map_spec in map_specs:
            map_data = sample.maps[map_spec["key"]]
            valid_indices = np.where(map_data.valid_rows & np.isfinite(sample.times))[0]
            for idx in valid_indices:
                all_time_metric_rows.extend(compute_metric_rows_for_index(sample, map_spec, int(idx), target_time=None))
    all_time_metrics_df = pd.DataFrame(all_time_metric_rows)

    group_summary_df = build_group_summary(target_metrics_df)
    available_metric_names = sorted(target_metrics_df["metric_name"].dropna().unique().tolist()) if not target_metrics_df.empty else []
    stats_metrics = [metric for metric in config["stats_metrics"] if metric in available_metric_names]
    pairwise_df, omnibus_df = build_statistics_tables(target_metrics_df, stats_metrics, alpha=config["stats_alpha"])

    extra_tables: Dict[str, pd.DataFrame] = {
        "per_sample_all_time_map_metrics.csv": all_time_metrics_df,
    }

    window_stats_metrics_config = config.get("window_stats_metrics")
    window_stats_metrics = (
        [metric for metric in window_stats_metrics_config if metric in available_metric_names]
        if window_stats_metrics_config
        else stats_metrics
    )
    window_sample_summary_df = build_window_sample_summary(all_time_metrics_df, config["report_windows"])
    if not window_sample_summary_df.empty:
        window_group_summary_df = build_window_group_summary(window_sample_summary_df)
        window_pairwise_df, window_omnibus_df = build_window_statistics_tables(
            window_sample_summary_df,
            window_stats_metrics,
            alpha=config["stats_alpha"],
        )
        extra_tables["window_per_sample_map_metric_summary.csv"] = window_sample_summary_df
        extra_tables["window_tracer_group_map_metric_summary.csv"] = window_group_summary_df
        extra_tables["window_pairwise_map_significance_tests.csv"] = window_pairwise_df
        extra_tables["window_omnibus_map_significance_tests.csv"] = window_omnibus_df

        for window_name in sorted(window_sample_summary_df["window_name"].dropna().unique()):
            safe_name = str(window_name).strip().replace(" ", "_")
            extra_tables[f"{safe_name}_per_sample_map_metric_summary.csv"] = window_sample_summary_df[
                window_sample_summary_df["window_name"] == window_name
            ].copy()
            extra_tables[f"{safe_name}_tracer_group_map_metric_summary.csv"] = window_group_summary_df[
                window_group_summary_df["window_name"] == window_name
            ].copy()
            extra_tables[f"{safe_name}_pairwise_map_significance_tests.csv"] = window_pairwise_df[
                window_pairwise_df["window_name"] == window_name
            ].copy()
            extra_tables[f"{safe_name}_omnibus_map_significance_tests.csv"] = window_omnibus_df[
                window_omnibus_df["window_name"] == window_name
            ].copy()

    table_outputs = write_tables(
        config["out_folder"],
        target_metrics_df,
        group_summary_df,
        pairwise_df,
        omnibus_df,
        extra_tables=extra_tables,
    )

    audit_payload = {
        "config_path": str(config_path),
        "results_source": config.get("results_source"),
        "out_folder": config["out_folder"],
        "figure_title": config["figure_title"],
        "target_time_map_key": config["target_time_map_key"],
        "target_times_requested_min": config.get("target_times_min"),
        "target_times_used_min": [float(x) for x in target_times],
        "shared_time_axis_min": float(time_grid[0]),
        "shared_time_axis_max": float(time_grid[-1]),
        "common_depth_grid_points": int(len(depth_grid)),
        "common_time_grid_points": int(len(time_grid)),
        "map_keys": [spec["key"] for spec in map_specs],
        "stats_metrics": stats_metrics,
        "window_stats_metrics": window_stats_metrics,
        "stats_alpha": float(config["stats_alpha"]),
        "profile_fixed_roi_fill_alpha": float(config["profile_fixed_roi_fill_alpha"]),
        "profile_fixed_roi_z": float(config["profile_fixed_roi_z"]),
        "report_windows": config["report_windows"],
        "tracers": {
            tracer_name: {
                "label": tracer_labels[tracer_name],
                "sample_ids": [sample.sample_id for sample in tracer_samples],
                "sample_runs": [sample.run_display for sample in tracer_samples],
                "roi_folders": sorted({sample.roi_folder for sample in tracer_samples}),
            }
            for tracer_name, tracer_samples in samples_by_tracer.items()
        },
        "matched_times_by_sample": {
            sample.sample_id: {
                map_key: {
                    f"{target_time:.6f}": {
                        "matched_index": int(matched_indices[sample.sample_id][map_key][target_time]),
                        "actual_time_min": float(sample.times[matched_indices[sample.sample_id][map_key][target_time]]),
                    }
                    for target_time in target_times
                }
                for map_key in [spec["key"] for spec in map_specs]
            }
            for sample in all_samples
        },
        "sample_audit": {sample.sample_id: sample.audit_info for sample in all_samples},
        "local_effective_diffusivity_supplement": local_d_supplement_outputs.get("audit", {}) if local_d_supplement_outputs else {},
        "figure_outputs": figure_outputs,
        "table_outputs": table_outputs,
    }
    save_audit_report(config["out_folder"], audit_payload)

    print("Saved map comparison outputs under:", str(Path(config["out_folder"])))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
