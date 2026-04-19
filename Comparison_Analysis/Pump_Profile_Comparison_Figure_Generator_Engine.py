import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "comparison" / "pump_profile_comparison_gad_n3.json"

FIGURE_TITLE_DEFAULT = "Pump Profile Comparison"
TIME_UNIT_LABEL_DEFAULT = "min"
CONCENTRATION_LABEL_DEFAULT = "Concentration (mg/mL)"
PUMP_PROFILE_ROOT_NAME = "pump_profile_comparison"
USE_NORMALIZED_DEPTH_DEFAULT = True
COMMON_DEPTH_POINTS_DEFAULT = 200
NORMALIZE_TO_PEAK_DEFAULT = True
NORMALIZED_MAP_VMIN_DEFAULT = 0.0
NORMALIZED_MAP_VMAX_DEFAULT = 1.0
DIFFERENCE_MAP_ABS_LIMIT_DEFAULT = 0.30
DEEP_REGION_START_DEFAULT = 0.67

SAMPLE_LINEWIDTH = 1.15
SAMPLE_LINE_ALPHA = 0.18
CENTER_LINEWIDTH = 2.4
GROUP_FILL_ALPHA = 0.10
MAP_TIME_LINE_ALPHA = 0.95

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

METRIC_LABELS = {
    "center_of_mass_depth": "Center-of-Mass Depth",
    "deep_tail_fraction": "Deep-Tail Fraction",
    "penetration_depth_10pct_peak": "Penetration Depth at 10% Peak",
}

DEFAULT_REPORT_WINDOWS = [
    {
        "name": "post_5min",
        "min_time_min": 5.0,
        "max_time_min": None,
    }
]


@dataclass
class SampleData:
    condition_name: str
    condition_label: str
    condition_color: str
    condition_marker: str
    tracer_name: str
    tracer_label: str
    sample_id: str
    roi_folder: str
    dx_mm: Optional[float]
    run_display: str
    selection_mode: str
    analysis_config_path: Optional[str]
    run_metadata_path: Optional[str]
    paths_used: Dict[str, str]
    time_min: np.ndarray
    profiles_raw: np.ndarray
    depth_raw: np.ndarray
    valid_rows: np.ndarray
    audit_info: Dict[str, Any]


@dataclass
class AlignedSampleData:
    sample: SampleData
    absolute_profiles: np.ndarray
    normalized_profiles: np.ndarray
    center_of_mass_depth: np.ndarray
    deep_tail_fraction: np.ndarray
    penetration_depth_10pct_peak: np.ndarray


@dataclass
class ConditionAggregate:
    name: str
    label: str
    color: str
    marker: str
    sample_count: int
    absolute_stack: np.ndarray
    normalized_stack: np.ndarray
    absolute_mean: np.ndarray
    absolute_std: np.ndarray
    normalized_mean: np.ndarray
    normalized_std: np.ndarray
    center_of_mass_mean: np.ndarray
    center_of_mass_std: np.ndarray
    deep_tail_mean: np.ndarray
    deep_tail_std: np.ndarray
    penetration_depth_mean: np.ndarray
    penetration_depth_std: np.ndarray
    aligned_samples: List[AlignedSampleData]


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
    for col in ["time", "time_plot", "time_min", "time_minutes", "time_seconds"]:
        if col in df.columns:
            return col
    raise KeyError(
        "Could not find a time column. Checked: time, time_plot, time_min, time_minutes, time_seconds."
        "\nAvailable columns:\n" + "\n".join(df.columns.tolist())
    )


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
    manual_named_rois = analysis_cfg.get("settings", {}).get("manual_named_rois")
    if isinstance(manual_named_rois, list) and len(manual_named_rois) == 1 and len(manual_named_rois[0]) >= 1:
        roi_name = manual_named_rois[0][0]
        if roi_name not in (None, ""):
            return str(roi_name)
    return None


def depth_spacing_from_run_metadata(run_metadata: Optional[Dict[str, Any]]) -> Optional[float]:
    if not run_metadata:
        return None
    dx_value = run_metadata.get("frame_geometry", {}).get("depth_spacing_mm")
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


def build_profile_rel_paths(run_rel: str, roi_folder: str) -> Dict[str, str]:
    run_rel = _normalize_rel_path(run_rel)
    roi_folder = str(roi_folder).strip("/").strip("\\")
    base = f"{run_rel}/{roi_folder}"
    return {
        "measured_profiles_csv": f"{base}/CSVs_Profiles/measured_profiles_depth_vs_time.csv",
        "fit_parameters_csv": f"{base}/CSVs_Summaries/fit_parameters_vs_time.csv",
        "run_metadata_json": f"{run_rel}/run_metadata.json",
    }


def build_profile_abs_paths(run_path: str, roi_folder: str) -> Dict[str, str]:
    run_dir = Path(run_path)
    roi_dir = run_dir / str(roi_folder)
    return {
        "measured_profiles_csv": str(roi_dir / "CSVs_Profiles" / "measured_profiles_depth_vs_time.csv"),
        "fit_parameters_csv": str(roi_dir / "CSVs_Summaries" / "fit_parameters_vs_time.csv"),
        "run_metadata_json": str(run_dir / "run_metadata.json"),
    }


def build_depth_axis(n_depth: int, use_normalized_depth: bool) -> np.ndarray:
    if use_normalized_depth:
        if n_depth <= 1:
            return np.array([0.0], dtype=float)
        return np.linspace(0.0, 1.0, n_depth)
    return np.arange(n_depth, dtype=float)


def interpolate_profiles_to_common_depth(
    profiles: np.ndarray,
    old_depth: np.ndarray,
    new_depth: np.ndarray,
) -> np.ndarray:
    out = np.full((profiles.shape[0], len(new_depth)), np.nan, dtype=float)
    for idx in range(profiles.shape[0]):
        row = np.asarray(profiles[idx], dtype=float)
        valid = np.isfinite(old_depth) & np.isfinite(row)
        if np.count_nonzero(valid) < 2:
            continue
        out[idx] = np.interp(new_depth, old_depth[valid], row[valid], left=np.nan, right=np.nan)
    return out


def interpolate_profiles_over_time(
    profiles: np.ndarray,
    old_time: np.ndarray,
    new_time: np.ndarray,
) -> np.ndarray:
    profiles = np.asarray(profiles, dtype=float)
    old_time = np.asarray(old_time, dtype=float)
    new_time = np.asarray(new_time, dtype=float)

    out = np.full((len(new_time), profiles.shape[1]), np.nan, dtype=float)
    valid_rows = np.isfinite(old_time) & np.any(np.isfinite(profiles), axis=1)
    if np.count_nonzero(valid_rows) < 2:
        return out

    time_valid = old_time[valid_rows]
    profiles_valid = profiles[valid_rows]
    order = np.argsort(time_valid)
    time_valid = time_valid[order]
    profiles_valid = profiles_valid[order]

    time_unique, unique_indices = np.unique(time_valid, return_index=True)
    profiles_unique = profiles_valid[unique_indices]
    if time_unique.size < 2:
        return out

    for depth_idx in range(profiles.shape[1]):
        values = profiles_unique[:, depth_idx]
        valid = np.isfinite(time_unique) & np.isfinite(values)
        if np.count_nonzero(valid) < 2:
            continue
        out[:, depth_idx] = np.interp(new_time, time_unique[valid], values[valid], left=np.nan, right=np.nan)
    return out


def normalize_profiles_to_peak(profiles: np.ndarray) -> np.ndarray:
    out = np.asarray(profiles, dtype=float).copy()
    for idx in range(out.shape[0]):
        row = out[idx]
        if not np.any(np.isfinite(row)):
            continue
        peak = float(np.nanmax(row))
        if np.isfinite(peak) and peak > 0:
            out[idx] = row / peak
    return out


def center_of_mass_depth(norm_profiles: np.ndarray, depth: np.ndarray) -> np.ndarray:
    com = np.full(norm_profiles.shape[0], np.nan, dtype=float)
    for idx in range(norm_profiles.shape[0]):
        profile = np.asarray(norm_profiles[idx], dtype=float)
        valid = np.isfinite(depth) & np.isfinite(profile)
        if np.count_nonzero(valid) < 2:
            continue
        values = profile[valid]
        coords = depth[valid]
        total = np.sum(values)
        if total > 0:
            com[idx] = np.sum(coords * values) / total
    return com


def deep_tail_fraction(norm_profiles: np.ndarray, depth: np.ndarray, deep_start: float) -> np.ndarray:
    frac = np.full(norm_profiles.shape[0], np.nan, dtype=float)
    mask = depth >= float(deep_start)
    if not np.any(mask):
        return frac
    for idx in range(norm_profiles.shape[0]):
        profile = np.asarray(norm_profiles[idx], dtype=float)
        if not np.any(np.isfinite(profile)):
            continue
        total = np.nansum(profile)
        deep = np.nansum(profile[mask])
        if total > 0:
            frac[idx] = deep / total
    return frac


def penetration_depth(norm_profiles: np.ndarray, depth: np.ndarray, threshold_fraction: float = 0.10) -> np.ndarray:
    pen = np.full(norm_profiles.shape[0], np.nan, dtype=float)
    for idx in range(norm_profiles.shape[0]):
        profile = np.asarray(norm_profiles[idx], dtype=float)
        valid = np.isfinite(depth) & np.isfinite(profile)
        if np.count_nonzero(valid) < 2:
            continue
        profile_valid = profile[valid]
        peak = np.nanmax(profile_valid)
        if not np.isfinite(peak) or peak <= 0:
            continue
        threshold = float(threshold_fraction) * peak
        passing = depth[valid][profile_valid >= threshold]
        if passing.size:
            pen[idx] = float(np.nanmax(passing))
    return pen


def build_panel_names(n_panels: int) -> List[str]:
    if n_panels == 3:
        return ["Early", "Mid", "Late"]
    return [f"Time {idx + 1}" for idx in range(n_panels)]


def select_target_time_indices(matched_times_min: np.ndarray, targets_min: Sequence[float]) -> List[int]:
    return [int(np.argmin(np.abs(matched_times_min - float(target)))) for target in targets_min]


def add_reference_time_lines(
    ax,
    matched_times_min: np.ndarray,
    reference_times_min: Sequence[float],
    color: str = "white",
) -> None:
    if matched_times_min.size == 0:
        return
    used: List[float] = []
    for target in reference_times_min:
        idx = int(np.argmin(np.abs(matched_times_min - float(target))))
        time_value = float(matched_times_min[idx])
        if any(abs(time_value - existing) < 1e-9 for existing in used):
            continue
        used.append(time_value)
        ax.axvline(
            time_value,
            color=color,
            linestyle="--",
            linewidth=1.3,
            alpha=MAP_TIME_LINE_ALPHA,
            zorder=5,
        )


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


def nanmean_and_std(stack: np.ndarray, axis: int) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.asarray(stack, dtype=float)
    valid = np.isfinite(stack)
    count = np.sum(valid, axis=axis)
    total = np.nansum(np.where(valid, stack, 0.0), axis=axis)
    mean = total / np.where(count > 0, count, np.nan)

    expanded_mean = np.expand_dims(mean, axis=axis)
    sq_diff = (stack - expanded_mean) ** 2
    valid_var = valid & np.isfinite(expanded_mean)
    var_denom = np.where(count > 1, count - 1, np.nan)
    var = np.nansum(np.where(valid_var, sq_diff, 0.0), axis=axis) / var_denom
    std = np.sqrt(var)
    return mean, std


def choose_target_times(time_grid: np.ndarray, manual_target_times: Optional[Sequence[float]]) -> List[float]:
    if manual_target_times is not None:
        if len(manual_target_times) == 0:
            raise ValueError("target_times_min cannot be empty.")
        return [float(value) for value in manual_target_times]
    if time_grid.size == 0:
        raise ValueError("Cannot auto-select target times from an empty common time grid.")
    start = float(time_grid[0])
    end = float(time_grid[-1])
    return [start, 0.5 * (start + end), end]


def safe_name(text: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(text).strip())
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "unnamed"


def build_common_depth_grid(
    samples: Sequence[SampleData],
    use_normalized_depth: bool,
    common_depth_points: int,
) -> np.ndarray:
    if common_depth_points < 2:
        raise ValueError("common_depth_points must be at least 2.")
    if use_normalized_depth:
        return np.linspace(0.0, 1.0, int(common_depth_points))

    max_shared_depth = min(float(np.nanmax(sample.depth_raw)) for sample in samples)
    if not np.isfinite(max_shared_depth) or max_shared_depth <= 0:
        raise ValueError("Could not build a valid common depth grid from the selected samples.")
    return np.linspace(0.0, max_shared_depth, int(common_depth_points))


def build_overlap_time_grid(
    samples: Sequence[SampleData],
    min_time_min: float,
    shared_time_axis_min: Optional[float],
    shared_time_axis_max: Optional[float],
) -> np.ndarray:
    min_times: List[float] = []
    max_times: List[float] = []
    dt_candidates: List[float] = []

    for sample in samples:
        valid_mask = sample.valid_rows & np.isfinite(sample.time_min) & (sample.time_min >= float(min_time_min))
        valid_times = sample.time_min[valid_mask]
        if valid_times.size == 0:
            raise ValueError(
                f"Sample '{sample.sample_id}' has no valid profile times after applying min_time_min={min_time_min}."
            )
        min_times.append(float(np.min(valid_times)))
        max_times.append(float(np.max(valid_times)))
        if valid_times.size > 1:
            diffs = np.diff(np.sort(np.unique(valid_times)))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                dt_candidates.append(float(np.min(diffs)))

    overlap_start = max(min_times)
    overlap_end = min(max_times)

    if shared_time_axis_min is not None:
        overlap_start = max(overlap_start, float(shared_time_axis_min))
    if shared_time_axis_max is not None:
        overlap_end = min(overlap_end, float(shared_time_axis_max))

    if overlap_end < overlap_start:
        raise ValueError(
            "The selected samples do not share a common valid time window after applying the current time limits."
        )

    dt = min(dt_candidates) if dt_candidates else 1.0
    point_count = int(np.floor((overlap_end - overlap_start) / dt)) + 1
    grid = overlap_start + np.arange(point_count, dtype=float) * dt
    if grid.size == 0 or grid[-1] < overlap_end - 0.25 * dt:
        grid = np.append(grid, overlap_end)
    return grid


def align_sample_to_common_grids(
    sample: SampleData,
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
    normalize_each_profile_to_own_peak: bool,
    deep_region_start: float,
    min_time_min: float,
) -> AlignedSampleData:
    keep_rows = sample.valid_rows & np.isfinite(sample.time_min) & (sample.time_min >= float(min_time_min))
    kept_profiles = sample.profiles_raw[keep_rows]
    kept_times = sample.time_min[keep_rows]

    absolute_depth_aligned = interpolate_profiles_to_common_depth(kept_profiles, sample.depth_raw, depth_grid)
    normalized_depth_aligned = (
        normalize_profiles_to_peak(absolute_depth_aligned)
        if normalize_each_profile_to_own_peak
        else absolute_depth_aligned.copy()
    )

    absolute_profiles = interpolate_profiles_over_time(absolute_depth_aligned, kept_times, time_grid)
    normalized_profiles = interpolate_profiles_over_time(normalized_depth_aligned, kept_times, time_grid)

    return AlignedSampleData(
        sample=sample,
        absolute_profiles=absolute_profiles,
        normalized_profiles=normalized_profiles,
        center_of_mass_depth=center_of_mass_depth(normalized_profiles, depth_grid),
        deep_tail_fraction=deep_tail_fraction(normalized_profiles, depth_grid, deep_region_start),
        penetration_depth_10pct_peak=penetration_depth(normalized_profiles, depth_grid, threshold_fraction=0.10),
    )


def aggregate_condition(condition_cfg: Dict[str, Any], aligned_samples: Sequence[AlignedSampleData]) -> ConditionAggregate:
    absolute_stack = np.stack([aligned.absolute_profiles for aligned in aligned_samples], axis=0)
    normalized_stack = np.stack([aligned.normalized_profiles for aligned in aligned_samples], axis=0)
    center_stack = np.stack([aligned.center_of_mass_depth for aligned in aligned_samples], axis=0)
    deep_stack = np.stack([aligned.deep_tail_fraction for aligned in aligned_samples], axis=0)
    penetration_stack = np.stack([aligned.penetration_depth_10pct_peak for aligned in aligned_samples], axis=0)

    absolute_mean, absolute_std = nanmean_and_std(absolute_stack, axis=0)
    normalized_mean, normalized_std = nanmean_and_std(normalized_stack, axis=0)
    center_of_mass_mean, center_of_mass_std = nanmean_and_std(center_stack, axis=0)
    deep_tail_mean, deep_tail_std = nanmean_and_std(deep_stack, axis=0)
    penetration_depth_mean, penetration_depth_std = nanmean_and_std(penetration_stack, axis=0)

    return ConditionAggregate(
        name=condition_cfg["name"],
        label=condition_cfg["label"],
        color=condition_cfg["color"],
        marker=condition_cfg["marker"],
        sample_count=len(aligned_samples),
        absolute_stack=absolute_stack,
        normalized_stack=normalized_stack,
        absolute_mean=absolute_mean,
        absolute_std=absolute_std,
        normalized_mean=normalized_mean,
        normalized_std=normalized_std,
        center_of_mass_mean=center_of_mass_mean,
        center_of_mass_std=center_of_mass_std,
        deep_tail_mean=deep_tail_mean,
        deep_tail_std=deep_tail_std,
        penetration_depth_mean=penetration_depth_mean,
        penetration_depth_std=penetration_depth_std,
        aligned_samples=list(aligned_samples),
    )


def validate_and_resolve_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    if "conditions" not in config or not config["conditions"]:
        raise ValueError("Pump profile comparison config must define a non-empty 'conditions' list.")
    if len(config["conditions"]) < 2:
        raise ValueError("Pump profile comparison config must define at least two conditions.")

    resolved: Dict[str, Any] = {}
    resolved["description"] = config.get("description")
    resolved["notes"] = config.get("notes", [])
    resolved["results_source"] = _resolve_path_from_config(config.get("results_source"), config_path)
    resolved["out_folder"] = _resolve_path_from_config(config.get("out_folder"), config_path)
    if resolved["out_folder"] in (None, ""):
        raise ValueError("Pump profile comparison config must define a non-empty 'out_folder'.")

    resolved["figure_title"] = config.get("figure_title", FIGURE_TITLE_DEFAULT)
    resolved["tracer_name"] = config.get("tracer_name", "tracer")
    resolved["tracer_label"] = config.get("tracer_label", resolved["tracer_name"])
    resolved["time_unit_label"] = config.get("time_unit_label", TIME_UNIT_LABEL_DEFAULT)
    resolved["concentration_label"] = config.get("concentration_label", CONCENTRATION_LABEL_DEFAULT)
    resolved["use_normalized_depth"] = bool(config.get("use_normalized_depth", USE_NORMALIZED_DEPTH_DEFAULT))
    resolved["common_depth_points"] = int(config.get("common_depth_points", COMMON_DEPTH_POINTS_DEFAULT))
    resolved["normalize_each_profile_to_own_peak"] = bool(
        config.get("normalize_each_profile_to_own_peak", NORMALIZE_TO_PEAK_DEFAULT)
    )
    resolved["min_time_min"] = float(config.get("min_time_min", 0.0))
    resolved["shared_time_axis_min"] = (
        float(config["shared_time_axis_min"]) if config.get("shared_time_axis_min") is not None else None
    )
    resolved["shared_time_axis_max"] = (
        float(config["shared_time_axis_max"]) if config.get("shared_time_axis_max") is not None else None
    )
    resolved["target_times_min"] = config.get("target_times_min")
    resolved["normalized_map_vmin"] = float(config.get("normalized_map_vmin", NORMALIZED_MAP_VMIN_DEFAULT))
    resolved["normalized_map_vmax"] = float(config.get("normalized_map_vmax", NORMALIZED_MAP_VMAX_DEFAULT))
    resolved["difference_map_abs_limit"] = float(config.get("difference_map_abs_limit", DIFFERENCE_MAP_ABS_LIMIT_DEFAULT))
    resolved["deep_region_start"] = float(config.get("deep_region_start", DEEP_REGION_START_DEFAULT))
    resolved["report_windows"] = config.get("report_windows", DEFAULT_REPORT_WINDOWS)
    resolved["stats_alpha"] = float(config.get("stats_alpha", 0.05))
    resolved["roi_folder"] = config.get("roi_folder")
    resolved["dx_mm"] = float(config["dx_mm"]) if config.get("dx_mm") not in (None, "") else None

    conditions_resolved: List[Dict[str, Any]] = []
    for condition_index, condition_cfg in enumerate(config["conditions"]):
        condition_name = condition_cfg.get("name")
        if condition_name in (None, ""):
            raise ValueError("Each condition entry needs a non-empty 'name'.")

        samples_cfg = condition_cfg.get("samples", [])
        if not samples_cfg:
            raise ValueError(f"Condition '{condition_name}' must define at least one sample.")

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
                    f"Each sample for condition '{condition_name}' must define one of: run_rel, run_path, analysis_config."
                )
            samples_resolved.append(sample_entry)

        conditions_resolved.append(
            {
                "name": condition_name,
                "label": condition_cfg.get("label", condition_name),
                "color": condition_cfg.get("color", DEFAULT_COLORS[condition_index % len(DEFAULT_COLORS)]),
                "marker": condition_cfg.get("marker", MARKERS[condition_index % len(MARKERS)]),
                "roi_folder": condition_cfg.get("roi_folder", resolved["roi_folder"]),
                "dx_mm": (
                    float(condition_cfg["dx_mm"])
                    if condition_cfg.get("dx_mm") not in (None, "")
                    else resolved["dx_mm"]
                ),
                "samples": samples_resolved,
            }
        )

    resolved["conditions"] = conditions_resolved
    return resolved


def load_sample_data(
    tracer_name: str,
    tracer_label: str,
    condition_cfg: Dict[str, Any],
    sample_cfg: Dict[str, Any],
    results_source: Optional[str],
    use_normalized_depth: bool,
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
        raise ValueError(
            f"Sample in condition '{condition_cfg['name']}' must define run_rel, run_path, or analysis_config."
        )

    roi_folder = (
        sample_cfg.get("roi_folder")
        or condition_cfg.get("roi_folder")
        or first_roi_name_from_run_metadata(run_metadata_doc)
        or first_roi_name_from_analysis_config(analysis_config_doc)
    )
    if roi_folder in (None, ""):
        raise ValueError(
            f"Could not infer roi_folder for sample '{sample_cfg.get('sample_id', run_display)}'. "
            "Set roi_folder in the comparison config or use an analysis run with exactly one selected ROI."
        )

    if selection_mode in {"analysis_config", "run_path"}:
        paths_used = build_profile_abs_paths(resolved_run, str(roi_folder))
        loader_required = load_csv_required_path
    else:
        paths_used = build_profile_rel_paths(resolved_run, str(roi_folder))
        loader_required = lambda relative_path: load_csv_required_any(results_source, relative_path)

    measured_df = loader_required(paths_used["measured_profiles_csv"])
    params_df = loader_required(paths_used["fit_parameters_csv"])

    measured_profiles = measured_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    time_col = find_time_column(params_df)
    time_values = pd.to_numeric(params_df[time_col], errors="coerce").to_numpy(dtype=float)
    time_min = time_values / 60.0 if time_col == "time_seconds" else time_values

    row_count = min(len(time_min), measured_profiles.shape[0])
    time_min = time_min[:row_count]
    measured_profiles = measured_profiles[:row_count, :]
    valid_rows = np.isfinite(time_min) & np.any(np.isfinite(measured_profiles), axis=1)
    depth_raw = build_depth_axis(measured_profiles.shape[1], use_normalized_depth=use_normalized_depth)

    dx_value = sample_cfg.get("dx_mm")
    if dx_value in (None, ""):
        dx_value = condition_cfg.get("dx_mm")
    if dx_value in (None, ""):
        dx_value = depth_spacing_from_run_metadata(run_metadata_doc)
    dx_mm = None if dx_value in (None, "") else float(dx_value)

    sample_id = sample_cfg.get("sample_id")
    if sample_id in (None, ""):
        sample_id = Path(run_display).name

    audit_info = {
        "analysis_config_path": analysis_config_path,
        "run_metadata_path": run_metadata_path,
        "paths_used": paths_used,
        "time_column": time_col,
        "roi_folder": str(roi_folder),
        "roi_inferred_from_run_metadata": first_roi_name_from_run_metadata(run_metadata_doc),
        "roi_inferred_from_analysis_config": first_roi_name_from_analysis_config(analysis_config_doc),
        "dx_mm_from_config_or_metadata": dx_mm,
        "selection_mode": selection_mode,
        "row_count_loaded": int(row_count),
        "valid_row_count": int(np.count_nonzero(valid_rows)),
    }

    return SampleData(
        condition_name=condition_cfg["name"],
        condition_label=condition_cfg["label"],
        condition_color=condition_cfg["color"],
        condition_marker=condition_cfg["marker"],
        tracer_name=tracer_name,
        tracer_label=tracer_label,
        sample_id=str(sample_id),
        roi_folder=str(roi_folder),
        dx_mm=dx_mm,
        run_display=run_display,
        selection_mode=selection_mode,
        analysis_config_path=analysis_config_path,
        run_metadata_path=run_metadata_path,
        paths_used=paths_used,
        time_min=time_min,
        profiles_raw=measured_profiles,
        depth_raw=depth_raw,
        valid_rows=valid_rows,
        audit_info=audit_info,
    )


def build_depth_label(use_normalized_depth: bool) -> str:
    return "Normalized depth (0=top, 1=bottom)" if use_normalized_depth else "Depth index"


def plot_condition_sample_lines(
    ax,
    aggregate: ConditionAggregate,
    depth_common: np.ndarray,
    time_idx: int,
    normalized: bool,
) -> None:
    for aligned in aggregate.aligned_samples:
        values = aligned.normalized_profiles[time_idx] if normalized else aligned.absolute_profiles[time_idx]
        ax.plot(
            values,
            depth_common,
            color=aggregate.color,
            linewidth=SAMPLE_LINEWIDTH,
            alpha=SAMPLE_LINE_ALPHA,
        )


def make_main_4panel_figure(
    figure_title: str,
    matched_times_min: np.ndarray,
    depth_common: np.ndarray,
    condition_a: ConditionAggregate,
    condition_b: ConditionAggregate,
    diff_norm: np.ndarray,
    profile_time_idxs: Sequence[int],
    out_path: Path,
    use_normalized_depth: bool,
    norm_map_vmin: float,
    norm_map_vmax: float,
    diff_map_abs_lim: float,
    target_times_min: Sequence[float],
) -> None:
    fig = plt.figure(figsize=(15.5, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.34)

    y_label = build_depth_label(use_normalized_depth)
    x_label = "Time (min)"

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        condition_a.normalized_mean.T,
        aspect="auto",
        origin="upper",
        extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
        vmin=norm_map_vmin,
        vmax=norm_map_vmax,
    )
    ax1.set_title(f"A. {condition_a.label.upper()}: MEAN NORMALIZED MEASURED MAP (n={condition_a.sample_count})")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    add_reference_time_lines(ax1, matched_times_min, target_times_min, color="white")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("Normalized concentration")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        condition_b.normalized_mean.T,
        aspect="auto",
        origin="upper",
        extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
        vmin=norm_map_vmin,
        vmax=norm_map_vmax,
    )
    ax2.set_title(f"B. {condition_b.label.upper()}: MEAN NORMALIZED MEASURED MAP (n={condition_b.sample_count})")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    add_reference_time_lines(ax2, matched_times_min, target_times_min, color="white")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Normalized concentration")

    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(
        diff_norm.T,
        aspect="auto",
        origin="upper",
        extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
        vmin=-diff_map_abs_lim,
        vmax=diff_map_abs_lim,
        cmap="coolwarm",
    )
    ax3.set_title(f"C. NORMALIZED DIFFERENCE MAP ({condition_b.label.upper()} - {condition_a.label.upper()})")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    add_reference_time_lines(ax3, matched_times_min, target_times_min, color="white")
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04).set_label("Delta normalized concentration")

    subgs = gs[1, 1].subgridspec(1, len(profile_time_idxs), wspace=0.52)
    panel_names = build_panel_names(len(profile_time_idxs))
    for panel_idx, time_idx in enumerate(profile_time_idxs):
        ax = fig.add_subplot(subgs[0, panel_idx])
        actual_time = float(matched_times_min[time_idx])
        plot_condition_sample_lines(ax, condition_a, depth_common, time_idx, normalized=True)
        plot_condition_sample_lines(ax, condition_b, depth_common, time_idx, normalized=True)
        ax.plot(
            condition_a.normalized_mean[time_idx],
            depth_common,
            color=condition_a.color,
            linewidth=CENTER_LINEWIDTH,
            label=f"{condition_a.label} mean (n={condition_a.sample_count})",
        )
        ax.plot(
            condition_b.normalized_mean[time_idx],
            depth_common,
            color=condition_b.color,
            linewidth=CENTER_LINEWIDTH,
            label=f"{condition_b.label} mean (n={condition_b.sample_count})",
        )
        ax.set_ylim(depth_common.max(), depth_common.min())
        ax.set_title(f"D{panel_idx + 1}. {panel_names[panel_idx]}\n{actual_time:.2f} min")
        ax.set_xlabel("Normalized concentration")
        if panel_idx == 0:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if panel_idx == len(profile_time_idxs) - 1:
            ax.legend(loc="lower right", frameon=False)

    fig.suptitle(f"{figure_title}\nMeasured Profile Comparison: {condition_a.label} vs {condition_b.label}", fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.90)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metric_with_group_band(
    ax,
    time_grid: np.ndarray,
    aggregate: ConditionAggregate,
    metric_name: str,
) -> None:
    if metric_name == "center_of_mass_depth":
        sample_series = [aligned.center_of_mass_depth for aligned in aggregate.aligned_samples]
        mean_series = aggregate.center_of_mass_mean
        std_series = aggregate.center_of_mass_std
    elif metric_name == "deep_tail_fraction":
        sample_series = [aligned.deep_tail_fraction for aligned in aggregate.aligned_samples]
        mean_series = aggregate.deep_tail_mean
        std_series = aggregate.deep_tail_std
    else:
        sample_series = [aligned.penetration_depth_10pct_peak for aligned in aggregate.aligned_samples]
        mean_series = aggregate.penetration_depth_mean
        std_series = aggregate.penetration_depth_std

    for series in sample_series:
        ax.plot(time_grid, series, color=aggregate.color, linewidth=SAMPLE_LINEWIDTH, alpha=SAMPLE_LINE_ALPHA)

    if aggregate.sample_count > 1:
        ax.fill_between(
            time_grid,
            mean_series - std_series,
            mean_series + std_series,
            color=aggregate.color,
            alpha=GROUP_FILL_ALPHA,
        )

    ax.plot(
        time_grid,
        mean_series,
        color=aggregate.color,
        linewidth=CENTER_LINEWIDTH,
        marker=aggregate.marker,
        markersize=4.0,
        markevery=max(1, len(time_grid) // 12),
        label=f"{aggregate.label} mean (n={aggregate.sample_count})",
    )


def make_metrics_figure(
    figure_title: str,
    time_grid: np.ndarray,
    condition_a: ConditionAggregate,
    condition_b: ConditionAggregate,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18.6, 5.4))

    plot_metric_with_group_band(axes[0], time_grid, condition_a, "center_of_mass_depth")
    plot_metric_with_group_band(axes[0], time_grid, condition_b, "center_of_mass_depth")
    axes[0].set_title("Center-of-Mass Depth vs Time")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Normalized depth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    plot_metric_with_group_band(axes[1], time_grid, condition_a, "deep_tail_fraction")
    plot_metric_with_group_band(axes[1], time_grid, condition_b, "deep_tail_fraction")
    axes[1].set_title("Deep-Tail Fraction vs Time")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Fraction of profile in deep region")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    plot_metric_with_group_band(axes[2], time_grid, condition_a, "penetration_depth_10pct_peak")
    plot_metric_with_group_band(axes[2], time_grid, condition_b, "penetration_depth_10pct_peak")
    axes[2].set_title("Penetration Depth at 10% Peak vs Time")
    axes[2].set_xlabel("Time (min)")
    axes[2].set_ylabel("Normalized penetration depth")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(frameon=False)

    fig.suptitle(f"{figure_title}\nShape-Based Metrics from Normalized Measured Profiles", fontsize=14.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_supplementary_absolute_figure(
    figure_title: str,
    matched_times_min: np.ndarray,
    depth_common: np.ndarray,
    condition_a: ConditionAggregate,
    condition_b: ConditionAggregate,
    profile_time_idxs: Sequence[int],
    out_path: Path,
    use_normalized_depth: bool,
    concentration_label: str,
    target_times_min: Sequence[float],
) -> None:
    fig = plt.figure(figsize=(15.5, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.34)

    y_label = build_depth_label(use_normalized_depth)
    vmax = float(np.nanmax([np.nanmax(condition_a.absolute_mean), np.nanmax(condition_b.absolute_mean)]))
    vmin = 0.0

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(
        condition_a.absolute_mean.T,
        aspect="auto",
        origin="upper",
        extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
        vmin=vmin,
        vmax=vmax,
    )
    ax1.set_title(f"{condition_a.label.upper()}: MEAN ABSOLUTE MEASURED MAP (n={condition_a.sample_count})")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel(y_label)
    add_reference_time_lines(ax1, matched_times_min, target_times_min, color="white")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label(concentration_label)

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(
        condition_b.absolute_mean.T,
        aspect="auto",
        origin="upper",
        extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
        vmin=vmin,
        vmax=vmax,
    )
    ax2.set_title(f"{condition_b.label.upper()}: MEAN ABSOLUTE MEASURED MAP (n={condition_b.sample_count})")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel(y_label)
    add_reference_time_lines(ax2, matched_times_min, target_times_min, color="white")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label(concentration_label)

    subgs = gs[1, :].subgridspec(1, len(profile_time_idxs), wspace=0.46)
    panel_names = build_panel_names(len(profile_time_idxs))
    for panel_idx, time_idx in enumerate(profile_time_idxs):
        ax = fig.add_subplot(subgs[0, panel_idx])
        actual_time = float(matched_times_min[time_idx])
        plot_condition_sample_lines(ax, condition_a, depth_common, time_idx, normalized=False)
        plot_condition_sample_lines(ax, condition_b, depth_common, time_idx, normalized=False)
        ax.plot(
            condition_a.absolute_mean[time_idx],
            depth_common,
            color=condition_a.color,
            linewidth=CENTER_LINEWIDTH,
            label=f"{condition_a.label} mean (n={condition_a.sample_count})",
        )
        ax.plot(
            condition_b.absolute_mean[time_idx],
            depth_common,
            color=condition_b.color,
            linewidth=CENTER_LINEWIDTH,
            label=f"{condition_b.label} mean (n={condition_b.sample_count})",
        )
        ax.set_ylim(depth_common.max(), depth_common.min())
        ax.set_title(f"{panel_names[panel_idx]} Absolute Profiles\n{actual_time:.2f} min")
        ax.set_xlabel(concentration_label)
        if panel_idx == 0:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if panel_idx == len(profile_time_idxs) - 1:
            ax.legend(loc="lower right", frameon=False)

    fig.suptitle(f"{figure_title}\nSupplementary Absolute Measured-Profile Comparison", fontsize=16, y=0.99)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_per_sample_all_time_metrics(
    aligned_samples: Sequence[AlignedSampleData],
    time_grid: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for aligned in aligned_samples:
        sample = aligned.sample
        for idx, time_value in enumerate(time_grid):
            rows.append(
                {
                    "tracer_name": sample.tracer_name,
                    "tracer_label": sample.tracer_label,
                    "condition_name": sample.condition_name,
                    "condition_label": sample.condition_label,
                    "sample_id": sample.sample_id,
                    "actual_time_min": float(time_value),
                    "run_display": sample.run_display,
                    "roi_folder": sample.roi_folder,
                    "dx_mm": sample.dx_mm,
                    "center_of_mass_depth": aligned.center_of_mass_depth[idx],
                    "deep_tail_fraction": aligned.deep_tail_fraction[idx],
                    "penetration_depth_10pct_peak": aligned.penetration_depth_10pct_peak[idx],
                }
            )
    return pd.DataFrame(rows)


def build_condition_group_all_time_metrics(
    aggregates: Sequence[ConditionAggregate],
    time_grid: np.ndarray,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for aggregate in aggregates:
        for idx, time_value in enumerate(time_grid):
            rows.append(
                {
                    "condition_name": aggregate.name,
                    "condition_label": aggregate.label,
                    "actual_time_min": float(time_value),
                    "sample_count": aggregate.sample_count,
                    "center_of_mass_depth_mean": aggregate.center_of_mass_mean[idx],
                    "center_of_mass_depth_std": aggregate.center_of_mass_std[idx],
                    "center_of_mass_depth_sem": (
                        aggregate.center_of_mass_std[idx] / np.sqrt(aggregate.sample_count)
                        if aggregate.sample_count > 1 and np.isfinite(aggregate.center_of_mass_std[idx])
                        else float("nan")
                    ),
                    "deep_tail_fraction_mean": aggregate.deep_tail_mean[idx],
                    "deep_tail_fraction_std": aggregate.deep_tail_std[idx],
                    "deep_tail_fraction_sem": (
                        aggregate.deep_tail_std[idx] / np.sqrt(aggregate.sample_count)
                        if aggregate.sample_count > 1 and np.isfinite(aggregate.deep_tail_std[idx])
                        else float("nan")
                    ),
                    "penetration_depth_10pct_peak_mean": aggregate.penetration_depth_mean[idx],
                    "penetration_depth_10pct_peak_std": aggregate.penetration_depth_std[idx],
                    "penetration_depth_10pct_peak_sem": (
                        aggregate.penetration_depth_std[idx] / np.sqrt(aggregate.sample_count)
                        if aggregate.sample_count > 1 and np.isfinite(aggregate.penetration_depth_std[idx])
                        else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def compute_standardized_effect_sizes(values_a: np.ndarray, values_b: np.ndarray) -> Tuple[float, float]:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if values_a.size < 2 or values_b.size < 2:
        return float("nan"), float("nan")

    sd_a = float(np.std(values_a, ddof=1))
    sd_b = float(np.std(values_b, ddof=1))
    if not np.isfinite(sd_a) or not np.isfinite(sd_b):
        return float("nan"), float("nan")

    pooled_var_num = (values_a.size - 1) * (sd_a ** 2) + (values_b.size - 1) * (sd_b ** 2)
    pooled_var_den = values_a.size + values_b.size - 2
    if pooled_var_den <= 0:
        return float("nan"), float("nan")

    pooled_sd = float(np.sqrt(pooled_var_num / pooled_var_den))
    if not np.isfinite(pooled_sd) or pooled_sd <= 0:
        return float("nan"), float("nan")

    cohens_d = float((np.mean(values_a) - np.mean(values_b)) / pooled_sd)
    total_n = values_a.size + values_b.size
    if total_n <= 3:
        return cohens_d, float("nan")

    correction = 1.0 - (3.0 / (4.0 * total_n - 9.0))
    hedges_g = float(correction * cohens_d)
    return cohens_d, hedges_g


def build_window_sample_summary(
    all_time_metrics_df: pd.DataFrame,
    report_windows: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for window in report_windows:
        window_name = str(window["name"])
        min_time = float(window["min_time_min"]) if window.get("min_time_min") is not None else None
        max_time = float(window["max_time_min"]) if window.get("max_time_min") is not None else None

        window_df = all_time_metrics_df.copy()
        if min_time is not None:
            window_df = window_df[window_df["actual_time_min"] >= min_time]
        if max_time is not None:
            window_df = window_df[window_df["actual_time_min"] <= max_time]

        for (
            condition_name,
            condition_label,
            sample_id,
        ), sample_df in window_df.groupby(["condition_name", "condition_label", "sample_id"], dropna=False):
            run_display = sample_df["run_display"].iloc[0]
            roi_folder = sample_df["roi_folder"].iloc[0]
            tracer_name = sample_df["tracer_name"].iloc[0]
            tracer_label = sample_df["tracer_label"].iloc[0]
            for metric in metrics:
                if metric not in sample_df.columns:
                    continue
                stats_map = finite_stats(sample_df[metric])
                rows.append(
                    {
                        "window_name": window_name,
                        "window_min_time_min": min_time,
                        "window_max_time_min": max_time,
                        "tracer_name": tracer_name,
                        "tracer_label": tracer_label,
                        "condition_name": condition_name,
                        "condition_label": condition_label,
                        "sample_id": sample_id,
                        "run_display": run_display,
                        "roi_folder": roi_folder,
                        "metric_name": metric,
                        "metric_label": METRIC_LABELS.get(metric, metric),
                        "timepoint_count": int(stats_map["n"]),
                        "mean": stats_map["mean"],
                        "std": stats_map["std"],
                        "median": stats_map["median"],
                        "iqr": stats_map["iqr"],
                        "min": stats_map["min"],
                        "max": stats_map["max"],
                    }
                )
    return pd.DataFrame(rows)


def build_window_group_summary(window_sample_summary_df: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "metric_name",
        "metric_label",
        "condition_name",
        "condition_label",
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
        "metric_name",
        "metric_label",
        "condition_name",
        "condition_label",
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


def build_window_pairwise_statistics(
    window_sample_summary_df: pd.DataFrame,
    alpha: float,
) -> pd.DataFrame:
    columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "metric_name",
        "metric_label",
        "condition_a",
        "condition_a_label",
        "condition_b",
        "condition_b_label",
        "n_a",
        "n_b",
        "mean_a",
        "mean_b",
        "mean_diff_a_minus_b",
        "relative_change_b_vs_a_percent",
        "cohens_d_a_minus_b",
        "hedges_g_a_minus_b",
        "statistic",
        "p_value",
        "alpha",
        "significant",
        "note",
    ]
    rows: List[Dict[str, Any]] = []
    group_cols = ["window_name", "window_min_time_min", "window_max_time_min", "metric_name", "metric_label"]
    for group_values, metric_df in window_sample_summary_df.groupby(group_cols, dropna=False):
        condition_groups: Dict[str, np.ndarray] = {}
        condition_labels: Dict[str, str] = {}
        for condition_name, condition_df in metric_df.groupby("condition_name"):
            values = pd.to_numeric(condition_df["mean"], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size:
                condition_groups[condition_name] = values
                condition_labels[condition_name] = condition_df["condition_label"].iloc[0]

        for condition_a, condition_b in combinations(sorted(condition_groups.keys()), 2):
            values_a = condition_groups[condition_a]
            values_b = condition_groups[condition_b]
            note = ""
            statistic = float("nan")
            p_value = float("nan")
            significant = False
            cohens_d = float("nan")
            hedges_g = float("nan")
            if values_a.size >= 2 and values_b.size >= 2:
                statistic, p_value = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
                significant = bool(np.isfinite(p_value) and p_value < alpha)
                cohens_d, hedges_g = compute_standardized_effect_sizes(values_a, values_b)
            else:
                note = "Need n>=2 in each condition for Welch t-test."

            mean_a = float(np.mean(values_a)) if values_a.size else float("nan")
            mean_b = float(np.mean(values_b)) if values_b.size else float("nan")
            if np.isfinite(mean_a) and mean_a != 0 and np.isfinite(mean_b):
                relative_change_percent = float(100.0 * ((mean_b - mean_a) / mean_a))
            else:
                relative_change_percent = float("nan")

            window_name, window_min_time_min, window_max_time_min, metric_name, metric_label = group_values
            rows.append(
                {
                    "window_name": window_name,
                    "window_min_time_min": window_min_time_min,
                    "window_max_time_min": window_max_time_min,
                    "metric_name": metric_name,
                    "metric_label": metric_label,
                    "condition_a": condition_a,
                    "condition_a_label": condition_labels[condition_a],
                    "condition_b": condition_b,
                    "condition_b_label": condition_labels[condition_b],
                    "n_a": int(values_a.size),
                    "n_b": int(values_b.size),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "mean_diff_a_minus_b": (
                        float(mean_a - mean_b)
                        if values_a.size and values_b.size
                        else float("nan")
                    ),
                    "relative_change_b_vs_a_percent": relative_change_percent,
                    "cohens_d_a_minus_b": cohens_d,
                    "hedges_g_a_minus_b": hedges_g,
                    "statistic": float(statistic) if np.isfinite(statistic) else float("nan"),
                    "p_value": float(p_value) if np.isfinite(p_value) else float("nan"),
                    "alpha": float(alpha),
                    "significant": significant,
                    "note": note,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def build_window_effect_size_summary(pairwise_stats_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "metric_name",
        "metric_label",
        "condition_a",
        "condition_b",
        "mean_a",
        "mean_b",
        "mean_diff_a_minus_b",
        "relative_change_b_vs_a_percent",
        "cohens_d_a_minus_b",
        "hedges_g_a_minus_b",
        "p_value",
        "significant",
        "note",
    ]
    if pairwise_stats_df.empty:
        return pd.DataFrame(columns=columns)
    return pairwise_stats_df.loc[:, columns].copy()


def write_tables(
    out_root: Path,
    per_sample_df: pd.DataFrame,
    group_df: pd.DataFrame,
    window_sample_summary_df: pd.DataFrame,
    window_group_summary_df: pd.DataFrame,
    pairwise_stats_df: pd.DataFrame,
    effect_size_summary_df: pd.DataFrame,
) -> List[str]:
    summary_dir = out_root / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []

    tables = {
        "per_sample_all_time_metrics.csv": per_sample_df,
        "condition_group_all_time_metrics.csv": group_df,
        "window_sample_metric_summary.csv": window_sample_summary_df,
        "window_condition_group_metric_summary.csv": window_group_summary_df,
        "window_pairwise_statistics.csv": pairwise_stats_df,
        "window_effect_size_summary.csv": effect_size_summary_df,
    }
    for filename, df in tables.items():
        out_path = summary_dir / filename
        df.to_csv(out_path, index=False)
        outputs.append(str(out_path))
    return outputs


def write_aligned_outputs(
    out_root: Path,
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
    target_times: Sequence[float],
    target_indices: Sequence[int],
    condition_a: ConditionAggregate,
    condition_b: ConditionAggregate,
    diff_norm: np.ndarray,
) -> List[str]:
    aligned_dir = out_root / "aligned_profiles"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[str] = []

    axis_tables = {
        "common_time_min.csv": pd.DataFrame({"time_min": time_grid}),
        "common_depth_axis.csv": pd.DataFrame({"depth_axis": depth_grid}),
        "target_times_selected.csv": pd.DataFrame(
            {
                "requested_time_min": [float(value) for value in target_times],
                "actual_time_min": [float(time_grid[idx]) for idx in target_indices],
                "time_index": [int(idx) for idx in target_indices],
            }
        ),
    }
    for filename, df in axis_tables.items():
        out_path = aligned_dir / filename
        df.to_csv(out_path, index=False)
        outputs.append(str(out_path))

    matrix_tables = {
        f"{safe_name(condition_a.name)}_mean_absolute_profiles.csv": pd.DataFrame(condition_a.absolute_mean),
        f"{safe_name(condition_a.name)}_mean_normalized_profiles.csv": pd.DataFrame(condition_a.normalized_mean),
        f"{safe_name(condition_b.name)}_mean_absolute_profiles.csv": pd.DataFrame(condition_b.absolute_mean),
        f"{safe_name(condition_b.name)}_mean_normalized_profiles.csv": pd.DataFrame(condition_b.normalized_mean),
        f"{safe_name(condition_b.name)}_minus_{safe_name(condition_a.name)}_normalized_difference_profiles.csv": pd.DataFrame(diff_norm),
    }
    for filename, df in matrix_tables.items():
        out_path = aligned_dir / filename
        df.to_csv(out_path, index=False)
        outputs.append(str(out_path))
    return outputs


def write_audit_json(
    out_root: Path,
    config_path: Path,
    config: Dict[str, Any],
    time_grid: np.ndarray,
    depth_grid: np.ndarray,
    conditions: Sequence[ConditionAggregate],
    all_samples: Sequence[SampleData],
) -> str:
    audit_path = out_root / "comparison_audit.json"
    audit_doc = {
        "config_path": str(config_path),
        "figure_title": config["figure_title"],
        "tracer_name": config["tracer_name"],
        "tracer_label": config["tracer_label"],
        "out_root": str(out_root),
        "condition_order": [condition.name for condition in conditions],
        "common_time_axis": time_grid.tolist(),
        "common_depth_axis": depth_grid.tolist(),
        "use_normalized_depth": bool(config["use_normalized_depth"]),
        "common_depth_points": int(config["common_depth_points"]),
        "normalize_each_profile_to_own_peak": bool(config["normalize_each_profile_to_own_peak"]),
        "min_time_min": float(config["min_time_min"]),
        "conditions": {
            condition.name: {
                "label": condition.label,
                "sample_count": condition.sample_count,
                "sample_ids": [aligned.sample.sample_id for aligned in condition.aligned_samples],
                "sample_runs": [aligned.sample.run_display for aligned in condition.aligned_samples],
                "roi_folders": sorted({aligned.sample.roi_folder for aligned in condition.aligned_samples}),
            }
            for condition in conditions
        },
        "sample_audit": {sample.sample_id: sample.audit_info for sample in all_samples},
    }
    audit_path.write_text(json.dumps(audit_doc, indent=2), encoding="utf-8")
    return str(audit_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare measured pump-off vs pump-on concentration profiles across multiple replicate samples."
    )
    parser.add_argument(
        "--config",
        required=True,
        help=(
            "Path to a JSON pump profile comparison config. "
            f"Example: {DEFAULT_CONFIG_PATH.relative_to(REPO_ROOT).as_posix()}"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = validate_and_resolve_config(load_config(config_path), config_path)

    samples_by_condition: Dict[str, List[SampleData]] = {}
    all_samples: List[SampleData] = []

    for condition_cfg in config["conditions"]:
        condition_samples = [
            load_sample_data(
                tracer_name=config["tracer_name"],
                tracer_label=config["tracer_label"],
                condition_cfg=condition_cfg,
                sample_cfg=sample_cfg,
                results_source=config.get("results_source"),
                use_normalized_depth=config["use_normalized_depth"],
            )
            for sample_cfg in condition_cfg["samples"]
        ]
        samples_by_condition[condition_cfg["name"]] = condition_samples
        all_samples.extend(condition_samples)

    depth_grid = build_common_depth_grid(
        all_samples,
        use_normalized_depth=config["use_normalized_depth"],
        common_depth_points=config["common_depth_points"],
    )
    time_grid = build_overlap_time_grid(
        all_samples,
        min_time_min=config["min_time_min"],
        shared_time_axis_min=config.get("shared_time_axis_min"),
        shared_time_axis_max=config.get("shared_time_axis_max"),
    )
    target_times = choose_target_times(time_grid, config.get("target_times_min"))
    target_indices = select_target_time_indices(time_grid, target_times)

    aggregates: List[ConditionAggregate] = []
    aligned_samples_all: List[AlignedSampleData] = []
    for condition_cfg in config["conditions"]:
        aligned_samples = [
            align_sample_to_common_grids(
                sample=sample,
                time_grid=time_grid,
                depth_grid=depth_grid,
                normalize_each_profile_to_own_peak=config["normalize_each_profile_to_own_peak"],
                deep_region_start=config["deep_region_start"],
                min_time_min=config["min_time_min"],
            )
            for sample in samples_by_condition[condition_cfg["name"]]
        ]
        aligned_samples_all.extend(aligned_samples)
        aggregates.append(aggregate_condition(condition_cfg, aligned_samples))

    if len(aggregates) < 2:
        raise ValueError("Need at least two aggregated conditions to build the pump profile comparison.")

    condition_a = aggregates[0]
    condition_b = aggregates[1]
    diff_norm = condition_b.normalized_mean - condition_a.normalized_mean

    out_root = Path(config["out_folder"]) / PUMP_PROFILE_ROOT_NAME
    out_root.mkdir(parents=True, exist_ok=True)

    make_main_4panel_figure(
        figure_title=config["figure_title"],
        matched_times_min=time_grid,
        depth_common=depth_grid,
        condition_a=condition_a,
        condition_b=condition_b,
        diff_norm=diff_norm,
        profile_time_idxs=target_indices,
        out_path=out_root / "main_4panel_normalized_measured_profile_comparison.png",
        use_normalized_depth=config["use_normalized_depth"],
        norm_map_vmin=config["normalized_map_vmin"],
        norm_map_vmax=config["normalized_map_vmax"],
        diff_map_abs_lim=config["difference_map_abs_limit"],
        target_times_min=target_times,
    )

    make_metrics_figure(
        figure_title=config["figure_title"],
        time_grid=time_grid,
        condition_a=condition_a,
        condition_b=condition_b,
        out_path=out_root / "normalized_profile_shape_metrics_vs_time.png",
    )

    make_supplementary_absolute_figure(
        figure_title=config["figure_title"],
        matched_times_min=time_grid,
        depth_common=depth_grid,
        condition_a=condition_a,
        condition_b=condition_b,
        profile_time_idxs=target_indices,
        out_path=out_root / "supplementary_absolute_measured_profile_comparison.png",
        use_normalized_depth=config["use_normalized_depth"],
        concentration_label=config["concentration_label"],
        target_times_min=target_times,
    )

    per_sample_df = build_per_sample_all_time_metrics(aligned_samples_all, time_grid)
    group_df = build_condition_group_all_time_metrics(aggregates, time_grid)
    metric_names = list(METRIC_LABELS.keys())
    window_sample_summary_df = build_window_sample_summary(per_sample_df, config["report_windows"], metric_names)
    window_group_summary_df = build_window_group_summary(window_sample_summary_df)
    pairwise_stats_df = build_window_pairwise_statistics(window_sample_summary_df, config["stats_alpha"])
    effect_size_summary_df = build_window_effect_size_summary(pairwise_stats_df)

    table_outputs = write_tables(
        out_root=out_root,
        per_sample_df=per_sample_df,
        group_df=group_df,
        window_sample_summary_df=window_sample_summary_df,
        window_group_summary_df=window_group_summary_df,
        pairwise_stats_df=pairwise_stats_df,
        effect_size_summary_df=effect_size_summary_df,
    )
    aligned_outputs = write_aligned_outputs(
        out_root=out_root,
        time_grid=time_grid,
        depth_grid=depth_grid,
        target_times=target_times,
        target_indices=target_indices,
        condition_a=condition_a,
        condition_b=condition_b,
        diff_norm=diff_norm,
    )
    audit_path = write_audit_json(
        out_root=out_root,
        config_path=config_path,
        config=config,
        time_grid=time_grid,
        depth_grid=depth_grid,
        conditions=aggregates,
        all_samples=all_samples,
    )

    print("Pump profile comparison complete.")
    print(f"Output directory: {out_root}")
    print(f"Summary tables written: {len(table_outputs)}")
    print(f"Aligned profile tables written: {len(aligned_outputs)}")
    print(f"Audit JSON: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
