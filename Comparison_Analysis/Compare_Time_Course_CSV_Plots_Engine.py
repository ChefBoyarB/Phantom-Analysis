import argparse
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

FIGURE_TITLE_DEFAULT = "Time Course Comparison"
TIME_UNIT_LABEL_DEFAULT = "min"
TIMECOURSE_ROOT_NAME = "time_course_comparisons"

Z95 = 1.96
UNC_FILL_ALPHA = 0.08
CENTER_LINEWIDTH = 2.3
SAMPLE_LINEWIDTH = 1.1
SAMPLE_LINE_ALPHA = 0.22
LINE_MARKER_SIZE = 4.5

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

DEFAULT_METRIC_YLIMS = {
    "effective_diffusivity": [0.0, 0.035],
    "effective_diffusivity_zoomed": [0.0, 0.005],
    "fitted_Cs": [15.0, 70.0],
    "profile_fit_r2": [0.0, 1.01],
}

DEFAULT_REPORT_WINDOWS = [
    {
        "name": "post_5min",
        "min_time_min": 5.0,
        "max_time_min": None,
    }
]

UNCERTAINTY_DEFINITIONS = {
    "fixed_roi": [
        "model_fit_uncertainty",
        "hu_noise_uncertainty",
        "calibration_uncertainty",
    ],
    "combined": [
        "model_fit_uncertainty",
        "hu_noise_uncertainty",
        "roi_sensitivity_uncertainty",
        "calibration_uncertainty",
    ],
}

TIMECOURSE_METRIC_SPECS = [
    {
        "key": "mean_concentration",
        "label": "Mean concentration (mg/mL)",
        "center_candidates": ["mean_conc", "mean_concentration"],
        "fixed_direct_std_candidates": [
            "mean_conc_fixed_roi_std",
            "mean_concentration_fixed_roi_std",
        ],
        "fixed_ci_low_candidates": [
            "mean_conc_fixed_roi_ci_low",
            "mean_concentration_fixed_roi_ci_low",
        ],
        "fixed_ci_high_candidates": [
            "mean_conc_fixed_roi_ci_high",
            "mean_concentration_fixed_roi_ci_high",
        ],
        "fixed_component_candidates": [
            ["mean_conc_hu_noise_std", "mean_concentration_hu_noise_std"],
            ["mean_conc_calibration_std", "mean_concentration_calibration_std"],
        ],
        "combined_direct_std_candidates": [
            "mean_conc_combined_std",
            "mean_concentration_combined_std",
        ],
        "combined_ci_low_candidates": [
            "mean_conc_combined_ci_low",
            "mean_concentration_combined_ci_low",
        ],
        "combined_ci_high_candidates": [
            "mean_conc_combined_ci_high",
            "mean_concentration_combined_ci_high",
        ],
        "combined_component_candidates": [
            ["mean_conc_hu_noise_std", "mean_concentration_hu_noise_std"],
            ["mean_conc_roi_sensitivity_std", "mean_concentration_roi_sensitivity_std"],
            ["mean_conc_calibration_std", "mean_concentration_calibration_std"],
        ],
    },
    {
        "key": "effective_diffusivity",
        "label": r"Fitted effective diffusivity (mm$^2$/s)",
        "center_candidates": ["effective_diffusivity", "effective_diffusivity_mm2_s"],
        "fixed_direct_std_candidates": [
            "effective_diffusivity_fixed_roi_std",
            "effective_diffusivity_fixed_roi_std_mm2_s",
        ],
        "fixed_ci_low_candidates": [
            "effective_diffusivity_fixed_roi_ci_low",
            "effective_diffusivity_fixed_roi_ci_low_mm2_s",
        ],
        "fixed_ci_high_candidates": [
            "effective_diffusivity_fixed_roi_ci_high",
            "effective_diffusivity_fixed_roi_ci_high_mm2_s",
        ],
        "fixed_component_candidates": [
            ["effective_diffusivity_std", "effective_diffusivity_std_mm2_s"],
            ["effective_diffusivity_hu_noise_std", "effective_diffusivity_hu_noise_std_mm2_s"],
            ["effective_diffusivity_calibration_std", "effective_diffusivity_calibration_std_mm2_s"],
        ],
        "combined_direct_std_candidates": [
            "effective_diffusivity_combined_std",
            "effective_diffusivity_combined_std_mm2_s",
        ],
        "combined_ci_low_candidates": [
            "effective_diffusivity_combined_ci_low",
            "effective_diffusivity_combined_ci_low_mm2_s",
        ],
        "combined_ci_high_candidates": [
            "effective_diffusivity_combined_ci_high",
            "effective_diffusivity_combined_ci_high_mm2_s",
        ],
        "combined_component_candidates": [
            ["effective_diffusivity_std", "effective_diffusivity_std_mm2_s"],
            ["effective_diffusivity_hu_noise_std", "effective_diffusivity_hu_noise_std_mm2_s"],
            ["effective_diffusivity_roi_sensitivity_std", "effective_diffusivity_roi_sensitivity_std_mm2_s"],
            ["effective_diffusivity_calibration_std", "effective_diffusivity_calibration_std_mm2_s"],
        ],
    },
    {
        "key": "fitted_Cs",
        "label": r"Effective fitted boundary concentration, C$_s$ (mg/mL)",
        "center_candidates": ["fitted_Cs"],
        "fixed_direct_std_candidates": ["fitted_Cs_fixed_roi_std"],
        "fixed_ci_low_candidates": ["fitted_Cs_fixed_roi_ci_low"],
        "fixed_ci_high_candidates": ["fitted_Cs_fixed_roi_ci_high"],
        "fixed_component_candidates": [
            ["fitted_Cs_std"],
            ["fitted_Cs_hu_noise_std"],
            ["fitted_Cs_calibration_std"],
        ],
        "combined_direct_std_candidates": ["fitted_Cs_combined_std"],
        "combined_ci_low_candidates": ["fitted_Cs_combined_ci_low"],
        "combined_ci_high_candidates": ["fitted_Cs_combined_ci_high"],
        "combined_component_candidates": [
            ["fitted_Cs_std"],
            ["fitted_Cs_hu_noise_std"],
            ["fitted_Cs_roi_sensitivity_std"],
            ["fitted_Cs_calibration_std"],
        ],
    },
    {
        "key": "profile_fit_rmse",
        "label": "Profile-fit RMSE (mg/mL)",
        "center_candidates": ["profile_fit_rmse"],
    },
    {
        "key": "profile_fit_r2",
        "label": r"Profile-fit $R^2$",
        "center_candidates": ["profile_fit_r2"],
    },
    {
        "key": "mean_diffusive_flux_magnitude",
        "label": r"Mean $|J_{diff}|$",
        "center_candidates": ["mean_diffusive_flux_magnitude"],
    },
    {
        "key": "regularized_effective_diffusivity",
        "label": r"Temporally regularized effective diffusivity (mm$^2$/s)",
        "center_candidates": [
            "regularized_effective_diffusivity_plot_mm2_s",
            "regularized_effective_diffusivity_mm2_s",
        ],
    },
]

METRIC_SPEC_BY_KEY = {spec["key"]: spec for spec in TIMECOURSE_METRIC_SPECS}

BAND_PLOT_SPECS = [
    {
        "metric_key": "mean_concentration",
        "title": "Mean concentration vs time",
        "out_stem": "mean_concentration",
        "ylim_key": "mean_concentration",
    },
    {
        "metric_key": "effective_diffusivity",
        "title": "Effective diffusivity vs time",
        "out_stem": "effective_diffusivity",
        "ylim_key": "effective_diffusivity",
    },
    {
        "metric_key": "effective_diffusivity",
        "title": "Effective diffusivity vs time (zoomed)",
        "out_stem": "effective_diffusivity_zoomed",
        "ylim_key": "effective_diffusivity_zoomed",
    },
    {
        "metric_key": "fitted_Cs",
        "title": "Effective fitted boundary concentration vs time",
        "out_stem": "fitted_Cs",
        "ylim_key": "fitted_Cs",
    },
]

DIAGNOSTIC_PLOT_SPECS = [
    {
        "metric_key": "profile_fit_rmse",
        "title": "Profile-fit RMSE vs time",
        "out_stem": "profile_fit_rmse",
        "ylim_key": "profile_fit_rmse",
    },
    {
        "metric_key": "profile_fit_r2",
        "title": r"Profile-fit $R^2$ vs time",
        "out_stem": "profile_fit_r2",
        "ylim_key": "profile_fit_r2",
    },
    {
        "metric_key": "mean_diffusive_flux_magnitude",
        "title": "Mean diffusive flux magnitude vs time",
        "out_stem": "mean_diffusive_flux_magnitude",
        "ylim_key": "mean_diffusive_flux_magnitude",
    },
    {
        "metric_key": "regularized_effective_diffusivity",
        "title": "Temporally regularized effective diffusivity vs time",
        "out_stem": "regularized_effective_diffusivity",
        "ylim_key": "regularized_effective_diffusivity",
    },
]

BAND_MODE_SPECS = [
    ("combined_95ci", "combined_95CI", "combined_95CI"),
    ("combined_1sd", "combined_1SD", "combined_1SD"),
    ("fixed_roi_95ci", "fixedROI_95CI", "fixedROI_95CI"),
    ("fixed_roi_1sd", "fixedROI_1SD", "fixedROI_1SD"),
]


@dataclass
class TimedSeries:
    time: np.ndarray
    values: np.ndarray
    source_csv: str
    source_column: str


@dataclass
class MetricSeries:
    key: str
    label: str
    center: Optional[TimedSeries]
    fixed_std: Optional[TimedSeries]
    fixed_ci_low: Optional[TimedSeries]
    fixed_ci_high: Optional[TimedSeries]
    combined_std: Optional[TimedSeries]
    combined_ci_low: Optional[TimedSeries]
    combined_ci_high: Optional[TimedSeries]
    audit: Dict[str, Any]


@dataclass
class SampleData:
    tracer_name: str
    tracer_label: str
    tracer_color: str
    tracer_marker: str
    sample_id: str
    roi_folder: str
    roi_prefix: str
    dx_mm: Optional[float]
    run_display: str
    selection_mode: str
    analysis_config_path: Optional[str]
    run_metadata_path: Optional[str]
    time_column_main: str
    time_column_extra: str
    paths_used: Dict[str, str]
    metrics: Dict[str, MetricSeries]
    audit_info: Dict[str, Any]


@dataclass
class AggregatedMetricData:
    key: str
    label: str
    time: np.ndarray
    center_mean: np.ndarray
    center_sample_count: np.ndarray
    between_sample_sd: np.ndarray
    between_sample_sem: np.ndarray
    sample_count_total: int
    sample_curves: List[Dict[str, Any]]
    fixed_std_mean: Optional[np.ndarray]
    fixed_ci_low_mean: Optional[np.ndarray]
    fixed_ci_high_mean: Optional[np.ndarray]
    combined_std_mean: Optional[np.ndarray]
    combined_ci_low_mean: Optional[np.ndarray]
    combined_ci_high_mean: Optional[np.ndarray]
    time_ranges: List[Tuple[float, float]]


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


def _read_json_any(base: str, relative_path: str) -> Dict[str, Any]:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return json.loads(zf.read(rel).decode("utf-8-sig"))
    with open(Path(base) / rel, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def load_csv_required_any(base: str, relative_path: str) -> pd.DataFrame:
    if not _path_exists_any(base, relative_path):
        raise FileNotFoundError(f"Required file not found: {relative_path}")
    return _read_csv_any(base, relative_path)


def load_csv_optional_any(base: str, relative_path: Optional[str]) -> Optional[pd.DataFrame]:
    if relative_path in (None, ""):
        return None
    if not _path_exists_any(base, relative_path):
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


def build_timecourse_rel_paths(run_rel: str, roi_folder: str) -> Dict[str, str]:
    run_rel = _normalize_rel_path(run_rel)
    roi_folder = str(roi_folder).strip("/").strip("\\")
    return {
        "main_csv": f"{run_rel}/multi_roi_timecourse_comparison.csv",
        "fit_parameters_csv": f"{run_rel}/{roi_folder}/CSVs_Summaries/fit_parameters_vs_time.csv",
        "run_metadata_json": f"{run_rel}/run_metadata.json",
    }


def build_timecourse_abs_paths(run_path: str, roi_folder: str) -> Dict[str, str]:
    run_dir = Path(run_path)
    return {
        "main_csv": str(run_dir / "multi_roi_timecourse_comparison.csv"),
        "fit_parameters_csv": str(run_dir / str(roi_folder) / "CSVs_Summaries" / "fit_parameters_vs_time.csv"),
        "run_metadata_json": str(run_dir / "run_metadata.json"),
    }


def infer_prefix_from_metric(df: pd.DataFrame, metric_suffixes: Sequence[str], user_prefix: Optional[str]) -> str:
    metric_suffixes = list(metric_suffixes)
    if user_prefix not in (None, ""):
        prefix = str(user_prefix)
        if any(f"{prefix}_{suffix}" in df.columns for suffix in metric_suffixes):
            return prefix

    matches = set()
    for column in df.columns:
        for suffix in metric_suffixes:
            suffix_text = f"_{suffix}"
            if column.endswith(suffix_text):
                matches.add(column[: -len(suffix_text)])
    if not matches:
        raise KeyError(
            "Could not infer a ROI prefix from the time-course CSV.\nAvailable columns:\n"
            + "\n".join(df.columns.tolist())
        )
    if len(matches) > 1:
        raise ValueError(
            "Could not infer a unique roi_prefix from the time-course CSV. "
            f"Found prefixes: {sorted(matches)}. Set roi_prefix explicitly in the comparison config."
        )
    return next(iter(matches))


def first_matching_column(df: pd.DataFrame, prefix: str, candidates: Sequence[str]) -> Optional[str]:
    for candidate in candidates:
        prefixed = f"{prefix}_{candidate}"
        if prefixed in df.columns:
            return prefixed
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def build_timed_series(df: pd.DataFrame, time_col: str, value_col: str, source_csv: str) -> TimedSeries:
    return TimedSeries(
        time=pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float),
        values=pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float),
        source_csv=source_csv,
        source_column=value_col,
    )


def extract_timed_series_any(
    main_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    prefix: str,
    candidates: Sequence[str],
    main_time_col: str,
    extra_time_col: str,
) -> Optional[TimedSeries]:
    if not candidates:
        return None
    main_column = first_matching_column(main_df, prefix, candidates)
    if main_column is not None:
        return build_timed_series(main_df, main_time_col, main_column, "main_csv")
    extra_column = first_matching_column(extra_df, prefix, candidates)
    if extra_column is not None:
        return build_timed_series(extra_df, extra_time_col, extra_column, "fit_parameters_csv")
    return None


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


def build_time_grid_from_arrays(time_arrays: Sequence[np.ndarray]) -> np.ndarray:
    dt_candidates = []
    min_time = None
    max_time = None
    has_any = False
    for arr in time_arrays:
        arr = np.asarray(arr, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        has_any = True
        current_min = float(np.min(finite))
        current_max = float(np.max(finite))
        min_time = current_min if min_time is None else min(min_time, current_min)
        max_time = current_max if max_time is None else max(max_time, current_max)
        if finite.size > 1:
            diffs = np.diff(np.sort(finite))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                dt_candidates.append(float(np.min(diffs)))
    if not has_any or min_time is None or max_time is None:
        return np.asarray([], dtype=float)
    dt = min(dt_candidates) if dt_candidates else 1.0
    point_count = int(np.floor((max_time - min_time) / dt)) + 1
    grid = min_time + np.arange(point_count, dtype=float) * dt
    if grid.size == 0 or grid[-1] < max_time - 0.25 * dt:
        grid = np.append(grid, max_time)
    return grid


def regrid_series(series: Optional[TimedSeries], target_time: np.ndarray) -> Optional[np.ndarray]:
    if series is None:
        return None
    return interp_series(series.time, series.values, target_time)


def combine_series_terms(*terms: Optional[TimedSeries]) -> Optional[TimedSeries]:
    valid_terms = [term for term in terms if term is not None]
    if not valid_terms:
        return None
    grid = build_time_grid_from_arrays([term.time for term in valid_terms])
    if grid.size == 0:
        return None
    regridded = np.vstack([regrid_series(term, grid) for term in valid_terms])
    finite_mask = np.isfinite(regridded)
    any_finite = np.any(finite_mask, axis=0)
    rss = np.sqrt(np.nansum(np.where(finite_mask, regridded, 0.0) ** 2, axis=0))
    rss[~any_finite] = np.nan
    return TimedSeries(
        time=grid,
        values=rss,
        source_csv="derived",
        source_column="rss(" + ", ".join(term.source_column for term in valid_terms) + ")",
    )


def approx_ci_from_std_series(
    center: Optional[TimedSeries],
    std_values: Optional[TimedSeries],
    z_value: float = Z95,
) -> Tuple[Optional[TimedSeries], Optional[TimedSeries]]:
    if center is None or std_values is None:
        return None, None
    center_on_std_grid = interp_series(center.time, center.values, std_values.time)
    return (
        TimedSeries(
            time=std_values.time.copy(),
            values=center_on_std_grid - z_value * std_values.values,
            source_csv="derived",
            source_column=f"{center.source_column}-z*{std_values.source_column}",
        ),
        TimedSeries(
            time=std_values.time.copy(),
            values=center_on_std_grid + z_value * std_values.values,
            source_csv="derived",
            source_column=f"{center.source_column}+z*{std_values.source_column}",
        ),
    )


def build_metric_series(
    spec: Dict[str, Any],
    main_df: pd.DataFrame,
    extra_df: pd.DataFrame,
    prefix: str,
    main_time_col: str,
    extra_time_col: str,
) -> MetricSeries:
    center = extract_timed_series_any(main_df, extra_df, prefix, spec["center_candidates"], main_time_col, extra_time_col)

    fixed_std_direct = extract_timed_series_any(
        main_df,
        extra_df,
        prefix,
        spec.get("fixed_direct_std_candidates", []),
        main_time_col,
        extra_time_col,
    )
    fixed_std_components = [
        extract_timed_series_any(main_df, extra_df, prefix, candidates, main_time_col, extra_time_col)
        for candidates in spec.get("fixed_component_candidates", [])
    ]
    fixed_std = fixed_std_direct if fixed_std_direct is not None else combine_series_terms(*fixed_std_components)
    fixed_ci_low = extract_timed_series_any(
        main_df,
        extra_df,
        prefix,
        spec.get("fixed_ci_low_candidates", []),
        main_time_col,
        extra_time_col,
    )
    fixed_ci_high = extract_timed_series_any(
        main_df,
        extra_df,
        prefix,
        spec.get("fixed_ci_high_candidates", []),
        main_time_col,
        extra_time_col,
    )
    if fixed_ci_low is None or fixed_ci_high is None:
        approx_low, approx_high = approx_ci_from_std_series(center, fixed_std, z_value=Z95)
        fixed_ci_low = fixed_ci_low or approx_low
        fixed_ci_high = fixed_ci_high or approx_high

    combined_std_direct = extract_timed_series_any(
        main_df,
        extra_df,
        prefix,
        spec.get("combined_direct_std_candidates", []),
        main_time_col,
        extra_time_col,
    )
    combined_std_components = [
        extract_timed_series_any(main_df, extra_df, prefix, candidates, main_time_col, extra_time_col)
        for candidates in spec.get("combined_component_candidates", [])
    ]
    combined_std = combined_std_direct if combined_std_direct is not None else combine_series_terms(*combined_std_components)
    combined_ci_low = extract_timed_series_any(
        main_df,
        extra_df,
        prefix,
        spec.get("combined_ci_low_candidates", []),
        main_time_col,
        extra_time_col,
    )
    combined_ci_high = extract_timed_series_any(
        main_df,
        extra_df,
        prefix,
        spec.get("combined_ci_high_candidates", []),
        main_time_col,
        extra_time_col,
    )
    if combined_ci_low is None or combined_ci_high is None:
        approx_low, approx_high = approx_ci_from_std_series(center, combined_std, z_value=Z95)
        combined_ci_low = combined_ci_low or approx_low
        combined_ci_high = combined_ci_high or approx_high

    return MetricSeries(
        key=spec["key"],
        label=spec["label"],
        center=center,
        fixed_std=fixed_std,
        fixed_ci_low=fixed_ci_low,
        fixed_ci_high=fixed_ci_high,
        combined_std=combined_std,
        combined_ci_low=combined_ci_low,
        combined_ci_high=combined_ci_high,
        audit={
            "center_source": None if center is None else f"{center.source_csv}:{center.source_column}",
            "fixed_std_source": None if fixed_std is None else f"{fixed_std.source_csv}:{fixed_std.source_column}",
            "fixed_ci_low_source": None if fixed_ci_low is None else f"{fixed_ci_low.source_csv}:{fixed_ci_low.source_column}",
            "fixed_ci_high_source": None if fixed_ci_high is None else f"{fixed_ci_high.source_csv}:{fixed_ci_high.source_column}",
            "combined_std_source": None if combined_std is None else f"{combined_std.source_csv}:{combined_std.source_column}",
            "combined_ci_low_source": None if combined_ci_low is None else f"{combined_ci_low.source_csv}:{combined_ci_low.source_column}",
            "combined_ci_high_source": None if combined_ci_high is None else f"{combined_ci_high.source_csv}:{combined_ci_high.source_column}",
        },
    )


def validate_and_resolve_config(config: Dict[str, Any], config_path: Path) -> Dict[str, Any]:
    if "tracers" not in config or not config["tracers"]:
        raise ValueError("Comparison config must define a non-empty 'tracers' list.")

    resolved: Dict[str, Any] = {}
    resolved["results_source"] = _resolve_path_from_config(config.get("results_source"), config_path)
    resolved["out_folder"] = _resolve_path_from_config(config.get("out_folder"), config_path)
    if resolved["out_folder"] in (None, ""):
        raise ValueError("Comparison config must define a non-empty 'out_folder'.")

    resolved["figure_title"] = config.get("figure_title", FIGURE_TITLE_DEFAULT)
    resolved["time_unit_label"] = config.get("time_unit_label", TIME_UNIT_LABEL_DEFAULT)
    resolved["shared_time_axis_min"] = (
        float(config["shared_time_axis_min"]) if config.get("shared_time_axis_min") is not None else None
    )
    resolved["shared_time_axis_max"] = (
        float(config["shared_time_axis_max"]) if config.get("shared_time_axis_max") is not None else None
    )
    resolved["report_windows"] = config.get("report_windows", DEFAULT_REPORT_WINDOWS)

    metric_ylims = dict(DEFAULT_METRIC_YLIMS)
    override_metric_ylims = config.get("metric_ylims", {}) or {}
    if not isinstance(override_metric_ylims, dict):
        raise ValueError("metric_ylims must be a JSON object mapping metric keys to [ymin, ymax].")
    for key, value in override_metric_ylims.items():
        if value in (None, []):
            continue
        if not isinstance(value, list) or len(value) != 2:
            raise ValueError(f"metric_ylims['{key}'] must be a 2-item list.")
        metric_ylims[key] = [float(value[0]), float(value[1])]
    resolved["metric_ylims"] = metric_ylims

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
                "roi_prefix": tracer_cfg.get("roi_prefix"),
                "dx_mm": tracer_cfg.get("dx_mm"),
                "samples": samples_resolved,
            }
        )
    resolved["tracers"] = tracers_resolved
    return resolved


def load_sample_data(tracer_cfg: Dict[str, Any], sample_cfg: Dict[str, Any], results_source: Optional[str]) -> SampleData:
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

    if selection_mode in {"analysis_config", "run_path"}:
        paths_used = build_timecourse_abs_paths(resolved_run, str(roi_folder))
        loader_required = load_csv_required_path
    else:
        paths_used = build_timecourse_rel_paths(resolved_run, str(roi_folder))
        loader_required = lambda relative_path: load_csv_required_any(results_source, relative_path)

    main_df = loader_required(paths_used["main_csv"])
    extra_df = loader_required(paths_used["fit_parameters_csv"])

    main_time_col = find_time_column(main_df)
    extra_time_col = find_time_column(extra_df)

    roi_prefix_candidate = sample_cfg.get("roi_prefix") or tracer_cfg.get("roi_prefix") or str(roi_folder)
    roi_prefix = infer_prefix_from_metric(
        main_df,
        ["effective_diffusivity", "mean_conc", "fitted_Cs"],
        roi_prefix_candidate,
    )

    dx_value = sample_cfg.get("dx_mm")
    if dx_value in (None, ""):
        dx_value = tracer_cfg.get("dx_mm")
    if dx_value in (None, ""):
        dx_value = depth_spacing_from_run_metadata(run_metadata_doc)
    dx_mm = None if dx_value in (None, "") else float(dx_value)

    metrics = {
        spec["key"]: build_metric_series(spec, main_df, extra_df, roi_prefix, main_time_col, extra_time_col)
        for spec in TIMECOURSE_METRIC_SPECS
    }

    sample_id = sample_cfg.get("sample_id")
    if sample_id in (None, ""):
        sample_id = Path(run_display).name

    audit_info = {
        "analysis_config_path": analysis_config_path,
        "run_metadata_path": run_metadata_path,
        "paths_used": paths_used,
        "time_column_main": main_time_col,
        "time_column_extra": extra_time_col,
        "roi_folder": str(roi_folder),
        "roi_prefix": roi_prefix,
        "roi_inferred_from_run_metadata": first_roi_name_from_run_metadata(run_metadata_doc),
        "roi_inferred_from_analysis_config": first_roi_name_from_analysis_config(analysis_config_doc),
        "dx_mm": dx_mm,
        "dx_inferred_from_run_metadata": depth_spacing_from_run_metadata(run_metadata_doc),
        "metrics": {metric_key: metric.audit for metric_key, metric in metrics.items()},
    }

    return SampleData(
        tracer_name=tracer_cfg["name"],
        tracer_label=tracer_cfg["label"],
        tracer_color=tracer_cfg["color"],
        tracer_marker=tracer_cfg["marker"],
        sample_id=str(sample_id),
        roi_folder=str(roi_folder),
        roi_prefix=roi_prefix,
        dx_mm=dx_mm,
        run_display=run_display,
        selection_mode=selection_mode,
        analysis_config_path=analysis_config_path,
        run_metadata_path=run_metadata_path,
        time_column_main=main_time_col,
        time_column_extra=extra_time_col,
        paths_used=paths_used,
        metrics=metrics,
        audit_info=audit_info,
    )


def build_common_time_grid(
    samples: Sequence[SampleData],
    shared_tmin: Optional[float],
    shared_tmax: Optional[float],
) -> np.ndarray:
    time_arrays = []
    for sample in samples:
        for metric in sample.metrics.values():
            if metric.center is not None:
                time_arrays.append(metric.center.time)
    grid = build_time_grid_from_arrays(time_arrays)
    if grid.size == 0:
        raise ValueError("Could not build a common time grid because no valid time-course metric was found.")

    min_time = float(shared_tmin) if shared_tmin is not None else float(grid[0])
    max_time = float(shared_tmax) if shared_tmax is not None else float(grid[-1])
    if max_time < min_time:
        raise ValueError("shared_time_axis_max must be >= shared_time_axis_min.")

    dt_candidates = []
    for arr in time_arrays:
        finite = np.asarray(arr, dtype=float)
        finite = finite[np.isfinite(finite)]
        if finite.size > 1:
            diffs = np.diff(np.sort(finite))
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if diffs.size:
                dt_candidates.append(float(np.min(diffs)))
    dt = min(dt_candidates) if dt_candidates else 1.0
    point_count = int(np.floor((max_time - min_time) / dt)) + 1
    common_grid = min_time + np.arange(point_count, dtype=float) * dt
    if common_grid.size == 0 or common_grid[-1] < max_time - 0.25 * dt:
        common_grid = np.append(common_grid, max_time)
    return common_grid


def nanmean_stack(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    finite = np.isfinite(arr)
    counts = np.sum(finite, axis=0)
    totals = np.sum(np.where(finite, arr, 0.0), axis=0)
    out = np.full(arr.shape[1:], np.nan, dtype=float)
    valid = counts > 0
    out[valid] = totals[valid] / counts[valid]
    return out


def rowwise_sample_sd(stack: np.ndarray) -> np.ndarray:
    arr = np.asarray(stack, dtype=float)
    counts = np.sum(np.isfinite(arr), axis=0)
    means = nanmean_stack(arr)
    centered = np.where(np.isfinite(arr), arr - means, np.nan)
    ss = np.nansum(centered ** 2, axis=0)
    out = np.full(means.shape, np.nan, dtype=float)
    valid = counts > 1
    out[valid] = np.sqrt(ss[valid] / (counts[valid] - 1))
    return out


def aggregate_tracer_metric(
    samples: Sequence[SampleData],
    metric_key: str,
    time_grid: np.ndarray,
) -> Optional[AggregatedMetricData]:
    sample_curves = []
    center_rows = []
    fixed_std_rows = []
    fixed_ci_low_rows = []
    fixed_ci_high_rows = []
    combined_std_rows = []
    combined_ci_low_rows = []
    combined_ci_high_rows = []
    time_ranges = []

    for sample in samples:
        metric = sample.metrics.get(metric_key)
        if metric is None or metric.center is None:
            continue

        center_values = regrid_series(metric.center, time_grid)
        center_rows.append(center_values)
        sample_curves.append({"sample_id": sample.sample_id, "values": center_values})

        finite_times = metric.center.time[np.isfinite(metric.center.time) & np.isfinite(metric.center.values)]
        if finite_times.size:
            time_ranges.append((float(np.min(finite_times)), float(np.max(finite_times))))

        fixed_std_values = regrid_series(metric.fixed_std, time_grid)
        if fixed_std_values is not None:
            fixed_std_rows.append(fixed_std_values)
        fixed_ci_low_values = regrid_series(metric.fixed_ci_low, time_grid)
        if fixed_ci_low_values is not None:
            fixed_ci_low_rows.append(fixed_ci_low_values)
        fixed_ci_high_values = regrid_series(metric.fixed_ci_high, time_grid)
        if fixed_ci_high_values is not None:
            fixed_ci_high_rows.append(fixed_ci_high_values)

        combined_std_values = regrid_series(metric.combined_std, time_grid)
        if combined_std_values is not None:
            combined_std_rows.append(combined_std_values)
        combined_ci_low_values = regrid_series(metric.combined_ci_low, time_grid)
        if combined_ci_low_values is not None:
            combined_ci_low_rows.append(combined_ci_low_values)
        combined_ci_high_values = regrid_series(metric.combined_ci_high, time_grid)
        if combined_ci_high_values is not None:
            combined_ci_high_rows.append(combined_ci_high_values)

    if not center_rows:
        return None

    center_stack = np.vstack(center_rows)
    center_count = np.sum(np.isfinite(center_stack), axis=0)
    center_mean = nanmean_stack(center_stack)
    between_sd = rowwise_sample_sd(center_stack)
    between_sem = np.full(center_mean.shape, np.nan, dtype=float)
    valid_sem = center_count > 0
    between_sem[valid_sem] = between_sd[valid_sem] / np.sqrt(center_count[valid_sem])

    return AggregatedMetricData(
        key=metric_key,
        label=METRIC_SPEC_BY_KEY[metric_key]["label"],
        time=time_grid,
        center_mean=center_mean,
        center_sample_count=center_count,
        between_sample_sd=between_sd,
        between_sample_sem=between_sem,
        sample_count_total=len(center_rows),
        sample_curves=sample_curves,
        fixed_std_mean=None if not fixed_std_rows else nanmean_stack(np.vstack(fixed_std_rows)),
        fixed_ci_low_mean=None if not fixed_ci_low_rows else nanmean_stack(np.vstack(fixed_ci_low_rows)),
        fixed_ci_high_mean=None if not fixed_ci_high_rows else nanmean_stack(np.vstack(fixed_ci_high_rows)),
        combined_std_mean=None if not combined_std_rows else nanmean_stack(np.vstack(combined_std_rows)),
        combined_ci_low_mean=None if not combined_ci_low_rows else nanmean_stack(np.vstack(combined_ci_low_rows)),
        combined_ci_high_mean=None if not combined_ci_high_rows else nanmean_stack(np.vstack(combined_ci_high_rows)),
        time_ranges=time_ranges,
    )


def build_uncertainty_note(mode: str) -> str:
    notes = {
        "combined_95ci": "Band shows the average within-sample combined 95% interval.",
        "combined_1sd": "Band shows the average within-sample combined 1 SD envelope.",
        "fixed_roi_95ci": "Band shows the average within-sample fixed-ROI 95% interval.",
        "fixed_roi_1sd": "Band shows the average within-sample fixed-ROI 1 SD envelope.",
        "none": "Bold lines show tracer means; thin lines show the contributing samples.",
    }
    return notes[mode]


def resolve_aggregated_band(
    aggregate: AggregatedMetricData,
    mode: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if mode == "combined_95ci":
        return aggregate.combined_ci_low_mean, aggregate.combined_ci_high_mean
    if mode == "fixed_roi_95ci":
        return aggregate.fixed_ci_low_mean, aggregate.fixed_ci_high_mean
    if mode == "combined_1sd" and aggregate.combined_std_mean is not None:
        return aggregate.center_mean - aggregate.combined_std_mean, aggregate.center_mean + aggregate.combined_std_mean
    if mode == "fixed_roi_1sd" and aggregate.fixed_std_mean is not None:
        return aggregate.center_mean - aggregate.fixed_std_mean, aggregate.center_mean + aggregate.fixed_std_mean
    return None, None


def output_root(out_folder: str) -> Path:
    return Path(out_folder) / TIMECOURSE_ROOT_NAME


def save_plot(fig: plt.Figure, out_folder: str, subfolder: str, out_name: str) -> str:
    folder = output_root(out_folder) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    out_path = folder / out_name
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def save_timecourse_plot(
    out_folder: str,
    subfolder: str,
    out_name: str,
    mode: str,
    plot_spec: Dict[str, str],
    tracer_order: Sequence[str],
    tracer_labels: Dict[str, str],
    tracer_styles: Dict[str, Dict[str, str]],
    aggregated_by_tracer: Dict[str, Dict[str, AggregatedMetricData]],
    figure_title: str,
    time_unit_label: str,
    metric_ylims: Dict[str, List[float]],
) -> Optional[str]:
    metric_key = plot_spec["metric_key"]
    plotted_any = False
    fig, ax = plt.subplots(figsize=(10, 6))

    for tracer_name in tracer_order:
        aggregate = aggregated_by_tracer.get(tracer_name, {}).get(metric_key)
        if aggregate is None:
            continue
        plotted_any = True
        style = tracer_styles[tracer_name]

        for sample_curve in aggregate.sample_curves:
            ax.plot(
                aggregate.time,
                sample_curve["values"],
                color=style["color"],
                linewidth=SAMPLE_LINEWIDTH,
                alpha=SAMPLE_LINE_ALPHA,
            )

        band_low, band_high = resolve_aggregated_band(aggregate, mode)
        if band_low is not None and band_high is not None:
            valid = np.isfinite(aggregate.time) & np.isfinite(band_low) & np.isfinite(band_high)
            if np.any(valid):
                ax.fill_between(
                    aggregate.time[valid],
                    band_low[valid],
                    band_high[valid],
                    color=style["color"],
                    alpha=UNC_FILL_ALPHA,
                )

        ax.plot(
            aggregate.time,
            aggregate.center_mean,
            color=style["color"],
            linewidth=CENTER_LINEWIDTH,
            marker=style["marker"],
            markersize=LINE_MARKER_SIZE,
            markevery=max(1, len(aggregate.time) // 10),
            label=f"{tracer_labels[tracer_name]} mean (n={aggregate.sample_count_total})",
        )

    if not plotted_any:
        plt.close(fig)
        return None

    ax.set_xlabel(f"Time ({time_unit_label})")
    ax.set_ylabel(METRIC_SPEC_BY_KEY[metric_key]["label"])
    ax.set_title(f"{plot_spec['title']}\n{build_uncertainty_note(mode)}")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if plot_spec["ylim_key"] in metric_ylims:
        ymin, ymax = metric_ylims[plot_spec["ylim_key"]]
        ax.set_ylim(ymin, ymax)

    first_aggregate = next(
        aggregate
        for tracer_name in tracer_order
        if (aggregate := aggregated_by_tracer.get(tracer_name, {}).get(metric_key)) is not None
    )
    ax.set_xlim(float(first_aggregate.time[0]), float(first_aggregate.time[-1]))
    fig.suptitle(figure_title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return save_plot(fig, out_folder, subfolder, out_name)


def build_per_sample_all_time_metrics(samples: Sequence[SampleData]) -> pd.DataFrame:
    rows = []
    for sample in samples:
        for metric_key, metric in sample.metrics.items():
            if metric.center is None:
                continue

            target_time = metric.center.time
            fixed_std = regrid_series(metric.fixed_std, target_time)
            fixed_ci_low = regrid_series(metric.fixed_ci_low, target_time)
            fixed_ci_high = regrid_series(metric.fixed_ci_high, target_time)
            combined_std = regrid_series(metric.combined_std, target_time)
            combined_ci_low = regrid_series(metric.combined_ci_low, target_time)
            combined_ci_high = regrid_series(metric.combined_ci_high, target_time)

            for idx, time_value in enumerate(target_time):
                if not np.isfinite(time_value):
                    continue
                rows.append(
                    {
                        "tracer_name": sample.tracer_name,
                        "tracer_label": sample.tracer_label,
                        "sample_id": sample.sample_id,
                        "run_display": sample.run_display,
                        "selection_mode": sample.selection_mode,
                        "roi_folder": sample.roi_folder,
                        "roi_prefix": sample.roi_prefix,
                        "dx_mm": sample.dx_mm,
                        "metric_key": metric_key,
                        "metric_label": metric.label,
                        "time_min": float(time_value),
                        "center_value": float(metric.center.values[idx]) if np.isfinite(metric.center.values[idx]) else float("nan"),
                        "fixed_std": float(fixed_std[idx]) if fixed_std is not None and np.isfinite(fixed_std[idx]) else float("nan"),
                        "fixed_ci_low": float(fixed_ci_low[idx]) if fixed_ci_low is not None and np.isfinite(fixed_ci_low[idx]) else float("nan"),
                        "fixed_ci_high": float(fixed_ci_high[idx]) if fixed_ci_high is not None and np.isfinite(fixed_ci_high[idx]) else float("nan"),
                        "combined_std": float(combined_std[idx]) if combined_std is not None and np.isfinite(combined_std[idx]) else float("nan"),
                        "combined_ci_low": float(combined_ci_low[idx]) if combined_ci_low is not None and np.isfinite(combined_ci_low[idx]) else float("nan"),
                        "combined_ci_high": float(combined_ci_high[idx]) if combined_ci_high is not None and np.isfinite(combined_ci_high[idx]) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def build_tracer_group_all_time_metrics(
    tracer_order: Sequence[str],
    tracer_labels: Dict[str, str],
    aggregated_by_tracer: Dict[str, Dict[str, AggregatedMetricData]],
) -> pd.DataFrame:
    rows = []
    for tracer_name in tracer_order:
        for metric_key, aggregate in aggregated_by_tracer.get(tracer_name, {}).items():
            for idx, time_value in enumerate(aggregate.time):
                rows.append(
                    {
                        "tracer_name": tracer_name,
                        "tracer_label": tracer_labels[tracer_name],
                        "metric_key": metric_key,
                        "metric_label": aggregate.label,
                        "time_min": float(time_value),
                        "sample_count_total": int(aggregate.sample_count_total),
                        "sample_count_at_time": int(aggregate.center_sample_count[idx]),
                        "center_mean": float(aggregate.center_mean[idx]) if np.isfinite(aggregate.center_mean[idx]) else float("nan"),
                        "between_sample_sd": float(aggregate.between_sample_sd[idx]) if np.isfinite(aggregate.between_sample_sd[idx]) else float("nan"),
                        "between_sample_sem": float(aggregate.between_sample_sem[idx]) if np.isfinite(aggregate.between_sample_sem[idx]) else float("nan"),
                        "fixed_std_mean": float(aggregate.fixed_std_mean[idx]) if aggregate.fixed_std_mean is not None and np.isfinite(aggregate.fixed_std_mean[idx]) else float("nan"),
                        "fixed_ci_low_mean": float(aggregate.fixed_ci_low_mean[idx]) if aggregate.fixed_ci_low_mean is not None and np.isfinite(aggregate.fixed_ci_low_mean[idx]) else float("nan"),
                        "fixed_ci_high_mean": float(aggregate.fixed_ci_high_mean[idx]) if aggregate.fixed_ci_high_mean is not None and np.isfinite(aggregate.fixed_ci_high_mean[idx]) else float("nan"),
                        "combined_std_mean": float(aggregate.combined_std_mean[idx]) if aggregate.combined_std_mean is not None and np.isfinite(aggregate.combined_std_mean[idx]) else float("nan"),
                        "combined_ci_low_mean": float(aggregate.combined_ci_low_mean[idx]) if aggregate.combined_ci_low_mean is not None and np.isfinite(aggregate.combined_ci_low_mean[idx]) else float("nan"),
                        "combined_ci_high_mean": float(aggregate.combined_ci_high_mean[idx]) if aggregate.combined_ci_high_mean is not None and np.isfinite(aggregate.combined_ci_high_mean[idx]) else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def summarize_metric_rows(rows: pd.DataFrame) -> Dict[str, Any]:
    center_values = pd.to_numeric(rows["center_value"], errors="coerce")
    fixed_std_values = pd.to_numeric(rows["fixed_std"], errors="coerce")
    combined_std_values = pd.to_numeric(rows["combined_std"], errors="coerce")
    fixed_ci_width = pd.to_numeric(rows["fixed_ci_high"], errors="coerce") - pd.to_numeric(rows["fixed_ci_low"], errors="coerce")
    combined_ci_width = pd.to_numeric(rows["combined_ci_high"], errors="coerce") - pd.to_numeric(rows["combined_ci_low"], errors="coerce")

    valid_center = center_values.dropna()
    return {
        "n_points": int(valid_center.shape[0]),
        "center_mean": float(valid_center.mean()) if not valid_center.empty else float("nan"),
        "center_std": float(valid_center.std(ddof=1)) if valid_center.shape[0] > 1 else float("nan"),
        "fixed_std_mean": float(fixed_std_values.dropna().mean()) if not fixed_std_values.dropna().empty else float("nan"),
        "combined_std_mean": float(combined_std_values.dropna().mean()) if not combined_std_values.dropna().empty else float("nan"),
        "fixed_ci_width_mean": float(fixed_ci_width.dropna().mean()) if not fixed_ci_width.dropna().empty else float("nan"),
        "combined_ci_width_mean": float(combined_ci_width.dropna().mean()) if not combined_ci_width.dropna().empty else float("nan"),
    }


def build_window_sample_summary(all_time_df: pd.DataFrame, report_windows: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    if all_time_df.empty:
        return pd.DataFrame(rows)

    for window in report_windows:
        window_name = str(window.get("name", "window"))
        min_time = float(window.get("min_time_min")) if window.get("min_time_min") is not None else None
        max_time = float(window.get("max_time_min")) if window.get("max_time_min") is not None else None

        mask = pd.Series(True, index=all_time_df.index)
        if min_time is not None:
            mask &= all_time_df["time_min"] >= min_time
        if max_time is not None:
            mask &= all_time_df["time_min"] <= max_time
        window_df = all_time_df[mask].copy()
        if window_df.empty:
            continue

        group_columns = [
            "tracer_name",
            "tracer_label",
            "sample_id",
            "run_display",
            "selection_mode",
            "roi_folder",
            "roi_prefix",
            "dx_mm",
            "metric_key",
            "metric_label",
        ]
        for group_values, group_df in window_df.groupby(group_columns, dropna=False):
            summary = summarize_metric_rows(group_df)
            row = dict(zip(group_columns, group_values))
            row.update(
                {
                    "window_name": window_name,
                    "window_min_time_min": min_time,
                    "window_max_time_min": max_time,
                    **summary,
                }
            )
            rows.append(row)

    return pd.DataFrame(rows)


def build_window_group_summary(window_sample_summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if window_sample_summary_df.empty:
        return pd.DataFrame(rows)

    group_columns = [
        "window_name",
        "window_min_time_min",
        "window_max_time_min",
        "tracer_name",
        "tracer_label",
        "metric_key",
        "metric_label",
    ]
    for group_values, group_df in window_sample_summary_df.groupby(group_columns, dropna=False):
        center_means = pd.to_numeric(group_df["center_mean"], errors="coerce").dropna()
        row = dict(zip(group_columns, group_values))
        row.update(
            {
                "sample_count": int(group_df["sample_id"].nunique()),
                "sample_mean_mean": float(center_means.mean()) if not center_means.empty else float("nan"),
                "sample_mean_sd": float(center_means.std(ddof=1)) if center_means.shape[0] > 1 else float("nan"),
                "sample_mean_sem": float(center_means.std(ddof=1) / np.sqrt(center_means.shape[0])) if center_means.shape[0] > 1 else float("nan"),
                "sample_center_std_mean": float(pd.to_numeric(group_df["center_std"], errors="coerce").dropna().mean()) if not pd.to_numeric(group_df["center_std"], errors="coerce").dropna().empty else float("nan"),
                "sample_fixed_std_mean": float(pd.to_numeric(group_df["fixed_std_mean"], errors="coerce").dropna().mean()) if not pd.to_numeric(group_df["fixed_std_mean"], errors="coerce").dropna().empty else float("nan"),
                "sample_combined_std_mean": float(pd.to_numeric(group_df["combined_std_mean"], errors="coerce").dropna().mean()) if not pd.to_numeric(group_df["combined_std_mean"], errors="coerce").dropna().empty else float("nan"),
                "sample_n_points_total": int(pd.to_numeric(group_df["n_points"], errors="coerce").fillna(0).sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_tables(
    out_folder: str,
    per_sample_all_time_df: pd.DataFrame,
    tracer_group_all_time_df: pd.DataFrame,
    window_sample_summary_df: pd.DataFrame,
    window_group_summary_df: pd.DataFrame,
) -> List[str]:
    summary_dir = output_root(out_folder) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    outputs = []
    table_specs = [
        ("per_sample_all_time_metrics.csv", per_sample_all_time_df),
        ("tracer_group_all_time_metrics.csv", tracer_group_all_time_df),
        ("window_per_sample_metric_summary.csv", window_sample_summary_df),
        ("window_tracer_group_metric_summary.csv", window_group_summary_df),
    ]
    for filename, df in table_specs:
        out_path = summary_dir / filename
        df.to_csv(out_path, index=False)
        outputs.append(str(out_path))

    if not window_sample_summary_df.empty:
        for window_name in sorted(window_sample_summary_df["window_name"].dropna().unique()):
            safe_name = str(window_name).strip().replace(" ", "_")
            out_path = summary_dir / f"{safe_name}_per_sample_metric_summary.csv"
            window_sample_summary_df[window_sample_summary_df["window_name"] == window_name].copy().to_csv(out_path, index=False)
            outputs.append(str(out_path))

    if not window_group_summary_df.empty:
        for window_name in sorted(window_group_summary_df["window_name"].dropna().unique()):
            safe_name = str(window_name).strip().replace(" ", "_")
            out_path = summary_dir / f"{safe_name}_tracer_group_metric_summary.csv"
            window_group_summary_df[window_group_summary_df["window_name"] == window_name].copy().to_csv(out_path, index=False)
            outputs.append(str(out_path))

    return outputs


def save_audit_report(out_folder: str, payload: Dict[str, Any]) -> None:
    out_dir = output_root(out_folder) / "audit"
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
        description="Compare time-course CSV outputs across multiple tracers and replicate samples."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to a JSON time-course comparison config.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = validate_and_resolve_config(load_config(config_path), config_path)

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
            load_sample_data(tracer_cfg, sample_cfg, config.get("results_source"))
            for sample_cfg in tracer_cfg["samples"]
        ]
        samples_by_tracer[tracer_cfg["name"]] = tracer_samples
        all_samples.extend(tracer_samples)

    time_grid = build_common_time_grid(
        all_samples,
        config["shared_time_axis_min"],
        config["shared_time_axis_max"],
    )

    aggregated_by_tracer: Dict[str, Dict[str, AggregatedMetricData]] = {}
    for tracer_name in tracer_order:
        aggregated_by_tracer[tracer_name] = {}
        for metric_key in METRIC_SPEC_BY_KEY:
            aggregate = aggregate_tracer_metric(samples_by_tracer[tracer_name], metric_key, time_grid)
            if aggregate is not None:
                aggregated_by_tracer[tracer_name][metric_key] = aggregate

    figure_outputs: Dict[str, List[str]] = {}
    for mode, subfolder, suffix in BAND_MODE_SPECS:
        mode_outputs = []
        for plot_spec in BAND_PLOT_SPECS:
            out_path = save_timecourse_plot(
                out_folder=config["out_folder"],
                subfolder=subfolder,
                out_name=f"{plot_spec['out_stem']}_{suffix}.png",
                mode=mode,
                plot_spec=plot_spec,
                tracer_order=tracer_order,
                tracer_labels=tracer_labels,
                tracer_styles=tracer_styles,
                aggregated_by_tracer=aggregated_by_tracer,
                figure_title=config["figure_title"],
                time_unit_label=config["time_unit_label"],
                metric_ylims=config["metric_ylims"],
            )
            if out_path is not None:
                mode_outputs.append(out_path)
        figure_outputs[subfolder] = mode_outputs

    diagnostic_outputs = []
    for plot_spec in DIAGNOSTIC_PLOT_SPECS:
        out_path = save_timecourse_plot(
            out_folder=config["out_folder"],
            subfolder="diagnostics",
            out_name=f"{plot_spec['out_stem']}.png",
            mode="none",
            plot_spec=plot_spec,
            tracer_order=tracer_order,
            tracer_labels=tracer_labels,
            tracer_styles=tracer_styles,
            aggregated_by_tracer=aggregated_by_tracer,
            figure_title=config["figure_title"],
            time_unit_label=config["time_unit_label"],
            metric_ylims=config["metric_ylims"],
        )
        if out_path is not None:
            diagnostic_outputs.append(out_path)
    figure_outputs["diagnostics"] = diagnostic_outputs

    per_sample_all_time_df = build_per_sample_all_time_metrics(all_samples)
    tracer_group_all_time_df = build_tracer_group_all_time_metrics(tracer_order, tracer_labels, aggregated_by_tracer)
    window_sample_summary_df = build_window_sample_summary(per_sample_all_time_df, config["report_windows"])
    window_group_summary_df = build_window_group_summary(window_sample_summary_df)
    table_outputs = write_tables(
        config["out_folder"],
        per_sample_all_time_df,
        tracer_group_all_time_df,
        window_sample_summary_df,
        window_group_summary_df,
    )

    audit_payload = {
        "config_path": str(config_path),
        "results_source": config.get("results_source"),
        "out_folder": config["out_folder"],
        "output_root": str(output_root(config["out_folder"])),
        "figure_title": config["figure_title"],
        "time_unit_label": config["time_unit_label"],
        "shared_time_axis_min_requested": config["shared_time_axis_min"],
        "shared_time_axis_max_requested": config["shared_time_axis_max"],
        "shared_time_axis_min_used": float(time_grid[0]),
        "shared_time_axis_max_used": float(time_grid[-1]),
        "common_time_grid_points": int(len(time_grid)),
        "metric_ylims": config["metric_ylims"],
        "report_windows": config["report_windows"],
        "uncertainty_definitions": UNCERTAINTY_DEFINITIONS,
        "tracers": {
            tracer_name: {
                "label": tracer_labels[tracer_name],
                "sample_ids": [sample.sample_id for sample in tracer_samples],
                "sample_runs": [sample.run_display for sample in tracer_samples],
                "roi_folders": sorted({sample.roi_folder for sample in tracer_samples}),
                "roi_prefixes": sorted({sample.roi_prefix for sample in tracer_samples}),
            }
            for tracer_name, tracer_samples in samples_by_tracer.items()
        },
        "sample_audit": {sample.sample_id: sample.audit_info for sample in all_samples},
        "figure_outputs": figure_outputs,
        "table_outputs": table_outputs,
    }
    save_audit_report(config["out_folder"], audit_payload)

    print("Saved time-course comparison outputs under:", str(output_root(config["out_folder"])))
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
