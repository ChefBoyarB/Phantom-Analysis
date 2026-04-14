from Compare_Time_Course_CSV_Plots_Engine import main


if __name__ == "__main__":
    raise SystemExit(main())


r'''
import io
import os
import json
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# USER INPUT
# -----------------------------
# Point this to either:
#   1) the Results_Paper_1.zip file, or
#   2) an extracted Results_Paper_1 folder
RESULTS_SOURCE = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1"

# -----------------------------------------------------------------
# DATASET SELECTION
# -----------------------------------------------------------------
# BEST PRACTICE FOR FINAL PAPER OUTPUTS:
# set the four explicit relative paths below for the exact VIS/GAD run
# you want to compare. This is the most robust option.
#
# Paths must be relative to RESULTS_SOURCE, for example:
# VIS_MAIN_REL = "VIS320_Swollen_No_Pressure_135kvp/multi_roi_timecourse_comparison.csv"
# VIS_EXTRA_REL = "VIS320_Swollen_No_Pressure_135kvp/VIS_320/CSVs_Summaries/fit_parameters_vs_time.csv"
#
# If any of these are left as None, the script falls back to hint-based search.
VIS_MAIN_REL = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\VIS320_Swollen_No_Pressure_135kvp\multi_roi_timecourse_comparison.csv"
VIS_EXTRA_REL = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\VIS320_Swollen_No_Pressure_135kvp\VIS_320\CSVs_Summaries\fit_parameters_vs_time.csv"
GAD_MAIN_REL = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\GAD_Swollen_No_Pressure_135kvp\multi_roi_timecourse_comparison.csv"
GAD_EXTRA_REL = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\GAD_Swollen_No_Pressure_135kvp\GAD\CSVs_Summaries\fit_parameters_vs_time.csv"

# Fallback hints used only when explicit relative paths above are not provided.
VIS_FOLDER_HINT = "VIS320_Swollen_No_Pressure_135kvp"
GAD_FOLDER_HINT = "GAD_Swollen_No_Pressure_135kvp"
VIS_ROI_FOLDER_HINT = "VIS_320"
GAD_ROI_FOLDER_HINT = "GAD"

# Optional explicit ROI prefixes. Leave as None to auto-infer from the selected CSV.
VIS_ROI_PREFIX = None
GAD_ROI_PREFIX = None

OUT_FOLDER = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\Paper_1_Comparisons"

YLIM_EFFECTIVE_DIFFUSIVITY = (0.0, 0.035)
YLIM_EFFECTIVE_DIFFUSIVITY_ZOOM = (0.0, 0.005)
YLIM_EFFECTIVE_CS = (15.0, 70.0)
YLIM_R2 = (0.0, 1.01)

# Summary window for paper-style late-time comparison
SUMMARY_MIN_TIME_MIN = 5.0
SUMMARY_MAX_TIME_MIN = None

# Clean uncertainty styling
UNC_FILL_ALPHA = 0.08
CENTER_LINEWIDTH = 2.0
MARKER = "o"
MARK_EVERY = None  # e.g. 4 for less clutter
Z95 = 1.96

# -----------------------------
# FILE HELPERS
# -----------------------------
def _is_zip_path(path: str) -> bool:
    return str(path).lower().endswith(".zip")


def _read_csv_any(base: str, relative_path: str) -> pd.DataFrame:
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return pd.read_csv(io.BytesIO(zf.read(relative_path)))
    return pd.read_csv(Path(base) / relative_path)


def _list_paths_any(base: str) -> Sequence[str]:
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return zf.namelist()
    out = []
    base_path = Path(base)
    for p in base_path.rglob("*"):
        if p.is_file():
            out.append(str(p.relative_to(base_path)).replace("\\", "/"))
    return out


def find_result_paths(base: str, folder_hint: str, roi_folder_hint: str) -> Tuple[str, str]:
    paths = _list_paths_any(base)
    main_candidates = [
        p for p in paths if folder_hint in p and p.endswith("multi_roi_timecourse_comparison.csv")
    ]
    extra_candidates = [
        p for p in paths
        if folder_hint in p and roi_folder_hint in p and p.endswith("CSVs_Summaries/fit_parameters_vs_time.csv")
    ]

    if not main_candidates:
        raise FileNotFoundError(f"Could not find multi_roi_timecourse_comparison.csv for {folder_hint}")
    if not extra_candidates:
        raise FileNotFoundError(f"Could not find fit_parameters_vs_time.csv for {folder_hint}/{roi_folder_hint}")

    return sorted(main_candidates)[0], sorted(extra_candidates)[0]


def _normalize_rel_path(rel_path: str) -> str:
    return str(rel_path).replace("\\", "/").strip("/")


def _path_exists_any(base: str, relative_path: str) -> bool:
    rel = _normalize_rel_path(relative_path)
    if _is_zip_path(base):
        with zipfile.ZipFile(base) as zf:
            return rel in zf.namelist()
    return (Path(base) / rel).exists()


def resolve_result_paths(base: str,
                         explicit_main_rel: Optional[str],
                         explicit_extra_rel: Optional[str],
                         folder_hint: str,
                         roi_folder_hint: str,
                         tracer_label: str) -> Tuple[str, str, str]:
    """
    Returns:
        main_rel, extra_rel, selection_mode
    selection_mode is one of:
        - "explicit"
        - "hint_search"
    """
    explicit_main_rel = None if explicit_main_rel in (None, "") else _normalize_rel_path(explicit_main_rel)
    explicit_extra_rel = None if explicit_extra_rel in (None, "") else _normalize_rel_path(explicit_extra_rel)

    if explicit_main_rel is not None or explicit_extra_rel is not None:
        if explicit_main_rel is None or explicit_extra_rel is None:
            raise ValueError(
                f"For {tracer_label}, either set both explicit paths or leave both as None. "
                f"Got main={explicit_main_rel}, extra={explicit_extra_rel}."
            )
        if not _path_exists_any(base, explicit_main_rel):
            raise FileNotFoundError(f"{tracer_label} explicit main CSV was not found: {explicit_main_rel}")
        if not _path_exists_any(base, explicit_extra_rel):
            raise FileNotFoundError(f"{tracer_label} explicit extra CSV was not found: {explicit_extra_rel}")
        return explicit_main_rel, explicit_extra_rel, "explicit"

    main_rel, extra_rel = find_result_paths(base, folder_hint, roi_folder_hint)
    return main_rel, extra_rel, "hint_search"


def save_audit_report(out_folder: str, payload: dict):
    out_dir = Path(out_folder) / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "comparison_audit.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    with open(out_dir / "comparison_audit.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -----------------------------
# DATA HELPERS
# -----------------------------
def infer_prefix_from_metric(df: pd.DataFrame, metric_suffix: str, user_prefix=None) -> str:
    if user_prefix is not None and str(user_prefix).strip() != "":
        candidate = f"{user_prefix}_{metric_suffix}"
        if candidate not in df.columns:
            raise KeyError(
                f"Column '{candidate}' was not found. Available columns are:\n" + "\n".join(df.columns.tolist())
            )
        return str(user_prefix)

    matches = [c for c in df.columns if c.endswith(f"_{metric_suffix}")]
    if not matches:
        raise KeyError(
            f"Could not find any column ending with '_{metric_suffix}'. Available columns are:\n" + "\n".join(df.columns.tolist())
        )
    return matches[0][: -len(f"_{metric_suffix}")]


def get_series_if_exists(df: Optional[pd.DataFrame], prefix: str, metric_suffix: str) -> Optional[pd.Series]:
    if df is None:
        return None

    prefixed_col = f"{prefix}_{metric_suffix}"
    if prefixed_col in df.columns:
        return pd.to_numeric(df[prefixed_col], errors="coerce")

    plain_col = metric_suffix
    if plain_col in df.columns:
        return pd.to_numeric(df[plain_col], errors="coerce")

    return None


def get_from_main_or_extra(main_df: pd.DataFrame,
                           extra_df: Optional[pd.DataFrame],
                           prefix: str,
                           metric_suffixes,
                           required: bool = False) -> Optional[pd.Series]:
    if isinstance(metric_suffixes, str):
        metric_suffixes = [metric_suffixes]

    for suffix in metric_suffixes:
        series = get_series_if_exists(main_df, prefix, suffix)
        if series is not None:
            return series

    for suffix in metric_suffixes:
        series = get_series_if_exists(extra_df, prefix, suffix)
        if series is not None:
            return series

    if required:
        available = main_df.columns.tolist()
        if extra_df is not None:
            available = sorted(set(available + extra_df.columns.tolist()))
        raise KeyError(
            f"Could not find any of {metric_suffixes} for prefix '{prefix}' in either the main or extra CSV.\n"
            f"Available columns are:\n" + "\n".join(available)
        )
    return None


def align_optional_series_to_time(time_series: pd.Series,
                                  value_series: Optional[pd.Series],
                                  source_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if value_series is None:
        return None
    if source_df is None:
        return value_series

    time_col = "time" if "time" in source_df.columns else "time_plot" if "time_plot" in source_df.columns else None
    if time_col is None:
        return value_series

    aligned = pd.DataFrame({"time": time_series}).merge(
        pd.DataFrame({"time": pd.to_numeric(source_df[time_col], errors="coerce"), "value": value_series}),
        on="time",
        how="left"
    )["value"]
    return pd.to_numeric(aligned, errors="coerce")


def get_aligned_from_main_or_extra(main_df: pd.DataFrame,
                                   extra_df: Optional[pd.DataFrame],
                                   prefix: str,
                                   metric_suffixes,
                                   time_series: pd.Series,
                                   required: bool = False) -> Optional[pd.Series]:
    if isinstance(metric_suffixes, str):
        metric_suffixes = [metric_suffixes]

    for suffix in metric_suffixes:
        series = get_series_if_exists(main_df, prefix, suffix)
        if series is not None:
            return series

    for suffix in metric_suffixes:
        series = get_series_if_exists(extra_df, prefix, suffix)
        if series is not None:
            return align_optional_series_to_time(time_series, series, extra_df)

    if required:
        return get_from_main_or_extra(main_df, extra_df, prefix, metric_suffixes, required=True)
    return None


def combine_std_terms(*terms: Optional[pd.Series]) -> Optional[pd.Series]:
    valid_terms = [pd.to_numeric(t, errors="coerce") for t in terms if t is not None]
    if not valid_terms:
        return None
    out = pd.Series(0.0, index=valid_terms[0].index, dtype=float)
    for term in valid_terms:
        out = out + term.fillna(0.0) ** 2
    return out.pow(0.5)


def ci_from_center_std(center: Optional[pd.Series], std: Optional[pd.Series], z: float = Z95):
    if center is None or std is None:
        return None, None
    center = pd.to_numeric(center, errors="coerce")
    std = pd.to_numeric(std, errors="coerce")
    return center - z * std, center + z * std


def save_plot(fig, subfolder: str, out_name: str):
    folder = Path(OUT_FOLDER) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    fig.savefig(folder / out_name, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# PLOTTING HELPERS
# -----------------------------
def make_band_plot(metric_vis: Optional[pd.Series],
                   metric_gad: Optional[pd.Series],
                   vis_time: pd.Series,
                   gad_time: pd.Series,
                   ylabel: str,
                   title: str,
                   subfolder: str,
                   out_name: str,
                   vis_low: Optional[pd.Series] = None,
                   vis_high: Optional[pd.Series] = None,
                   gad_low: Optional[pd.Series] = None,
                   gad_high: Optional[pd.Series] = None,
                   ylim=None,
                   vis_label: str = "VIS 320",
                   gad_label: str = "GAD"):
    if metric_vis is None or metric_gad is None:
        print(f"Skipping plot '{out_name}' because one or both required center curves are missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    vis_line, = ax.plot(vis_time, metric_vis, marker=MARKER, markevery=MARK_EVERY,
                        linewidth=CENTER_LINEWIDTH, label=vis_label)
    if vis_low is not None and vis_high is not None:
        vis_valid = metric_vis.notna() & vis_low.notna() & vis_high.notna()
        if vis_valid.any():
            ax.fill_between(vis_time[vis_valid], vis_low[vis_valid], vis_high[vis_valid],
                            alpha=UNC_FILL_ALPHA, color=vis_line.get_color())

    gad_line, = ax.plot(gad_time, metric_gad, marker=MARKER, markevery=MARK_EVERY,
                        linewidth=CENTER_LINEWIDTH, label=gad_label)
    if gad_low is not None and gad_high is not None:
        gad_valid = metric_gad.notna() & gad_low.notna() & gad_high.notna()
        if gad_valid.any():
            ax.fill_between(gad_time[gad_valid], gad_low[gad_valid], gad_high[gad_valid],
                            alpha=UNC_FILL_ALPHA, color=gad_line.get_color())

    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)
    ax.legend()
    save_plot(fig, subfolder, out_name)


def make_line_plot(vis_series: Optional[pd.Series],
                   gad_series: Optional[pd.Series],
                   vis_time: pd.Series,
                   gad_time: pd.Series,
                   ylabel: str,
                   title: str,
                   subfolder: str,
                   out_name: str,
                   ylim=None,
                   vis_label: str = "VIS 320",
                   gad_label: str = "GAD"):
    if vis_series is None or gad_series is None:
        print(f"Skipping plot '{out_name}' because one or both series are missing.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(vis_time, vis_series, marker=MARKER, markevery=MARK_EVERY, linewidth=CENTER_LINEWIDTH, label=vis_label)
    ax.plot(gad_time, gad_series, marker=MARKER, markevery=MARK_EVERY, linewidth=CENTER_LINEWIDTH, label=gad_label)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True)
    ax.legend()
    save_plot(fig, subfolder, out_name)


# -----------------------------
# LOAD
# -----------------------------
vis_main_rel, vis_extra_rel, vis_selection_mode = resolve_result_paths(
    RESULTS_SOURCE, VIS_MAIN_REL, VIS_EXTRA_REL, VIS_FOLDER_HINT, VIS_ROI_FOLDER_HINT, "VIS"
)
gad_main_rel, gad_extra_rel, gad_selection_mode = resolve_result_paths(
    RESULTS_SOURCE, GAD_MAIN_REL, GAD_EXTRA_REL, GAD_FOLDER_HINT, GAD_ROI_FOLDER_HINT, "GAD"
)

vis_main_df = _read_csv_any(RESULTS_SOURCE, vis_main_rel)
gad_main_df = _read_csv_any(RESULTS_SOURCE, gad_main_rel)
vis_extra_df = _read_csv_any(RESULTS_SOURCE, vis_extra_rel)
gad_extra_df = _read_csv_any(RESULTS_SOURCE, gad_extra_rel)

VIS_ROI_PREFIX = infer_prefix_from_metric(vis_main_df, "effective_diffusivity", VIS_ROI_PREFIX)
GAD_ROI_PREFIX = infer_prefix_from_metric(gad_main_df, "effective_diffusivity", GAD_ROI_PREFIX)

print(f"VIS ROI prefix used: {VIS_ROI_PREFIX}")
print(f"GAD ROI prefix used: {GAD_ROI_PREFIX}")
print(f"VIS selection mode: {vis_selection_mode}")
print(f"GAD selection mode: {gad_selection_mode}")
print(f"VIS main CSV: {vis_main_rel}")
print(f"GAD main CSV: {gad_main_rel}")
print(f"VIS extra CSV: {vis_extra_rel}")
print(f"GAD extra CSV: {gad_extra_rel}")

vis_time = pd.to_numeric(vis_main_df["time"], errors="coerce")
gad_time = pd.to_numeric(gad_main_df["time"], errors="coerce")

# -----------------------------
# CENTER CURVES
# -----------------------------
vis_mean_conc = get_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX, ["mean_conc"], required=False)
gad_mean_conc = get_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX, ["mean_conc"], required=False)

vis_d = get_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX, ["effective_diffusivity"], required=True)
gad_d = get_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX, ["effective_diffusivity"], required=True)

vis_cs = get_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX, ["fitted_Cs"], required=True)
gad_cs = get_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX, ["fitted_Cs"], required=True)

vis_rmse = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX, ["profile_fit_rmse"], vis_time, required=False)
gad_rmse = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX, ["profile_fit_rmse"], gad_time, required=False)

vis_r2 = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX, ["profile_fit_r2"], vis_time, required=False)
gad_r2 = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX, ["profile_fit_r2"], gad_time, required=False)

vis_flux = get_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX, ["mean_diffusive_flux_magnitude"], required=False)
gad_flux = get_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX, ["mean_diffusive_flux_magnitude"], required=False)

vis_reg_d = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                           ["regularized_effective_diffusivity_plot_mm2_s", "regularized_effective_diffusivity_mm2_s"],
                                           vis_time, required=False)
gad_reg_d = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                           ["regularized_effective_diffusivity_plot_mm2_s", "regularized_effective_diffusivity_mm2_s"],
                                           gad_time, required=False)

# -----------------------------
# COMBINED AND FIXED-ROI UNCERTAINTY
# fixed ROI = model fit + HU noise + calibration only (no ROI sensitivity)
# -----------------------------
# Mean concentration: no model-fit term available, so fixed ROI = HU noise + calibration
vis_mean_conc_combined_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                            ["mean_conc_combined_std", "mean_concentration_combined_std"],
                                                            vis_time, required=False)
gad_mean_conc_combined_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                            ["mean_conc_combined_std", "mean_concentration_combined_std"],
                                                            gad_time, required=False)
vis_mean_conc_hu_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                      ["mean_conc_hu_noise_std", "mean_concentration_hu_noise_std"], vis_time, required=False)
gad_mean_conc_hu_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                      ["mean_conc_hu_noise_std", "mean_concentration_hu_noise_std"], gad_time, required=False)
vis_mean_conc_cal_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                       ["mean_conc_calibration_std", "mean_concentration_calibration_std"], vis_time, required=False)
gad_mean_conc_cal_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                       ["mean_conc_calibration_std", "mean_concentration_calibration_std"], gad_time, required=False)
vis_mean_conc_fixed_std = combine_std_terms(vis_mean_conc_hu_std, vis_mean_conc_cal_std)
gad_mean_conc_fixed_std = combine_std_terms(gad_mean_conc_hu_std, gad_mean_conc_cal_std)

# Effective diffusivity
vis_d_model_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                 ["effective_diffusivity_std", "effective_diffusivity_std_mm2_s"], vis_time, required=False)
gad_d_model_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                 ["effective_diffusivity_std", "effective_diffusivity_std_mm2_s"], gad_time, required=False)
vis_d_hu_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                              ["effective_diffusivity_hu_noise_std", "effective_diffusivity_hu_noise_std_mm2_s"], vis_time, required=False)
gad_d_hu_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                              ["effective_diffusivity_hu_noise_std", "effective_diffusivity_hu_noise_std_mm2_s"], gad_time, required=False)
vis_d_cal_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                               ["effective_diffusivity_calibration_std", "effective_diffusivity_calibration_std_mm2_s"], vis_time, required=False)
gad_d_cal_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                               ["effective_diffusivity_calibration_std", "effective_diffusivity_calibration_std_mm2_s"], gad_time, required=False)
vis_d_combined_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                    ["effective_diffusivity_combined_std", "effective_diffusivity_combined_std_mm2_s"], vis_time, required=False)
gad_d_combined_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                    ["effective_diffusivity_combined_std", "effective_diffusivity_combined_std_mm2_s"], gad_time, required=False)
vis_d_fixed_std = combine_std_terms(vis_d_model_std, vis_d_hu_std, vis_d_cal_std)
gad_d_fixed_std = combine_std_terms(gad_d_model_std, gad_d_hu_std, gad_d_cal_std)

# Fitted Cs
vis_cs_model_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                  ["fitted_Cs_std"], vis_time, required=False)
gad_cs_model_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                  ["fitted_Cs_std"], gad_time, required=False)
vis_cs_hu_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                               ["fitted_Cs_hu_noise_std"], vis_time, required=False)
gad_cs_hu_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                               ["fitted_Cs_hu_noise_std"], gad_time, required=False)
vis_cs_cal_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                ["fitted_Cs_calibration_std"], vis_time, required=False)
gad_cs_cal_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                ["fitted_Cs_calibration_std"], gad_time, required=False)
vis_cs_combined_std = get_aligned_from_main_or_extra(vis_main_df, vis_extra_df, VIS_ROI_PREFIX,
                                                     ["fitted_Cs_combined_std"], vis_time, required=False)
gad_cs_combined_std = get_aligned_from_main_or_extra(gad_main_df, gad_extra_df, GAD_ROI_PREFIX,
                                                     ["fitted_Cs_combined_std"], gad_time, required=False)
vis_cs_fixed_std = combine_std_terms(vis_cs_model_std, vis_cs_hu_std, vis_cs_cal_std)
gad_cs_fixed_std = combine_std_terms(gad_cs_model_std, gad_cs_hu_std, gad_cs_cal_std)

# CI from std
vis_mean_conc_combined_ci_low, vis_mean_conc_combined_ci_high = ci_from_center_std(vis_mean_conc, vis_mean_conc_combined_std)
gad_mean_conc_combined_ci_low, gad_mean_conc_combined_ci_high = ci_from_center_std(gad_mean_conc, gad_mean_conc_combined_std)
vis_mean_conc_fixed_ci_low, vis_mean_conc_fixed_ci_high = ci_from_center_std(vis_mean_conc, vis_mean_conc_fixed_std)
gad_mean_conc_fixed_ci_low, gad_mean_conc_fixed_ci_high = ci_from_center_std(gad_mean_conc, gad_mean_conc_fixed_std)

vis_d_combined_ci_low, vis_d_combined_ci_high = ci_from_center_std(vis_d, vis_d_combined_std)
gad_d_combined_ci_low, gad_d_combined_ci_high = ci_from_center_std(gad_d, gad_d_combined_std)
vis_d_fixed_ci_low, vis_d_fixed_ci_high = ci_from_center_std(vis_d, vis_d_fixed_std)
gad_d_fixed_ci_low, gad_d_fixed_ci_high = ci_from_center_std(gad_d, gad_d_fixed_std)

vis_cs_combined_ci_low, vis_cs_combined_ci_high = ci_from_center_std(vis_cs, vis_cs_combined_std)
gad_cs_combined_ci_low, gad_cs_combined_ci_high = ci_from_center_std(gad_cs, gad_cs_combined_std)
vis_cs_fixed_ci_low, vis_cs_fixed_ci_high = ci_from_center_std(vis_cs, vis_cs_fixed_std)
gad_cs_fixed_ci_low, gad_cs_fixed_ci_high = ci_from_center_std(gad_cs, gad_cs_fixed_std)

# -----------------------------
# PLOTS: COMBINED 95% CI
# -----------------------------
make_band_plot(vis_mean_conc, gad_mean_conc, vis_time, gad_time,
               "Mean concentration (mg/mL)",
               "Mean Concentration vs Time: VIS 320 vs GAD (combined 95% CI)",
               "combined_95CI",
               "vis_vs_gad_mean_concentration_combined_95CI.png",
               vis_mean_conc_combined_ci_low, vis_mean_conc_combined_ci_high,
               gad_mean_conc_combined_ci_low, gad_mean_conc_combined_ci_high)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (combined 95% CI)",
               "combined_95CI",
               "vis_vs_gad_effective_diffusivity_combined_95CI.png",
               vis_d_combined_ci_low, vis_d_combined_ci_high,
               gad_d_combined_ci_low, gad_d_combined_ci_high,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (combined 95% CI, zoomed)",
               "combined_95CI",
               "vis_vs_gad_effective_diffusivity_combined_95CI_zoomed.png",
               vis_d_combined_ci_low, vis_d_combined_ci_high,
               gad_d_combined_ci_low, gad_d_combined_ci_high,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY_ZOOM)

make_band_plot(vis_cs, gad_cs, vis_time, gad_time,
               r"Effective fitted boundary concentration, C$_s$ (mg/mL)",
               "Effective Fitted Boundary Concentration vs Time: VIS 320 vs GAD (combined 95% CI)",
               "combined_95CI",
               "vis_vs_gad_effective_Cs_combined_95CI.png",
               vis_cs_combined_ci_low, vis_cs_combined_ci_high,
               gad_cs_combined_ci_low, gad_cs_combined_ci_high,
               ylim=YLIM_EFFECTIVE_CS)

# -----------------------------
# PLOTS: COMBINED ±1 SD
# -----------------------------
make_band_plot(vis_mean_conc, gad_mean_conc, vis_time, gad_time,
               "Mean concentration (mg/mL)",
               "Mean Concentration vs Time: VIS 320 vs GAD (combined ±1 SD)",
               "combined_1SD",
               "vis_vs_gad_mean_concentration_combined_1SD.png",
               vis_mean_conc - vis_mean_conc_combined_std if vis_mean_conc_combined_std is not None else None,
               vis_mean_conc + vis_mean_conc_combined_std if vis_mean_conc_combined_std is not None else None,
               gad_mean_conc - gad_mean_conc_combined_std if gad_mean_conc_combined_std is not None else None,
               gad_mean_conc + gad_mean_conc_combined_std if gad_mean_conc_combined_std is not None else None)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (combined ±1 SD)",
               "combined_1SD",
               "vis_vs_gad_effective_diffusivity_combined_1SD.png",
               vis_d - vis_d_combined_std if vis_d_combined_std is not None else None,
               vis_d + vis_d_combined_std if vis_d_combined_std is not None else None,
               gad_d - gad_d_combined_std if gad_d_combined_std is not None else None,
               gad_d + gad_d_combined_std if gad_d_combined_std is not None else None,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (combined ±1 SD, zoomed)",
               "combined_1SD",
               "vis_vs_gad_effective_diffusivity_combined_1SD_zoomed.png",
               vis_d - vis_d_combined_std if vis_d_combined_std is not None else None,
               vis_d + vis_d_combined_std if vis_d_combined_std is not None else None,
               gad_d - gad_d_combined_std if gad_d_combined_std is not None else None,
               gad_d + gad_d_combined_std if gad_d_combined_std is not None else None,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY_ZOOM)

make_band_plot(vis_cs, gad_cs, vis_time, gad_time,
               r"Effective fitted boundary concentration, C$_s$ (mg/mL)",
               "Effective Fitted Boundary Concentration vs Time: VIS 320 vs GAD (combined ±1 SD)",
               "combined_1SD",
               "vis_vs_gad_effective_Cs_combined_1SD.png",
               vis_cs - vis_cs_combined_std if vis_cs_combined_std is not None else None,
               vis_cs + vis_cs_combined_std if vis_cs_combined_std is not None else None,
               gad_cs - gad_cs_combined_std if gad_cs_combined_std is not None else None,
               gad_cs + gad_cs_combined_std if gad_cs_combined_std is not None else None,
               ylim=YLIM_EFFECTIVE_CS)

# -----------------------------
# PLOTS: FIXED ROI 95% CI (model fit + HU noise + calibration)
# -----------------------------
make_band_plot(vis_mean_conc, gad_mean_conc, vis_time, gad_time,
               "Mean concentration (mg/mL)",
               "Mean Concentration vs Time: VIS 320 vs GAD (fixed ROI 95% CI)",
               "fixedROI_95CI",
               "vis_vs_gad_mean_concentration_fixedROI_95CI.png",
               vis_mean_conc_fixed_ci_low, vis_mean_conc_fixed_ci_high,
               gad_mean_conc_fixed_ci_low, gad_mean_conc_fixed_ci_high)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (fixed ROI 95% CI)",
               "fixedROI_95CI",
               "vis_vs_gad_effective_diffusivity_fixedROI_95CI.png",
               vis_d_fixed_ci_low, vis_d_fixed_ci_high,
               gad_d_fixed_ci_low, gad_d_fixed_ci_high,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (fixed ROI 95% CI, zoomed)",
               "fixedROI_95CI",
               "vis_vs_gad_effective_diffusivity_fixedROI_95CI_zoomed.png",
               vis_d_fixed_ci_low, vis_d_fixed_ci_high,
               gad_d_fixed_ci_low, gad_d_fixed_ci_high,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY_ZOOM)

make_band_plot(vis_cs, gad_cs, vis_time, gad_time,
               r"Effective fitted boundary concentration, C$_s$ (mg/mL)",
               "Effective Fitted Boundary Concentration vs Time: VIS 320 vs GAD (fixed ROI 95% CI)",
               "fixedROI_95CI",
               "vis_vs_gad_effective_Cs_fixedROI_95CI.png",
               vis_cs_fixed_ci_low, vis_cs_fixed_ci_high,
               gad_cs_fixed_ci_low, gad_cs_fixed_ci_high,
               ylim=YLIM_EFFECTIVE_CS)

# -----------------------------
# PLOTS: FIXED ROI ±1 SD
# -----------------------------
make_band_plot(vis_mean_conc, gad_mean_conc, vis_time, gad_time,
               "Mean concentration (mg/mL)",
               "Mean Concentration vs Time: VIS 320 vs GAD (fixed ROI ±1 SD)",
               "fixedROI_1SD",
               "vis_vs_gad_mean_concentration_fixedROI_1SD.png",
               vis_mean_conc - vis_mean_conc_fixed_std if vis_mean_conc_fixed_std is not None else None,
               vis_mean_conc + vis_mean_conc_fixed_std if vis_mean_conc_fixed_std is not None else None,
               gad_mean_conc - gad_mean_conc_fixed_std if gad_mean_conc_fixed_std is not None else None,
               gad_mean_conc + gad_mean_conc_fixed_std if gad_mean_conc_fixed_std is not None else None)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (fixed ROI ±1 SD)",
               "fixedROI_1SD",
               "vis_vs_gad_effective_diffusivity_fixedROI_1SD.png",
               vis_d - vis_d_fixed_std if vis_d_fixed_std is not None else None,
               vis_d + vis_d_fixed_std if vis_d_fixed_std is not None else None,
               gad_d - gad_d_fixed_std if gad_d_fixed_std is not None else None,
               gad_d + gad_d_fixed_std if gad_d_fixed_std is not None else None,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY)

make_band_plot(vis_d, gad_d, vis_time, gad_time,
               r"Fitted effective diffusivity (mm$^2$/s)",
               "Effective Diffusivity vs Time: VIS 320 vs GAD (fixed ROI ±1 SD, zoomed)",
               "fixedROI_1SD",
               "vis_vs_gad_effective_diffusivity_fixedROI_1SD_zoomed.png",
               vis_d - vis_d_fixed_std if vis_d_fixed_std is not None else None,
               vis_d + vis_d_fixed_std if vis_d_fixed_std is not None else None,
               gad_d - gad_d_fixed_std if gad_d_fixed_std is not None else None,
               gad_d + gad_d_fixed_std if gad_d_fixed_std is not None else None,
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY_ZOOM)

make_band_plot(vis_cs, gad_cs, vis_time, gad_time,
               r"Effective fitted boundary concentration, C$_s$ (mg/mL)",
               "Effective Fitted Boundary Concentration vs Time: VIS 320 vs GAD (fixed ROI ±1 SD)",
               "fixedROI_1SD",
               "vis_vs_gad_effective_Cs_fixedROI_1SD.png",
               vis_cs - vis_cs_fixed_std if vis_cs_fixed_std is not None else None,
               vis_cs + vis_cs_fixed_std if vis_cs_fixed_std is not None else None,
               gad_cs - gad_cs_fixed_std if gad_cs_fixed_std is not None else None,
               gad_cs + gad_cs_fixed_std if gad_cs_fixed_std is not None else None,
               ylim=YLIM_EFFECTIVE_CS)

# -----------------------------
# Diagnostics (same current non-uncertainty measurements)
# -----------------------------
make_line_plot(vis_rmse, gad_rmse, vis_time, gad_time,
               "Profile-fit RMSE (mg/mL)",
               "Profile-Fit RMSE vs Time: VIS 320 vs GAD",
               "diagnostics",
               "vis_vs_gad_profile_fit_rmse.png")

make_line_plot(vis_r2, gad_r2, vis_time, gad_time,
               r"Profile-fit $R^2$",
               "Profile-Fit R² vs Time: VIS 320 vs GAD",
               "diagnostics",
               "vis_vs_gad_profile_fit_r2.png",
               ylim=YLIM_R2)

make_line_plot(vis_flux, gad_flux, vis_time, gad_time,
               r"Mean $|J_{diff}|$",
               "Mean Diffusive Flux Magnitude vs Time: VIS 320 vs GAD",
               "diagnostics",
               "vis_vs_gad_mean_diffusive_flux_magnitude.png")

make_line_plot(vis_reg_d, gad_reg_d, vis_time, gad_time,
               r"Temporally regularized effective diffusivity (mm$^2$/s)",
               "Temporally Regularized Effective Diffusivity vs Time: VIS 320 vs GAD",
               "diagnostics",
               "vis_vs_gad_temporally_regularized_effective_diffusivity.png",
               ylim=YLIM_EFFECTIVE_DIFFUSIVITY)

# -----------------------------
# Summary CSVs
# -----------------------------
def summarize_window(time_series: pd.Series,
                     center: Optional[pd.Series],
                     low: Optional[pd.Series],
                     high: Optional[pd.Series],
                     std: Optional[pd.Series],
                     label: str) -> dict:
    out = {"label": label}
    if center is None:
        out.update({"n_points": 0, "mean": pd.NA, "std_of_center": pd.NA,
                    "uncertainty_std_mean": pd.NA, "low_mean": pd.NA, "high_mean": pd.NA})
        return out

    mask = time_series >= SUMMARY_MIN_TIME_MIN
    if SUMMARY_MAX_TIME_MIN is not None:
        mask &= (time_series <= SUMMARY_MAX_TIME_MIN)

    vals = center[mask].dropna()
    out["n_points"] = int(vals.shape[0])
    out["mean"] = float(vals.mean()) if len(vals) else pd.NA
    out["std_of_center"] = float(vals.std(ddof=1)) if len(vals) > 1 else pd.NA

    if std is not None:
        s = std[mask].dropna()
        out["uncertainty_std_mean"] = float(s.mean()) if len(s) else pd.NA
    else:
        out["uncertainty_std_mean"] = pd.NA

    if low is not None:
        lows = low[mask].dropna()
        out["low_mean"] = float(lows.mean()) if len(lows) else pd.NA
    else:
        out["low_mean"] = pd.NA

    if high is not None:
        highs = high[mask].dropna()
        out["high_mean"] = float(highs.mean()) if len(highs) else pd.NA
    else:
        out["high_mean"] = pd.NA
    return out

summary_rows = [
    summarize_window(vis_time, vis_d, vis_d_combined_ci_low, vis_d_combined_ci_high, vis_d_combined_std, "VIS_320_effective_diffusivity_combined"),
    summarize_window(gad_time, gad_d, gad_d_combined_ci_low, gad_d_combined_ci_high, gad_d_combined_std, "GAD_effective_diffusivity_combined"),
    summarize_window(vis_time, vis_d, vis_d_fixed_ci_low, vis_d_fixed_ci_high, vis_d_fixed_std, "VIS_320_effective_diffusivity_fixedROI"),
    summarize_window(gad_time, gad_d, gad_d_fixed_ci_low, gad_d_fixed_ci_high, gad_d_fixed_std, "GAD_effective_diffusivity_fixedROI"),
    summarize_window(vis_time, vis_cs, vis_cs_combined_ci_low, vis_cs_combined_ci_high, vis_cs_combined_std, "VIS_320_fitted_Cs_combined"),
    summarize_window(gad_time, gad_cs, gad_cs_combined_ci_low, gad_cs_combined_ci_high, gad_cs_combined_std, "GAD_fitted_Cs_combined"),
    summarize_window(vis_time, vis_cs, vis_cs_fixed_ci_low, vis_cs_fixed_ci_high, vis_cs_fixed_std, "VIS_320_fitted_Cs_fixedROI"),
    summarize_window(gad_time, gad_cs, gad_cs_fixed_ci_low, gad_cs_fixed_ci_high, gad_cs_fixed_std, "GAD_fitted_Cs_fixedROI"),
    summarize_window(vis_time, vis_mean_conc, vis_mean_conc_combined_ci_low, vis_mean_conc_combined_ci_high, vis_mean_conc_combined_std, "VIS_320_mean_concentration_combined"),
    summarize_window(gad_time, gad_mean_conc, gad_mean_conc_combined_ci_low, gad_mean_conc_combined_ci_high, gad_mean_conc_combined_std, "GAD_mean_concentration_combined"),
    summarize_window(vis_time, vis_mean_conc, vis_mean_conc_fixed_ci_low, vis_mean_conc_fixed_ci_high, vis_mean_conc_fixed_std, "VIS_320_mean_concentration_fixedROI"),
    summarize_window(gad_time, gad_mean_conc, gad_mean_conc_fixed_ci_low, gad_mean_conc_fixed_ci_high, gad_mean_conc_fixed_std, "GAD_mean_concentration_fixedROI"),
]
summary_df = pd.DataFrame(summary_rows)
summary_dir = Path(OUT_FOLDER) / "summaries"
summary_dir.mkdir(parents=True, exist_ok=True)
summary_df.to_csv(summary_dir / "paper_1_late_time_summary_all_uncertainties.csv", index=False)


audit_payload = {
    "results_source": RESULTS_SOURCE,
    "vis_selection_mode": vis_selection_mode,
    "gad_selection_mode": gad_selection_mode,
    "vis_main_csv": vis_main_rel,
    "vis_extra_csv": vis_extra_rel,
    "gad_main_csv": gad_main_rel,
    "gad_extra_csv": gad_extra_rel,
    "vis_roi_prefix": VIS_ROI_PREFIX,
    "gad_roi_prefix": GAD_ROI_PREFIX,
    "vis_main_columns": vis_main_df.columns.tolist(),
    "vis_extra_columns": vis_extra_df.columns.tolist(),
    "gad_main_columns": gad_main_df.columns.tolist(),
    "gad_extra_columns": gad_extra_df.columns.tolist(),
}
save_audit_report(OUT_FOLDER, audit_payload)

print("Saved audit files to:", str(Path(OUT_FOLDER) / "audit"))
print("Done.")
'''
