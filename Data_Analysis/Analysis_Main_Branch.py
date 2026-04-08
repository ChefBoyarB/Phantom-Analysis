import os
import glob
import json
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import PchipInterpolator
from scipy.special import erfc
from scipy.stats import norm
import pydicom



def format_roi_display_name(roi_name: str) -> str:
    """Return a cleaner ROI name for plot titles/legends without changing internal IDs."""
    if roi_name is None:
        return "ROI"
    return " ".join(part for part in str(roi_name).replace("_", " ").split() if part)


# ============================================================
# USER SETTINGS
# ============================================================

DICOM_FOLDER = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\GAD_1_5.5P_2\GAD_1_5.5P_2"
OUTPUT_FOLDER = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Testing\GAD_Pressure_Test_Global_Shared_V_CODEX" # make sure to change this for each run to avoid overwriting previous outputs

# ------------------------------------------------------------
# Analysis mode
# ------------------------------------------------------------
PUMP_ON = True  # if False, assumes pure diffusion and uses diffusion-only model; if True, uses ADE model with convection

# If PUMP_ON is True:
#   FIT_VELOCITY = True  -> fit D, Cs, and v from each profile
#   FIT_VELOCITY = False -> compute/fix v from flow or Darcy and fit only D, Cs
FIT_VELOCITY = False

# velocity source if not fitting v
CONVECTION_METHOD = "darcy"    # "flow" or "darcy" or "none" for pure diffusion

# ------------------------------------------------------------
# HU -> concentration calibration
# absolute concentration = max((HU - HU_OFFSET), 0) / HU_PER_CONC
# no baseline subtraction (disabled in this absolute-concentration version) is applied in this version
# ------------------------------------------------------------
USE_FIRST_TIMEPOINT_AS_BASELINE = False  # Absolute-concentration fitting; no baseline subtraction (disabled in this absolute-concentration version)
MANUAL_BASELINE_HU = None
HU_PER_CONC = 8.681 # Need to change this based on calibration data, e.g. 8.681 HU per mg/mL for GAD at 135 kVp
HU_OFFSET = 12.705 # Need to change this based on calibration data, e.g. 12.271 HU offset for GAD at 135 kVp
# Leave these at 0.0 if the calibration slope/intercept uncertainty is unknown.
# The code will still run normally; calibration uncertainty will simply contribute 0.
HU_PER_CONC_STD = 0.0069 # Need to change this based on calibration data, e.g. 0.0117 HU per (mg/mL) std for GAD at 135 kVp
HU_OFFSET_STD = 0.176 # Need to change this based on calibration data, e.g. 0.2982 HU std for GAD at 135 kVp
CONCENTRATION_UNITS = "mg/mL"

# ------------------------------------------------------------
# Additional uncertainty settings
# ------------------------------------------------------------
ENABLE_HU_NOISE_UNCERTAINTY = True
DEEP_REGION_FRACTION_FOR_HU_NOISE = 0.10   # deepest fraction of the selected ROI used to estimate HU noise
HU_NOISE_MONTE_CARLO_SAMPLES = 10          # refit count for image-noise uncertainty propagation
ENABLE_ROI_SENSITIVITY_UNCERTAINTY = True
ROI_SENSITIVITY_SHIFTS = [-1, 0, 1]         # shift ROI along depth direction, preserving ROI size
UNCERTAINTY_RANDOM_SEED = 42

# ------------------------------------------------------------
# Flow-based convection
# v = Q / A
# ------------------------------------------------------------
FLOW_RATE_ML_MIN = 0.0
EFFECTIVE_AREA_MM2 = 75.0

# ------------------------------------------------------------
# Darcy-based convection
# v = -(k/mu) * dP/dx
# ------------------------------------------------------------
PERMEABILITY_MM2 = 3.7e-9
VISCOSITY_PA_S = 0.001
PRESSURE_GRADIENT_PA_MM = -25.24

# ------------------------------------------------------------
# Depth direction
# "rows" = depth goes top-to-bottom
# "cols" = depth goes left-to-right
# ------------------------------------------------------------
DEPTH_AXIS = "rows"

# ------------------------------------------------------------
# ROI selection
# ------------------------------------------------------------
# ROI_SELECTION_MODE options:
#   "interactive"   -> draw ROIs with the mouse
#   "manual_list"   -> reuse exact saved coordinates from MANUAL_NAMED_ROIS
#   "manual_prompt" -> type ROI coordinates into the terminal during the run 
ROI_SELECTION_MODE = "manual_list"

MANUAL_NAMED_ROIS = [[
      "GAD_Darcy_Fixed_V",
      [
        146,
        165,
        277,
        313
      ]
    ]
      
  ]
    # Coordinates are stored as: (r0, r1, c0, c1)
    # Example:
    # ("Chamber1_Center", (50, 180, 50, 160)),
    # ("Chamber1_Edge",   (60, 190, 180, 300)),

# Saves the selected ROIs from each run so they can be copied directly into
# MANUAL_NAMED_ROIS for exact reruns later.
SAVE_SELECTED_ROIS_JSON = True

# ------------------------------------------------------------
# Frame exclusion
# ------------------------------------------------------------
# Set to 0 to keep all frames.
# Set to 2, for example, to skip the first two frames before analysis.
SKIP_INITIAL_FRAMES = 2

# ------------------------------------------------------------
# Smoothing
# ------------------------------------------------------------
APPLY_SPATIAL_SMOOTHING = False
SPATIAL_SMOOTH_SIGMA = 0.8

APPLY_TEMPORAL_SMOOTHING = False
TEMPORAL_SMOOTH_SIGMA = 0.8

APPLY_DEPTH_PROFILE_SMOOTHING = False
DEPTH_PROFILE_SMOOTH_SIGMA = 1.0

# ------------------------------------------------------------
# Local-only smoothing for local effective diffusion map
# This leaves the main fits, plots, maps, and summary stats unchanged.
# ------------------------------------------------------------
LOCAL_MAP_APPLY_DEPTH_SMOOTHING = True
LOCAL_MAP_DEPTH_SMOOTH_SIGMA = 1.0

LOCAL_MAP_APPLY_TEMPORAL_SMOOTHING = False
LOCAL_MAP_TEMPORAL_SMOOTH_SIGMA = 0.8

# Local-map-only fit controls
# These settings affect ONLY the sliding-window local effective diffusion map.
LOCAL_MAP_FIT_PROFILE_THRESHOLD_FRACTION = 0.05
LOCAL_MAP_D_BOUNDS = (1e-6, 0.01)

# ------------------------------------------------------------
# Fitting control
# ------------------------------------------------------------
MIN_TIME_SECONDS_FOR_FIT = 30.0
MIN_VALID_POINTS_FOR_FIT = 3
FIT_ONLY_POSITIVE_CONCENTRATION = False
FIT_PROFILE_THRESHOLD_FRACTION = 0.03   # keep points above 5% of profile max
MAX_CS_FACTOR = 5.0                     # upper bound for Cs as factor of max profile concentration

# Diffusion bounds [mm^2/s]
D_BOUNDS = (1e-6, 0.01)

# Velocity bounds [mm/s] if fitting v
V_BOUNDS = (0.0, 3e-4)

# ------------------------------------------------------------
# Pixelwise map
# ------------------------------------------------------------
SAVE_PIXELWISE_APPARENT_DIFFUSION_MAP = False

# ------------------------------------------------------------
# Local effective diffusion map (sliding-window analytical fits)
# ------------------------------------------------------------
SAVE_EFFECTIVE_DIFFUSION_MAP = True
LOCAL_EFFECTIVE_MAP_WINDOW_SIZE = 13   # odd integer: 9, 11, 13 are good starting points

# ------------------------------------------------------------
# Derived effective diffusion map from the full fitted concentration field
# ------------------------------------------------------------
SAVE_DERIVED_EFFECTIVE_DIFFUSION_MAP = True
DERIVED_EFFECTIVE_D_BOUNDS = (1e-6, 0.01)
DERIVED_EFFECTIVE_CURVATURE_EPS = 1e-6


# ------------------------------------------------------------
# Global spatiotemporal fit
# ------------------------------------------------------------
ENABLE_GLOBAL_SPATIOTEMPORAL_FIT = True
GLOBAL_CS_NUM_CONTROL_POINTS = 6
GLOBAL_FIT_INCLUDE_VELOCITY = True  # only used if PUMP_ON and CONVECTION_METHOD != "none"
GLOBAL_FIT_MAX_NFEV = 50000

# ------------------------------------------------------------
# Temporally regularized per-timepoint fit
# Fits one profile per timepoint, but solves all timepoints together with
# temporal smoothness penalties so D(t) and Cs(t) do not jump frame-to-frame.
# ------------------------------------------------------------
ENABLE_TEMPORALLY_REGULARIZED_FIT = True
REGULARIZED_FIT_MAX_NFEV = 50000
# Penalty weights on second temporal differences; larger = smoother.
REGULARIZED_FIT_LAMBDA_D = 2.0
REGULARIZED_FIT_LAMBDA_CS = 1.0
# Soft penalty discouraging non-monotonic Cs(t) in pump-off diffusion runs.
REGULARIZED_FIT_LAMBDA_CS_MONOTONIC = 2.0

# ------------------------------------------------------------
# Plot time unit
# ------------------------------------------------------------
TIME_UNIT = "minutes"   # "seconds" or "minutes"

# ------------------------------------------------------------
# Plot orientation
# ------------------------------------------------------------
PLOT_DEPTH_ZERO_AT_TOP = True  # True = 0 mm at top, max depth at bottom

# ------------------------------------------------------------
# Convection-vs-diffusion summary metrics
# ------------------------------------------------------------
PECLET_LENGTH_MODE = "penetration_depth"   # "penetration_depth" or "roi_depth"
PECLET_THRESHOLD_FRACTION = 0.10            # used only for penetration-depth based Pe
PECLET_MIN_LENGTH_MM = 1e-6
TRANSPORT_EPS = 1e-12


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DicomFrame:
    filepath: str
    slice_location: float
    time_seconds: float
    image_hu: np.ndarray
    row_spacing_mm: float
    col_spacing_mm: float
    instance_number: int


# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_get(ds, name, default=None):
    return getattr(ds, name, default)


def dicom_time_to_seconds(ds) -> float:
    for attr in ["AcquisitionTime", "ContentTime", "SeriesTime"]:
        value = safe_get(ds, attr, None)
        if value is not None:
            try:
                t = str(value).split(".")[0].zfill(6)
                hh = int(t[0:2])
                mm = int(t[2:4])
                ss = int(t[4:6])
                return hh * 3600 + mm * 60 + ss
            except Exception:
                pass
    return float(safe_get(ds, "InstanceNumber", 0))


def get_slice_location(ds) -> float:
    sl = safe_get(ds, "SliceLocation", None)
    if sl is not None:
        return float(sl)

    ipp = safe_get(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        return float(ipp[2])

    raise ValueError("Could not determine slice location.")


def normalize_time(times_seconds: np.ndarray, unit: str = "minutes") -> np.ndarray:
    t0 = np.min(times_seconds)
    dt = times_seconds - t0
    if unit == "minutes":
        return dt / 60.0
    return dt


def save_figure(fig, out_path: str, dpi: int = 300):
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def sanitize_name(name: str) -> str:
    name = name.strip().replace(" ", "_")
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
    name = "".join(c for c in name if c in allowed)
    return name


def build_roi_output_dirs(roi_folder: str) -> dict:
    subdirs = {
        "root": roi_folder,
        "profiles": os.path.join(roi_folder, "Profiles"),
        "maps_diffusion": os.path.join(roi_folder, "Maps_Diffusion"),
        "maps_convection": os.path.join(roi_folder, "Maps_Convection"),
        "maps_uncertainty": os.path.join(roi_folder, "Maps_Uncertainty"),
        "maps_other": os.path.join(roi_folder, "Maps_Other"),
        "timecourses": os.path.join(roi_folder, "Timecourse_Plots"),
        "csv_profiles": os.path.join(roi_folder, "CSVs_Profiles"),
        "csv_diffusion": os.path.join(roi_folder, "CSVs_Diffusion"),
        "csv_convection": os.path.join(roi_folder, "CSVs_Convection"),
        "csv_uncertainty": os.path.join(roi_folder, "CSVs_Uncertainty"),
        "csv_summaries": os.path.join(roi_folder, "CSVs_Summaries"),
        "csv_other": os.path.join(roi_folder, "CSVs_Other"),
    }
    for path in subdirs.values():
        ensure_dir(path)
    return subdirs


def roi_output_path(output_dirs: dict, filename: str) -> str:
    name = os.path.basename(filename).lower()

    if name.endswith(".png"):
        if "profile_fit_examples" in name or "3d" in name:
            bucket = "profiles"
        elif "vs_time" in name:
            bucket = "timecourses"
        elif any(k in name for k in ["uncertainty", "std", "_ci_", "ci_", "relative_ci", "confidence"]):
            bucket = "maps_uncertainty"
        elif any(k in name for k in ["convection", "peclet", "velocity"]):
            bucket = "maps_convection"
        elif any(k in name for k in ["diffusion", "flux", "fitted_profiles_map", "apparent_diffusion", "local_effective"]):
            bucket = "maps_diffusion"
        else:
            bucket = "maps_other"
    elif name.endswith(".csv"):
        if any(k in name for k in ["uncertainty", "std", "_ci_", "ci_", "relative_ci", "residual"]):
            bucket = "csv_uncertainty"
        elif any(k in name for k in ["convection", "peclet", "velocity"]):
            bucket = "csv_convection"
        elif any(k in name for k in ["diffusion", "flux", "local_effective", "apparent_diffusion"]):
            bucket = "csv_diffusion"
        elif any(k in name for k in ["profile", "measured_profiles", "fitted_profiles"]):
            bucket = "csv_profiles"
        elif any(k in name for k in ["summary", "parameters", "control_points"]):
            bucket = "csv_summaries"
        else:
            bucket = "csv_other"
    else:
        bucket = "root"

    return os.path.join(output_dirs[bucket], filename)


def nan_safe_median(a):
    return np.nan if np.all(np.isnan(a)) else np.nanmedian(a)


def nan_safe_mean(a):
    return np.nan if np.all(np.isnan(a)) else np.nanmean(a)

def nan_safe_std(a):
    return np.nan if np.all(np.isnan(a)) else np.nanstd(a)

def nan_safe_iqr(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    q25, q75 = np.nanpercentile(a, [25, 75])
    return q75 - q25

def nan_safe_mad(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return np.nan
    med = np.nanmedian(a)
    return np.nanmedian(np.abs(a - med))


# ============================================================
# LOAD DICOMS
# ============================================================

def load_dicom_frames(folder: str) -> List[DicomFrame]:
    dicom_files = glob.glob(os.path.join(folder, "**", "*.dcm"), recursive=True)
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in: {folder}")

    frames = []
    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp)

            raw = ds.pixel_array.astype(np.float32)
            slope = float(safe_get(ds, "RescaleSlope", 1.0))
            intercept = float(safe_get(ds, "RescaleIntercept", 0.0))
            hu = raw * slope + intercept

            slice_loc = get_slice_location(ds)
            time_sec = dicom_time_to_seconds(ds)

            spacing = safe_get(ds, "PixelSpacing", [1.0, 1.0])
            row_spacing = float(spacing[0])
            col_spacing = float(spacing[1])

            inst_num = int(safe_get(ds, "InstanceNumber", 0))

            frames.append(
                DicomFrame(
                    filepath=fp,
                    slice_location=slice_loc,
                    time_seconds=time_sec,
                    image_hu=hu,
                    row_spacing_mm=row_spacing,
                    col_spacing_mm=col_spacing,
                    instance_number=inst_num
                )
            )
        except Exception as e:
            print(f"Skipping {fp}: {e}")

    if not frames:
        raise RuntimeError("No readable DICOM files found.")

    return frames


def group_frames_by_slice(frames: List[DicomFrame], tol: float = 1e-3) -> Dict[float, List[DicomFrame]]:
    groups: Dict[float, List[DicomFrame]] = {}
    unique_keys = []

    for fr in frames:
        matched_key = None
        for key in unique_keys:
            if abs(fr.slice_location - key) < tol:
                matched_key = key
                break

        if matched_key is None:
            unique_keys.append(fr.slice_location)
            groups[fr.slice_location] = [fr]
        else:
            groups[matched_key].append(fr)

    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x.time_seconds)

    return groups


# ============================================================
# ROI SELECTION
# ============================================================

class SingleROISelector:
    def __init__(self, image: np.ndarray, title: str = "Draw ROI and close window"):
        self.image = image
        self.roi = None
        self.title = title

    def select(self):
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(self.image, cmap="gray")
        ax.set_title(self.title)

        def onselect(eclick, erelease):
            if eclick.xdata is None or eclick.ydata is None or erelease.xdata is None or erelease.ydata is None:
                return
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)

            r0, r1 = sorted([y1, y2])
            c0, c1 = sorted([x1, x2])
            self.roi = (r0, r1, c0, c1)

        self.rs = RectangleSelector(
            ax,
            onselect,
            useblit=False,
            button=[1],
            minspanx=5,
            minspany=5,
            spancoords="pixels",
            interactive=True
        )
        plt.show()
        return self.roi


def collect_named_rois(image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    named_rois = []
    count = 1

    while True:
        selector = SingleROISelector(image, title=f"Draw ROI #{count}, then close window")
        roi = selector.select()

        if roi is None:
            ans = input("No ROI captured. Try again? (y/n): ").strip().lower()
            if ans == "y":
                continue
            break

        name = input(f"Enter name for ROI #{count} (leave blank for default ROI_{count}): ").strip()

        if name == "":
            name = f"ROI_{count}"
        else:
            name = sanitize_name(name)
            if name == "":
                name = f"ROI_{count}"

        existing_names = [n for n, _ in named_rois]
        original_name = name
        suffix = 2
        while name in existing_names:
            name = f"{original_name}_{suffix}"
            suffix += 1

        named_rois.append((name, roi))
        print(f"Captured {name}: {roi}")

        ans = input("Add another ROI? (y/n): ").strip().lower()
        if ans != "y":
            break

        count += 1

    return named_rois


def show_all_rois_overlay(image: np.ndarray,
                          named_rois: List[Tuple[str, Tuple[int, int, int, int]]],
                          out_path: Optional[str] = None):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray")

    for name, roi in named_rois:
        r0, r1, c0, c1 = roi
        rect = plt.Rectangle((c0, r0), c1 - c0, r1 - r0,
                             fill=False, edgecolor="red", linewidth=2)
        ax.add_patch(rect)
        ax.text(c0, max(r0 - 3, 0), name, color="yellow", fontsize=10, weight="bold")

    ax.set_title("All Selected ROIs")

    if out_path:
        save_figure(fig, out_path)
    else:
        plt.show()


def clamp_and_validate_roi(roi: Tuple[int, int, int, int], image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    rows, cols = image_shape
    r0, r1, c0, c1 = [int(v) for v in roi]

    r0 = max(0, min(r0, rows - 1))
    r1 = max(0, min(r1, rows))
    c0 = max(0, min(c0, cols - 1))
    c1 = max(0, min(c1, cols))

    if r1 <= r0 or c1 <= c0:
        raise ValueError(f"Invalid ROI after clamping: {(r0, r1, c0, c1)}")

    return (r0, r1, c0, c1)


def parse_roi_coordinate_string(coord_text: str) -> Tuple[int, int, int, int]:
    cleaned = coord_text.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [p.strip() for p in cleaned.split(",") if p.strip() != ""]
    if len(parts) != 4:
        raise ValueError("ROI coordinates must contain exactly 4 integers: r0, r1, c0, c1")
    return tuple(int(float(p)) for p in parts)


def collect_named_rois_manual_input(image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    named_rois = []
    count = 1
    rows, cols = image.shape[:2]

    print("\nManual ROI entry mode selected.")
    print(f"Image size: rows = {rows}, cols = {cols}")
    print("Enter ROI coordinates as: r0, r1, c0, c1")
    print("Example: 50, 180, 50, 160")

    while True:
        default_name = f"ROI_{count}"
        name = input(f"Enter name for ROI #{count} (leave blank for {default_name}): ").strip()
        if name == "":
            name = default_name
        else:
            name = sanitize_name(name) or default_name

        coord_text = input(f"Enter coordinates for {name} as r0, r1, c0, c1: ").strip()
        try:
            roi = parse_roi_coordinate_string(coord_text)
            roi = clamp_and_validate_roi(roi, (rows, cols))
        except Exception as e:
            print(f"Invalid ROI entry: {e}")
            retry = input("Try this ROI again? (y/n): ").strip().lower()
            if retry == "y":
                continue
            break

        existing_names = [n for n, _ in named_rois]
        original_name = name
        suffix = 2
        while name in existing_names:
            name = f"{original_name}_{suffix}"
            suffix += 1

        named_rois.append((name, roi))
        print(f"Captured {name}: {roi}")

        ans = input("Add another ROI? (y/n): ").strip().lower()
        if ans != "y":
            break
        count += 1

    return named_rois


def get_named_rois_from_settings(image: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    rows, cols = image.shape[:2]
    named_rois = []
    for name, roi in MANUAL_NAMED_ROIS:
        clean_name = sanitize_name(name) or "ROI"
        clean_roi = clamp_and_validate_roi(tuple(roi), (rows, cols))
        named_rois.append((clean_name, clean_roi))
    return named_rois


def save_selected_rois_json(output_root: str, named_rois: List[Tuple[str, Tuple[int, int, int, int]]]) -> Optional[str]:
    if not SAVE_SELECTED_ROIS_JSON:
        return None

    payload = {
        "roi_selection_mode_used": ROI_SELECTION_MODE,
        "manual_named_rois_ready_to_copy": [
            [name, [int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])]]
            for name, roi in named_rois
        ],
    }
    out_path = os.path.join(output_root, "selected_rois_for_rerun.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_path


# ============================================================
# STACK / CONCENTRATION PROCESSING
# ============================================================

def hu_to_concentration(hu_stack: np.ndarray,
                        baseline_hu_image: Optional[np.ndarray],
                        hu_per_conc: float,
                        hu_offset: Optional[float] = None) -> np.ndarray:
    # Absolute-concentration conversion using the calibration values passed in.
    if hu_offset is None:
        hu_offset = HU_OFFSET
    if hu_per_conc is None or float(hu_per_conc) == 0.0:
        raise ValueError("hu_per_conc must be non-zero for HU-to-concentration conversion")
    conc = (hu_stack - float(hu_offset)) / float(hu_per_conc)
    conc = np.asarray(conc, dtype=float)
    conc[conc < 0] = 0
    return conc



def estimate_calibration_uncertainty_via_refits(hu_stack: np.ndarray,
                                                roi: Tuple[int, int, int, int],
                                                times_sec: np.ndarray,
                                                dx_mm: float,
                                                depth_axis: str,
                                                hu_per_conc: float = HU_PER_CONC,
                                                hu_offset: float = HU_OFFSET,
                                                hu_per_conc_std: float = HU_PER_CONC_STD,
                                                hu_offset_std: float = HU_OFFSET_STD) -> Dict[str, np.ndarray]:
    """
    Estimate standalone calibration uncertainty by perturbing the calibration slope
    and intercept by ±1 SD, recomputing concentration, and refitting.

    This gives separate calibration-only uncertainty components for mean concentration,
    fitted effective diffusivity, fitted C_s, fitted velocity, and fitted profiles.
    """
    hu_per_conc = float(hu_per_conc)
    hu_offset = float(hu_offset)
    hu_per_conc_std = float(hu_per_conc_std)
    hu_offset_std = float(hu_offset_std)

    base_conc = hu_to_concentration(hu_stack, None, hu_per_conc, hu_offset)
    base_profiles = compute_depth_profiles(base_conc, roi, depth_axis)
    n_t, n_x = base_profiles.shape

    out = {
        "mean_conc_std": np.full(n_t, np.nan, dtype=float),
        "D_std": np.full(n_t, np.nan, dtype=float),
        "Cs_std": np.full(n_t, np.nan, dtype=float),
        "v_std": np.full(n_t, np.nan, dtype=float),
        "fitted_profiles_std": np.full((n_t, n_x), np.nan, dtype=float),
    }

    if hu_per_conc_std <= 0 and hu_offset_std <= 0:
        out["mean_conc_std"][:] = 0.0
        out["D_std"][:] = 0.0
        out["Cs_std"][:] = 0.0
        out["v_std"][:] = 0.0
        out["fitted_profiles_std"][:] = 0.0
        return out

    def _run_variant(hpc: float, hoff: float):
        conc_variant = hu_to_concentration(hu_stack, None, hpc, hoff)
        profiles_variant = compute_depth_profiles(conc_variant, roi, depth_axis)
        depth_mm_variant = np.arange(profiles_variant.shape[1]) * dx_mm
        fit_variant = fit_profiles_over_time(depth_mm_variant, profiles_variant, times_sec)
        r0, r1, c0, c1 = roi
        mean_variant = np.mean(conc_variant[:, r0:r1, c0:c1], axis=(1, 2))
        return {
            "mean_conc": mean_variant,
            "D": fit_variant[0],
            "Cs": fit_variant[1],
            "v": fit_variant[2],
            "fitted_profiles": fit_variant[3],
        }

    component_stds = []

    if hu_per_conc_std > 0:
        hpc_plus = max(hu_per_conc + hu_per_conc_std, 1e-12)
        hpc_minus = max(hu_per_conc - hu_per_conc_std, 1e-12)
        res_plus = _run_variant(hpc_plus, hu_offset)
        res_minus = _run_variant(hpc_minus, hu_offset)
        component_stds.append({
            "mean_conc_std": 0.5 * np.abs(res_plus["mean_conc"] - res_minus["mean_conc"]),
            "D_std": 0.5 * np.abs(res_plus["D"] - res_minus["D"]),
            "Cs_std": 0.5 * np.abs(res_plus["Cs"] - res_minus["Cs"]),
            "v_std": 0.5 * np.abs(res_plus["v"] - res_minus["v"]),
            "fitted_profiles_std": 0.5 * np.abs(res_plus["fitted_profiles"] - res_minus["fitted_profiles"]),
        })

    if hu_offset_std > 0:
        res_plus = _run_variant(hu_per_conc, hu_offset + hu_offset_std)
        res_minus = _run_variant(hu_per_conc, hu_offset - hu_offset_std)
        component_stds.append({
            "mean_conc_std": 0.5 * np.abs(res_plus["mean_conc"] - res_minus["mean_conc"]),
            "D_std": 0.5 * np.abs(res_plus["D"] - res_minus["D"]),
            "Cs_std": 0.5 * np.abs(res_plus["Cs"] - res_minus["Cs"]),
            "v_std": 0.5 * np.abs(res_plus["v"] - res_minus["v"]),
            "fitted_profiles_std": 0.5 * np.abs(res_plus["fitted_profiles"] - res_minus["fitted_profiles"]),
        })

    if component_stds:
        out["mean_conc_std"] = combine_uncertainty_terms(*[c["mean_conc_std"] for c in component_stds])
        out["D_std"] = combine_uncertainty_terms(*[c["D_std"] for c in component_stds])
        out["Cs_std"] = combine_uncertainty_terms(*[c["Cs_std"] for c in component_stds])
        out["v_std"] = combine_uncertainty_terms(*[c["v_std"] for c in component_stds])
        out["fitted_profiles_std"] = combine_uncertainty_terms(*[c["fitted_profiles_std"] for c in component_stds])

    return out


def shift_roi_along_depth(roi: Tuple[int, int, int, int], shift: int, image_shape: Tuple[int, int], depth_axis: str) -> Tuple[int, int, int, int]:
    r0, r1, c0, c1 = [int(v) for v in roi]
    n_rows, n_cols = image_shape
    if depth_axis == "rows":
        height = r1 - r0
        new_r0 = max(0, min(r0 + shift, n_rows - height))
        return (new_r0, new_r0 + height, c0, c1)
    width = c1 - c0
    new_c0 = max(0, min(c0 + shift, n_cols - width))
    return (r0, r1, new_c0, new_c0 + width)


def estimate_hu_noise_from_deep_region(hu_stack: np.ndarray,
                                       roi: Tuple[int, int, int, int],
                                       depth_axis: str,
                                       deep_fraction: float = DEEP_REGION_FRACTION_FOR_HU_NOISE) -> float:
    r0, r1, c0, c1 = roi
    sub = np.asarray(hu_stack[:, r0:r1, c0:c1], dtype=float)
    if sub.size == 0:
        return 0.0

    if depth_axis == "rows":
        n_depth = sub.shape[1]
        band = max(1, int(np.ceil(n_depth * float(deep_fraction))))
        deep_sub = sub[:, -band:, :]
    else:
        n_depth = sub.shape[2]
        band = max(1, int(np.ceil(n_depth * float(deep_fraction))))
        deep_sub = sub[:, :, -band:]

    if deep_sub.shape[0] < 2:
        sigma_hu = float(np.nanstd(deep_sub))
        return sigma_hu if np.isfinite(sigma_hu) else 0.0

    diffs = np.diff(deep_sub, axis=0) / np.sqrt(2.0)
    sigma_hu = float(np.nanstd(diffs))
    if not np.isfinite(sigma_hu):
        sigma_hu = float(np.nanstd(deep_sub)) if np.any(np.isfinite(deep_sub)) else 0.0
    return sigma_hu if np.isfinite(sigma_hu) else 0.0


def estimate_noise_uncertainty_via_refits(depth_mm: np.ndarray,
                                          profiles: np.ndarray,
                                          times_sec: np.ndarray,
                                          sigma_conc: float,
                                          n_samples: int = HU_NOISE_MONTE_CARLO_SAMPLES,
                                          seed: int = UNCERTAINTY_RANDOM_SEED) -> Dict[str, np.ndarray]:
    n_t, n_x = profiles.shape
    out = {
        "D_std": np.full(n_t, np.nan, dtype=float),
        "Cs_std": np.full(n_t, np.nan, dtype=float),
        "v_std": np.full(n_t, np.nan, dtype=float),
        "mean_conc_std": np.full(n_t, np.nan, dtype=float),
        "fitted_profiles_std": np.full((n_t, n_x), np.nan, dtype=float),
    }
    if sigma_conc <= 0 or n_samples <= 0:
        return out

    rng = np.random.default_rng(seed)
    D_samples, Cs_samples, v_samples, mean_samples, fitted_samples = [], [], [], [], []
    for _ in range(int(n_samples)):
        noisy_profiles = np.asarray(profiles, dtype=float) + rng.normal(0.0, sigma_conc, size=profiles.shape)
        noisy_profiles = np.clip(noisy_profiles, 0.0, None)
        fit = fit_profiles_over_time(depth_mm, noisy_profiles, times_sec)
        D_samples.append(fit[0])
        Cs_samples.append(fit[1])
        v_samples.append(fit[2])
        fitted_samples.append(fit[3])
        mean_samples.append(np.nanmean(noisy_profiles, axis=1))

    out["D_std"] = np.nanstd(np.stack(D_samples, axis=0), axis=0, ddof=1) if len(D_samples) > 1 else out["D_std"]
    out["Cs_std"] = np.nanstd(np.stack(Cs_samples, axis=0), axis=0, ddof=1) if len(Cs_samples) > 1 else out["Cs_std"]
    out["v_std"] = np.nanstd(np.stack(v_samples, axis=0), axis=0, ddof=1) if len(v_samples) > 1 else out["v_std"]
    out["mean_conc_std"] = np.nanstd(np.stack(mean_samples, axis=0), axis=0, ddof=1) if len(mean_samples) > 1 else out["mean_conc_std"]
    out["fitted_profiles_std"] = np.nanstd(np.stack(fitted_samples, axis=0), axis=0, ddof=1) if len(fitted_samples) > 1 else out["fitted_profiles_std"]
    return out


def estimate_roi_sensitivity_uncertainty(conc_stack: np.ndarray,
                                         roi: Tuple[int, int, int, int],
                                         times_sec: np.ndarray,
                                         dx_mm: float,
                                         depth_axis: str,
                                         shifts: List[int]) -> Dict[str, np.ndarray]:
    shifts = [int(s) for s in shifts]
    unique_rois = []
    seen = set()
    for shift in shifts:
        shifted = shift_roi_along_depth(roi, shift, conc_stack.shape[1:], depth_axis)
        if shifted not in seen:
            seen.add(shifted)
            unique_rois.append(shifted)

    n_t = conc_stack.shape[0]
    if len(unique_rois) <= 1:
        return {
            "mean_conc_std": np.full(n_t, np.nan, dtype=float),
            "D_std": np.full(n_t, np.nan, dtype=float),
            "Cs_std": np.full(n_t, np.nan, dtype=float),
            "v_std": np.full(n_t, np.nan, dtype=float),
            "fitted_profiles_std": np.full((n_t, len(np.arange(compute_depth_profiles(conc_stack, roi, depth_axis).shape[1]))), np.nan, dtype=float),
            "used_rois": unique_rois,
        }

    mean_conc_list, D_list, Cs_list, v_list, fitted_list = [], [], [], [], []
    depth_len = None
    for roi_variant in unique_rois:
        profiles_variant = compute_depth_profiles(conc_stack, roi_variant, depth_axis)
        if depth_len is None:
            depth_len = profiles_variant.shape[1]
        depth_mm_variant = np.arange(profiles_variant.shape[1]) * dx_mm
        fit = fit_profiles_over_time(depth_mm_variant, profiles_variant, times_sec)
        r0, r1, c0, c1 = roi_variant
        mean_conc_list.append(np.mean(conc_stack[:, r0:r1, c0:c1], axis=(1, 2)))
        D_list.append(fit[0])
        Cs_list.append(fit[1])
        v_list.append(fit[2])
        fitted_list.append(fit[3])

    return {
        "mean_conc_std": np.nanstd(np.stack(mean_conc_list, axis=0), axis=0, ddof=1),
        "D_std": np.nanstd(np.stack(D_list, axis=0), axis=0, ddof=1),
        "Cs_std": np.nanstd(np.stack(Cs_list, axis=0), axis=0, ddof=1),
        "v_std": np.nanstd(np.stack(v_list, axis=0), axis=0, ddof=1),
        "fitted_profiles_std": np.nanstd(np.stack(fitted_list, axis=0), axis=0, ddof=1),
        "used_rois": unique_rois,
    }


def combine_uncertainty_terms(*terms: np.ndarray) -> np.ndarray:
    arrays = [np.asarray(term, dtype=float) for term in terms if term is not None]
    if not arrays:
        return np.array([], dtype=float)
    stacked = np.stack([np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) for arr in arrays], axis=0)
    return np.sqrt(np.sum(stacked ** 2, axis=0))


def approx_ci_from_std(center: np.ndarray, std: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    center = np.asarray(center, dtype=float)
    std = np.asarray(std, dtype=float)
    low = np.full_like(center, np.nan, dtype=float)
    high = np.full_like(center, np.nan, dtype=float)
    mask = np.isfinite(center) & np.isfinite(std)
    low[mask] = center[mask] - z * std[mask]
    high[mask] = center[mask] + z * std[mask]
    return low, high


def smooth_stack(stack: np.ndarray) -> np.ndarray:
    out = stack.copy()

    if APPLY_SPATIAL_SMOOTHING:
        for i in range(out.shape[0]):
            out[i] = gaussian_filter(out[i], sigma=SPATIAL_SMOOTH_SIGMA)

    if APPLY_TEMPORAL_SMOOTHING and out.shape[0] > 2:
        out = gaussian_filter1d(out, sigma=TEMPORAL_SMOOTH_SIGMA, axis=0)

    return out


def smooth_profiles_for_local_map(profiles: np.ndarray) -> np.ndarray:
    """
    Smooth profiles only for the local effective diffusion map workflow.
    This leaves the main per-timepoint/global fits and summary stats unchanged.
    """
    out = np.asarray(profiles, dtype=float).copy()

    if LOCAL_MAP_APPLY_DEPTH_SMOOTHING and out.shape[1] > 2:
        out = gaussian_filter1d(out, sigma=LOCAL_MAP_DEPTH_SMOOTH_SIGMA, axis=1)

    if LOCAL_MAP_APPLY_TEMPORAL_SMOOTHING and out.shape[0] > 2:
        out = gaussian_filter1d(out, sigma=LOCAL_MAP_TEMPORAL_SMOOTH_SIGMA, axis=0)

    return out




def build_banded_effective_diffusion_maps(profiles: np.ndarray,
                                          hu_profiles: np.ndarray,
                                          times_sec: np.ndarray,
                                          dx_mm: float,
                                          sigma_conc: float,
                                          n_bands: int = 4) -> Dict[str, object]:
    """
    Build a non-sliding-window banded effective diffusivity map.
    Each depth band gets one fitted D(t) and Cs(t), which are painted back across
    the full rows belonging to that band.

    Combined uncertainty here includes:
      - model-fit uncertainty
      - HU-noise refit uncertainty
      - calibration uncertainty
    """
    profiles = np.asarray(profiles, dtype=float)
    hu_profiles = np.asarray(hu_profiles, dtype=float)
    times_sec = np.asarray(times_sec, dtype=float)

    n_t, n_x = profiles.shape
    if n_x < 2:
        empty = np.full_like(profiles, np.nan, dtype=float)
        return {
            "D_map": empty.copy(),
            "Cs_map": empty.copy(),
            "D_model_fit_std_map": empty.copy(),
            "D_combined_std_map": empty.copy(),
            "D_model_fit_ci_width_map": empty.copy(),
            "D_combined_ci_width_map": empty.copy(),
            "band_table": pd.DataFrame(),
            "band_edges": np.array([], dtype=int),
        }

    n_bands = int(max(1, min(n_bands, n_x)))
    band_edges = np.linspace(0, n_x, n_bands + 1, dtype=int)

    D_map = np.full((n_t, n_x), np.nan, dtype=float)
    Cs_map = np.full((n_t, n_x), np.nan, dtype=float)
    D_model_fit_std_map = np.full((n_t, n_x), np.nan, dtype=float)
    D_combined_std_map = np.full((n_t, n_x), np.nan, dtype=float)
    D_model_fit_ci_width_map = np.full((n_t, n_x), np.nan, dtype=float)
    D_combined_ci_width_map = np.full((n_t, n_x), np.nan, dtype=float)

    table = pd.DataFrame({"time_seconds": times_sec})
    if TIME_UNIT == "minutes":
        table["time"] = times_sec / 60.0
    else:
        table["time"] = times_sec.copy()

    def _calibration_uncertainty_from_hu_profiles(hu_band_profiles: np.ndarray) -> Dict[str, np.ndarray]:
        out = {
            "D_std": np.full(n_t, np.nan, dtype=float),
            "Cs_std": np.full(n_t, np.nan, dtype=float),
        }
        if HU_PER_CONC_STD <= 0 and HU_OFFSET_STD <= 0:
            out["D_std"][:] = 0.0
            out["Cs_std"][:] = 0.0
            return out

        d_terms = []
        cs_terms = []

        def _fit_variant(hpc: float, hoff: float):
            conc_variant = (hu_band_profiles - float(hoff)) / float(hpc)
            conc_variant = np.asarray(conc_variant, dtype=float)
            conc_variant[conc_variant < 0] = 0
            depth_band_mm = np.arange(conc_variant.shape[1], dtype=float) * float(dx_mm)
            fit_variant = fit_profiles_over_time(depth_band_mm, conc_variant, times_sec)
            return fit_variant[0], fit_variant[1]

        if HU_PER_CONC_STD > 0:
            hpc_plus = max(float(HU_PER_CONC) + float(HU_PER_CONC_STD), 1e-12)
            hpc_minus = max(float(HU_PER_CONC) - float(HU_PER_CONC_STD), 1e-12)
            D_plus, Cs_plus = _fit_variant(hpc_plus, HU_OFFSET)
            D_minus, Cs_minus = _fit_variant(hpc_minus, HU_OFFSET)
            d_terms.append(0.5 * np.abs(D_plus - D_minus))
            cs_terms.append(0.5 * np.abs(Cs_plus - Cs_minus))

        if HU_OFFSET_STD > 0:
            D_plus, Cs_plus = _fit_variant(HU_PER_CONC, float(HU_OFFSET) + float(HU_OFFSET_STD))
            D_minus, Cs_minus = _fit_variant(HU_PER_CONC, float(HU_OFFSET) - float(HU_OFFSET_STD))
            d_terms.append(0.5 * np.abs(D_plus - D_minus))
            cs_terms.append(0.5 * np.abs(Cs_plus - Cs_minus))

        if d_terms:
            out["D_std"] = combine_uncertainty_terms(*d_terms)
        if cs_terms:
            out["Cs_std"] = combine_uncertainty_terms(*cs_terms)
        return out

    for band_idx in range(n_bands):
        x0 = int(band_edges[band_idx])
        x1 = int(band_edges[band_idx + 1])
        if x1 <= x0:
            continue

        band_profiles = profiles[:, x0:x1]
        hu_band_profiles = hu_profiles[:, x0:x1]
        depth_band_mm = np.arange(band_profiles.shape[1], dtype=float) * float(dx_mm)

        fit_res = fit_profiles_over_time(depth_band_mm, band_profiles, times_sec)
        D_vs_time = np.asarray(fit_res[0], dtype=float)
        Cs_vs_time = np.asarray(fit_res[1], dtype=float)
        D_model_std = np.asarray(fit_res[4], dtype=float)

        noise_unc = estimate_noise_uncertainty_via_refits(depth_band_mm, band_profiles, times_sec, sigma_conc)
        cal_unc = _calibration_uncertainty_from_hu_profiles(hu_band_profiles)

        D_combined_std = combine_uncertainty_terms(D_model_std, noise_unc["D_std"], cal_unc["D_std"])

        D_model_ci_width = 2.0 * 1.96 * D_model_std
        D_combined_ci_width = 2.0 * 1.96 * D_combined_std

        D_map[:, x0:x1] = D_vs_time[:, None]
        Cs_map[:, x0:x1] = Cs_vs_time[:, None]
        D_model_fit_std_map[:, x0:x1] = D_model_std[:, None]
        D_combined_std_map[:, x0:x1] = D_combined_std[:, None]
        D_model_fit_ci_width_map[:, x0:x1] = D_model_ci_width[:, None]
        D_combined_ci_width_map[:, x0:x1] = D_combined_ci_width[:, None]

        prefix = f"band_{band_idx + 1}"
        table[f"{prefix}_depth_start_index"] = x0
        table[f"{prefix}_depth_end_index_inclusive"] = x1 - 1
        table[f"{prefix}_D_mm2_s"] = D_vs_time
        table[f"{prefix}_Cs"] = Cs_vs_time
        table[f"{prefix}_D_model_fit_std_mm2_s"] = D_model_std
        table[f"{prefix}_D_combined_std_mm2_s"] = D_combined_std
        table[f"{prefix}_D_model_fit_ci_width_mm2_s"] = D_model_ci_width
        table[f"{prefix}_D_combined_ci_width_mm2_s"] = D_combined_ci_width

    return {
        "D_map": D_map,
        "Cs_map": Cs_map,
        "D_model_fit_std_map": D_model_fit_std_map,
        "D_combined_std_map": D_combined_std_map,
        "D_model_fit_ci_width_map": D_model_fit_ci_width_map,
        "D_combined_ci_width_map": D_combined_ci_width_map,
        "band_table": table,
        "band_edges": band_edges,
    }

def get_depth_spacing_mm(row_spacing_mm: float, col_spacing_mm: float, depth_axis: str) -> float:
    if depth_axis == "rows":
        return row_spacing_mm
    elif depth_axis == "cols":
        return col_spacing_mm
    raise ValueError("DEPTH_AXIS must be 'rows' or 'cols'")


def compute_depth_profiles(conc_stack: np.ndarray,
                           roi: Tuple[int, int, int, int],
                           depth_axis: str) -> np.ndarray:
    r0, r1, c0, c1 = roi
    sub = conc_stack[:, r0:r1, c0:c1]

    if depth_axis == "rows":
        profiles = np.mean(sub, axis=2)
    else:
        profiles = np.mean(sub, axis=1)

    if APPLY_DEPTH_PROFILE_SMOOTHING:
        profiles = gaussian_filter1d(profiles, sigma=DEPTH_PROFILE_SMOOTH_SIGMA, axis=1)

    return profiles


# ============================================================
# TRANSPORT MODELS
# ============================================================

def diffusion_profile_model(x_mm: np.ndarray, D_mm2_s: float, Cs: float, t_s: float) -> np.ndarray:
    t_s = max(float(t_s), MIN_TIME_SECONDS_FOR_FIT)
    D_mm2_s = max(float(D_mm2_s), 1e-12)
    return Cs * erfc(x_mm / (2.0 * np.sqrt(D_mm2_s * t_s)))


def ade_profile_model_fixed_v(x_mm: np.ndarray, D_mm2_s: float, Cs: float, v_mm_s: float, t_s: float) -> np.ndarray:
    """
    Semi-infinite advection-diffusion boundary solution.
    """
    t_s = max(float(t_s), MIN_TIME_SECONDS_FOR_FIT)
    D_mm2_s = max(float(D_mm2_s), 1e-12)

    term1 = erfc((x_mm - v_mm_s * t_s) / (2.0 * np.sqrt(D_mm2_s * t_s)))
    term2 = np.exp((v_mm_s * x_mm) / D_mm2_s) * erfc((x_mm + v_mm_s * t_s) / (2.0 * np.sqrt(D_mm2_s * t_s)))
    return 0.5 * Cs * (term1 + term2)


def ade_profile_model_fit_v(x_mm: np.ndarray, D_mm2_s: float, Cs: float, v_mm_s: float, t_s: float) -> np.ndarray:
    return ade_profile_model_fixed_v(x_mm, D_mm2_s, Cs, v_mm_s, t_s)


def use_ade_model() -> bool:
    """
    Unified logic:
    - pump off  -> pure diffusion model
    - pump on   -> ADE model
    This means pump-on + CONVECTION_METHOD='none' uses ADE with v = 0.
    """
    return PUMP_ON


def compute_velocity_mm_s() -> float:
    """
    Returns the fixed convection velocity [mm/s] used when FIT_VELOCITY = False.
    For pump-on diffusion-only-within-ADE, choose CONVECTION_METHOD='none' and this returns 0.
    """
    if CONVECTION_METHOD == "none":
        return 0.0
    elif CONVECTION_METHOD == "flow":
        flow_mm3_s = FLOW_RATE_ML_MIN * 1000.0 / 60.0
        return flow_mm3_s / EFFECTIVE_AREA_MM2
    elif CONVECTION_METHOD == "darcy":
        return -(PERMEABILITY_MM2 / VISCOSITY_PA_S) * PRESSURE_GRADIENT_PA_MM
    else:
        raise ValueError("CONVECTION_METHOD must be 'none', 'flow', or 'darcy'")


# ============================================================
# FITTING
# ============================================================

def select_profile_points_for_fit(x_mm: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x_mm) & np.isfinite(y)

    if FIT_ONLY_POSITIVE_CONCENTRATION:
        valid &= (y > 0)

    if np.any(valid):
        ymax = np.nanmax(y[valid])
        if np.isfinite(ymax) and ymax > 0:
            valid &= (y >= FIT_PROFILE_THRESHOLD_FRACTION * ymax)

    x_fit = x_mm[valid]
    y_fit = y[valid]

    return x_fit, y_fit


def select_profile_points_for_local_map_fit(x_mm: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x_mm) & np.isfinite(y)

    if FIT_ONLY_POSITIVE_CONCENTRATION:
        valid &= (y > 0)

    if np.any(valid):
        ymax = np.nanmax(y[valid])
        if np.isfinite(ymax) and ymax > 0:
            valid &= (y >= LOCAL_MAP_FIT_PROFILE_THRESHOLD_FRACTION * ymax)

    x_fit = x_mm[valid]
    y_fit = y[valid]

    return x_fit, y_fit


def _empty_fit_result(x_mm: np.ndarray, v_fill: float = np.nan) -> Dict[str, np.ndarray]:
    y_nan = np.full_like(x_mm, np.nan, dtype=float)
    return {
        "D_fit": np.nan,
        "Cs_fit": np.nan,
        "v_fit": v_fill,
        "y_hat": y_nan.copy(),
        "D_std": np.nan,
        "Cs_std": np.nan,
        "v_std": np.nan,
        "D_ci_low": np.nan,
        "D_ci_high": np.nan,
        "Cs_ci_low": np.nan,
        "Cs_ci_high": np.nan,
        "v_ci_low": np.nan,
        "v_ci_high": np.nan,
        "y_std": y_nan.copy(),
        "y_ci_low": y_nan.copy(),
        "y_ci_high": y_nan.copy(),
    }


def _compute_prediction_uncertainty(model_fn, x_pred: np.ndarray, popt: np.ndarray, pcov: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_pred = np.asarray(x_pred, dtype=float)
    popt = np.asarray(popt, dtype=float)
    if pcov is None or np.ndim(pcov) != 2 or pcov.shape[0] != pcov.shape[1] or pcov.shape[0] != popt.size:
        nan_arr = np.full_like(x_pred, np.nan, dtype=float)
        return nan_arr, nan_arr.copy(), nan_arr.copy()

    try:
        y_hat = np.asarray(model_fn(x_pred, *popt), dtype=float)
        if not np.all(np.isfinite(y_hat)):
            nan_arr = np.full_like(x_pred, np.nan, dtype=float)
            return nan_arr, nan_arr.copy(), nan_arr.copy()

        grads = []
        for k, p in enumerate(popt):
            step = max(abs(p) * 1e-6, 1e-8)
            p_hi = popt.copy()
            p_lo = popt.copy()
            p_hi[k] += step
            p_lo[k] -= step
            y_hi = np.asarray(model_fn(x_pred, *p_hi), dtype=float)
            y_lo = np.asarray(model_fn(x_pred, *p_lo), dtype=float)
            grads.append((y_hi - y_lo) / (2.0 * step))
        G = np.column_stack(grads) if grads else np.zeros((x_pred.size, 0), dtype=float)
        y_var = np.einsum("ij,jk,ik->i", G, pcov, G)
        y_var = np.where(np.isfinite(y_var) & (y_var >= 0), y_var, np.nan)
        y_std = np.sqrt(y_var)
        z = norm.ppf(0.975)
        y_ci_low = y_hat - z * y_std
        y_ci_high = y_hat + z * y_std
        return y_std, y_ci_low, y_ci_high
    except Exception:
        nan_arr = np.full_like(x_pred, np.nan, dtype=float)
        return nan_arr, nan_arr.copy(), nan_arr.copy()


def _build_fit_result(x_mm: np.ndarray,
                      model_fn,
                      popt: np.ndarray,
                      pcov: np.ndarray,
                      fit_velocity: bool,
                      fixed_v: Optional[float] = None) -> Dict[str, np.ndarray]:
    result = _empty_fit_result(x_mm, v_fill=(fixed_v if fixed_v is not None else np.nan))
    z = norm.ppf(0.975)

    y_hat = np.asarray(model_fn(x_mm, *popt), dtype=float)
    stds = np.full(len(popt), np.nan, dtype=float)
    if pcov is not None and np.ndim(pcov) == 2 and pcov.shape[0] == pcov.shape[1] == len(popt):
        diag = np.diag(pcov)
        diag = np.where(np.isfinite(diag) & (diag >= 0), diag, np.nan)
        stds = np.sqrt(diag)

    result["D_fit"] = float(popt[0])
    result["Cs_fit"] = float(popt[1])
    result["D_std"] = float(stds[0]) if len(stds) > 0 else np.nan
    result["Cs_std"] = float(stds[1]) if len(stds) > 1 else np.nan
    result["D_ci_low"] = result["D_fit"] - z * result["D_std"] if np.isfinite(result["D_std"]) else np.nan
    result["D_ci_high"] = result["D_fit"] + z * result["D_std"] if np.isfinite(result["D_std"]) else np.nan
    result["Cs_ci_low"] = result["Cs_fit"] - z * result["Cs_std"] if np.isfinite(result["Cs_std"]) else np.nan
    result["Cs_ci_high"] = result["Cs_fit"] + z * result["Cs_std"] if np.isfinite(result["Cs_std"]) else np.nan

    if fit_velocity:
        result["v_fit"] = float(popt[2])
        result["v_std"] = float(stds[2]) if len(stds) > 2 else np.nan
        result["v_ci_low"] = result["v_fit"] - z * result["v_std"] if np.isfinite(result["v_std"]) else np.nan
        result["v_ci_high"] = result["v_fit"] + z * result["v_std"] if np.isfinite(result["v_std"]) else np.nan
    else:
        result["v_fit"] = float(fixed_v) if fixed_v is not None and np.isfinite(fixed_v) else 0.0

    y_std, y_ci_low, y_ci_high = _compute_prediction_uncertainty(model_fn, x_mm, popt, pcov)
    result["y_hat"] = y_hat
    result["y_std"] = y_std
    result["y_ci_low"] = y_ci_low
    result["y_ci_high"] = y_ci_high
    return result


def fit_diffusion_profile(x_mm: np.ndarray, y: np.ndarray, t_s: float) -> Dict[str, np.ndarray]:
    x_fit, y_fit = select_profile_points_for_fit(x_mm, y)

    if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or t_s < MIN_TIME_SECONDS_FOR_FIT:
        return _empty_fit_result(x_mm, v_fill=0.0)

    ymax = max(np.nanmax(y_fit), 1e-6)
    p0 = [1e-3, ymax]
    bounds = ([D_BOUNDS[0], 0.0], [D_BOUNDS[1], MAX_CS_FACTOR * ymax])

    try:
        def model(x, D, Cs):
            return diffusion_profile_model(x, D, Cs, t_s)

        popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
        print(f"t = {t_s:.3f} s")
        print("popt =", popt)
        return _build_fit_result(x_mm, model, np.asarray(popt, dtype=float), pcov, fit_velocity=False, fixed_v=0.0)
    except Exception as e:
        print(f"Fit failed at t={t_s:.3f} s with error: {e}")
        return _empty_fit_result(x_mm, v_fill=0.0)


def fit_ade_profile_fixed_v(x_mm: np.ndarray, y: np.ndarray, t_s: float, v_mm_s: float) -> Dict[str, np.ndarray]:
    x_fit, y_fit = select_profile_points_for_fit(x_mm, y)

    if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or t_s < MIN_TIME_SECONDS_FOR_FIT:
        return _empty_fit_result(x_mm, v_fill=v_mm_s)

    ymax = max(np.nanmax(y_fit), 1e-6)
    p0 = [1e-3, ymax]
    bounds = ([D_BOUNDS[0], 0.0], [D_BOUNDS[1], MAX_CS_FACTOR * ymax])

    try:
        def model(x, D, Cs):
            return ade_profile_model_fixed_v(x, D, Cs, v_mm_s, t_s)

        popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=30000)
        print(f"t = {t_s:.3f} s")
        print("popt =", popt)
        return _build_fit_result(x_mm, model, np.asarray(popt, dtype=float), pcov, fit_velocity=False, fixed_v=v_mm_s)
    except Exception as e:
        print(f"Fit failed at t={t_s:.3f} s with error: {e}")
        return _empty_fit_result(x_mm, v_fill=v_mm_s)


def fit_ade_profile_fit_v(x_mm: np.ndarray, y: np.ndarray, t_s: float) -> Dict[str, np.ndarray]:
    x_fit, y_fit = select_profile_points_for_fit(x_mm, y)

    if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or t_s < MIN_TIME_SECONDS_FOR_FIT:
        return _empty_fit_result(x_mm, v_fill=np.nan)

    ymax = max(np.nanmax(y_fit), 1e-6)
    p0 = [1e-3, ymax, 1e-4]
    bounds = (
        [D_BOUNDS[0], 0.0, V_BOUNDS[0]],
        [D_BOUNDS[1], MAX_CS_FACTOR * ymax, V_BOUNDS[1]]
    )

    try:
        def model(x, D, Cs, v):
            return ade_profile_model_fit_v(x, D, Cs, v, t_s)

        popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=40000)
        print(f"t = {t_s:.3f} s")
        print("popt =", popt)
        return _build_fit_result(x_mm, model, np.asarray(popt, dtype=float), pcov, fit_velocity=True)
    except Exception as e:
        print(f"Fit failed at t={t_s:.3f} s with error: {e}")
        return _empty_fit_result(x_mm, v_fill=np.nan)


def fit_diffusion_profile_local_map(x_mm: np.ndarray, y: np.ndarray, t_s: float) -> Dict[str, np.ndarray]:
    x_fit, y_fit = select_profile_points_for_local_map_fit(x_mm, y)

    if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or t_s < MIN_TIME_SECONDS_FOR_FIT:
        return _empty_fit_result(x_mm, v_fill=0.0)

    ymax = max(np.nanmax(y_fit), 1e-6)
    d_lo, d_hi = LOCAL_MAP_D_BOUNDS
    p0 = [min(1e-3, d_hi), ymax]
    bounds = ([d_lo, 0.0], [d_hi, MAX_CS_FACTOR * ymax])

    try:
        def model(x, D, Cs):
            return diffusion_profile_model(x, D, Cs, t_s)

        popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=20000)
        return _build_fit_result(x_mm, model, np.asarray(popt, dtype=float), pcov, fit_velocity=False, fixed_v=0.0)
    except Exception:
        return _empty_fit_result(x_mm, v_fill=0.0)


def fit_ade_profile_fixed_v_local_map(x_mm: np.ndarray, y: np.ndarray, t_s: float, v_mm_s: float) -> Dict[str, np.ndarray]:
    x_fit, y_fit = select_profile_points_for_local_map_fit(x_mm, y)

    if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or t_s < MIN_TIME_SECONDS_FOR_FIT:
        return _empty_fit_result(x_mm, v_fill=v_mm_s)

    ymax = max(np.nanmax(y_fit), 1e-6)
    d_lo, d_hi = LOCAL_MAP_D_BOUNDS
    p0 = [min(1e-3, d_hi), ymax]
    bounds = ([d_lo, 0.0], [d_hi, MAX_CS_FACTOR * ymax])

    try:
        def model(x, D, Cs):
            return ade_profile_model_fixed_v(x, D, Cs, v_mm_s, t_s)

        popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=30000)
        return _build_fit_result(x_mm, model, np.asarray(popt, dtype=float), pcov, fit_velocity=False, fixed_v=v_mm_s)
    except Exception:
        return _empty_fit_result(x_mm, v_fill=v_mm_s)


def fit_ade_profile_fit_v_local_map(x_mm: np.ndarray, y: np.ndarray, t_s: float) -> Dict[str, np.ndarray]:
    x_fit, y_fit = select_profile_points_for_local_map_fit(x_mm, y)

    if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or t_s < MIN_TIME_SECONDS_FOR_FIT:
        return _empty_fit_result(x_mm, v_fill=np.nan)

    ymax = max(np.nanmax(y_fit), 1e-6)
    d_lo, d_hi = LOCAL_MAP_D_BOUNDS
    p0 = [min(1e-3, d_hi), ymax, 0.1]
    bounds = (
        [d_lo, 0.0, V_BOUNDS[0]],
        [d_hi, MAX_CS_FACTOR * ymax, V_BOUNDS[1]]
    )

    try:
        def model(x, D, Cs, v):
            return ade_profile_model_fit_v(x, D, Cs, v, t_s)

        popt, pcov = curve_fit(model, x_fit, y_fit, p0=p0, bounds=bounds, maxfev=40000)
        return _build_fit_result(x_mm, model, np.asarray(popt, dtype=float), pcov, fit_velocity=True)
    except Exception:
        return _empty_fit_result(x_mm, v_fill=np.nan)


def fit_profiles_over_time(depth_mm: np.ndarray, profiles: np.ndarray, times_sec: np.ndarray):
    """
    Fit one depth profile per timepoint.

    Unified logic:
    - if use_ade_model() is False -> diffusion fit
    - if use_ade_model() is True and FIT_VELOCITY is False -> ADE with fixed v
    - if use_ade_model() is True and FIT_VELOCITY is True  -> ADE fitting D, Cs, v
    """
    n_t = profiles.shape[0]

    D_vs_time = np.full(n_t, np.nan)
    Cs_vs_time = np.full(n_t, np.nan)
    v_vs_time = np.full(n_t, np.nan)
    fitted_profiles = np.full_like(profiles, np.nan, dtype=float)

    D_std_vs_time = np.full(n_t, np.nan)
    Cs_std_vs_time = np.full(n_t, np.nan)
    v_std_vs_time = np.full(n_t, np.nan)

    D_ci_low_vs_time = np.full(n_t, np.nan)
    D_ci_high_vs_time = np.full(n_t, np.nan)
    Cs_ci_low_vs_time = np.full(n_t, np.nan)
    Cs_ci_high_vs_time = np.full(n_t, np.nan)
    v_ci_low_vs_time = np.full(n_t, np.nan)
    v_ci_high_vs_time = np.full(n_t, np.nan)

    fitted_profiles_std = np.full_like(profiles, np.nan, dtype=float)
    fitted_profiles_ci_low = np.full_like(profiles, np.nan, dtype=float)
    fitted_profiles_ci_high = np.full_like(profiles, np.nan, dtype=float)

    ade_mode = use_ade_model()

    if not ade_mode:
        for i in range(n_t):
            fit_res = fit_diffusion_profile(depth_mm, profiles[i], times_sec[i])
            D_vs_time[i] = fit_res["D_fit"]
            Cs_vs_time[i] = fit_res["Cs_fit"]
            v_vs_time[i] = fit_res["v_fit"]
            fitted_profiles[i] = fit_res["y_hat"]
            D_std_vs_time[i] = fit_res["D_std"]
            Cs_std_vs_time[i] = fit_res["Cs_std"]
            v_std_vs_time[i] = fit_res["v_std"]
            D_ci_low_vs_time[i] = fit_res["D_ci_low"]
            D_ci_high_vs_time[i] = fit_res["D_ci_high"]
            Cs_ci_low_vs_time[i] = fit_res["Cs_ci_low"]
            Cs_ci_high_vs_time[i] = fit_res["Cs_ci_high"]
            v_ci_low_vs_time[i] = fit_res["v_ci_low"]
            v_ci_high_vs_time[i] = fit_res["v_ci_high"]
            fitted_profiles_std[i] = fit_res["y_std"]
            fitted_profiles_ci_low[i] = fit_res["y_ci_low"]
            fitted_profiles_ci_high[i] = fit_res["y_ci_high"]
    else:
        if FIT_VELOCITY:
            for i in range(n_t):
                fit_res = fit_ade_profile_fit_v(depth_mm, profiles[i], times_sec[i])
                D_vs_time[i] = fit_res["D_fit"]
                Cs_vs_time[i] = fit_res["Cs_fit"]
                v_vs_time[i] = fit_res["v_fit"]
                fitted_profiles[i] = fit_res["y_hat"]
                D_std_vs_time[i] = fit_res["D_std"]
                Cs_std_vs_time[i] = fit_res["Cs_std"]
                v_std_vs_time[i] = fit_res["v_std"]
                D_ci_low_vs_time[i] = fit_res["D_ci_low"]
                D_ci_high_vs_time[i] = fit_res["D_ci_high"]
                Cs_ci_low_vs_time[i] = fit_res["Cs_ci_low"]
                Cs_ci_high_vs_time[i] = fit_res["Cs_ci_high"]
                v_ci_low_vs_time[i] = fit_res["v_ci_low"]
                v_ci_high_vs_time[i] = fit_res["v_ci_high"]
                fitted_profiles_std[i] = fit_res["y_std"]
                fitted_profiles_ci_low[i] = fit_res["y_ci_low"]
                fitted_profiles_ci_high[i] = fit_res["y_ci_high"]
        else:
            fixed_v = compute_velocity_mm_s()
            v_vs_time[:] = fixed_v
            for i in range(n_t):
                fit_res = fit_ade_profile_fixed_v(depth_mm, profiles[i], times_sec[i], fixed_v)
                D_vs_time[i] = fit_res["D_fit"]
                Cs_vs_time[i] = fit_res["Cs_fit"]
                v_vs_time[i] = fit_res["v_fit"]
                fitted_profiles[i] = fit_res["y_hat"]
                D_std_vs_time[i] = fit_res["D_std"]
                Cs_std_vs_time[i] = fit_res["Cs_std"]
                v_std_vs_time[i] = fit_res["v_std"]
                D_ci_low_vs_time[i] = fit_res["D_ci_low"]
                D_ci_high_vs_time[i] = fit_res["D_ci_high"]
                Cs_ci_low_vs_time[i] = fit_res["Cs_ci_low"]
                Cs_ci_high_vs_time[i] = fit_res["Cs_ci_high"]
                v_ci_low_vs_time[i] = fit_res["v_ci_low"]
                v_ci_high_vs_time[i] = fit_res["v_ci_high"]
                fitted_profiles_std[i] = fit_res["y_std"]
                fitted_profiles_ci_low[i] = fit_res["y_ci_low"]
                fitted_profiles_ci_high[i] = fit_res["y_ci_high"]

    return (
        D_vs_time, Cs_vs_time, v_vs_time, fitted_profiles,
        D_std_vs_time, Cs_std_vs_time, v_std_vs_time,
        D_ci_low_vs_time, D_ci_high_vs_time,
        Cs_ci_low_vs_time, Cs_ci_high_vs_time,
        v_ci_low_vs_time, v_ci_high_vs_time,
        fitted_profiles_std, fitted_profiles_ci_low, fitted_profiles_ci_high
    )


def _fill_nan_with_interpolation(values: np.ndarray, fallback: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float).copy()
    idx = np.arange(arr.size)
    valid = np.isfinite(arr)
    if np.sum(valid) == 0:
        arr[:] = fallback
        return arr
    if np.sum(valid) == 1:
        arr[:] = arr[valid][0]
        return arr
    arr[~valid] = np.interp(idx[~valid], idx[valid], arr[valid])
    return arr


def fit_profiles_over_time_temporally_regularized(depth_mm: np.ndarray, profiles: np.ndarray, times_sec: np.ndarray):
    """
    Hybrid fit: each timepoint still has its own D(t), Cs(t), but all timepoints
    are solved together with temporal smoothness penalties.

    This is intended as a middle ground between:
      - fully independent per-timepoint fitting
      - one-D global fitting
    """
    ade_mode = use_ade_model()

    # For now, keep velocity fixed unless user is already using fixed-v logic.
    if ade_mode and FIT_VELOCITY:
        print("Temporally regularized fit currently supports fixed-v or diffusion-only runs. Falling back to per-timepoint fit.")
        fit_init = fit_profiles_over_time(depth_mm, profiles, times_sec)
        D_init, Cs_init, v_init, fitted_init = fit_init[:4]
        return D_init, Cs_init, v_init, fitted_init, D_init.copy(), Cs_init.copy()

    n_t = profiles.shape[0]
    fit_init = fit_profiles_over_time(depth_mm, profiles, times_sec)
    D_init, Cs_init, v_init, fitted_init = fit_init[:4]
    valid_time_idx = []
    x_fit_list = []
    y_fit_list = []
    fit_time_list = []
    ymax_global = np.nanmax(profiles) if np.any(np.isfinite(profiles)) else 1.0
    if not np.isfinite(ymax_global) or ymax_global <= 0:
        ymax_global = 1.0

    for i in range(n_t):
        x_fit, y_fit = select_profile_points_for_fit(depth_mm, profiles[i])
        if len(x_fit) < MIN_VALID_POINTS_FOR_FIT or times_sec[i] < MIN_TIME_SECONDS_FOR_FIT:
            continue
        valid_time_idx.append(i)
        x_fit_list.append(x_fit)
        y_fit_list.append(y_fit)
        fit_time_list.append(times_sec[i])

    if len(valid_time_idx) < 2:
        return D_init, Cs_init, v_init, fitted_init, D_init.copy(), Cs_init.copy()

    valid_time_idx = np.asarray(valid_time_idx, dtype=int)
    fit_time_list = np.asarray(fit_time_list, dtype=float)

    finite_D = D_init[np.isfinite(D_init)]
    D_fallback = float(np.nanmedian(finite_D)) if finite_D.size else 1e-3
    D_fallback = float(np.clip(D_fallback, D_BOUNDS[0] * 10, D_BOUNDS[1] / 10))

    finite_Cs = Cs_init[np.isfinite(Cs_init)]
    Cs_fallback = float(np.nanmedian(finite_Cs)) if finite_Cs.size else ymax_global
    Cs_fallback = float(np.clip(Cs_fallback, 0.0, MAX_CS_FACTOR * ymax_global))

    D_valid_init = _fill_nan_with_interpolation(D_init[valid_time_idx], D_fallback)
    Cs_valid_init = _fill_nan_with_interpolation(Cs_init[valid_time_idx], Cs_fallback)
    logD_init = np.log10(np.clip(D_valid_init, D_BOUNDS[0], D_BOUNDS[1]))

    p0 = np.concatenate([logD_init, Cs_valid_init])
    lb = np.concatenate([
        np.full(len(valid_time_idx), np.log10(D_BOUNDS[0])),
        np.zeros(len(valid_time_idx))
    ])
    ub = np.concatenate([
        np.full(len(valid_time_idx), np.log10(D_BOUNDS[1])),
        np.full(len(valid_time_idx), MAX_CS_FACTOR * ymax_global)
    ])

    cs_scale = max(float(np.nanmax(Cs_valid_init)), 1.0)
    fixed_v = compute_velocity_mm_s() if ade_mode else 0.0

    def residual_func(params):
        n = len(valid_time_idx)
        logD_t = params[:n]
        Cs_t = params[n:]
        D_t = 10 ** logD_t

        data_res = []
        for x_fit, y_fit, t_s, D_i, Cs_i in zip(x_fit_list, y_fit_list, fit_time_list, D_t, Cs_t):
            if ade_mode:
                y_hat = ade_profile_model_fixed_v(x_fit, D_i, Cs_i, fixed_v, t_s)
            else:
                y_hat = diffusion_profile_model(x_fit, D_i, Cs_i, t_s)
            data_res.append(y_hat - y_fit)
        data_res = np.concatenate(data_res) if data_res else np.array([], dtype=float)

        reg_parts = [data_res]
        if len(logD_t) >= 3 and REGULARIZED_FIT_LAMBDA_D > 0:
            reg_parts.append(np.sqrt(REGULARIZED_FIT_LAMBDA_D) * (logD_t[:-2] - 2 * logD_t[1:-1] + logD_t[2:]))
        if len(Cs_t) >= 3 and REGULARIZED_FIT_LAMBDA_CS > 0:
            reg_parts.append(np.sqrt(REGULARIZED_FIT_LAMBDA_CS) * ((Cs_t[:-2] - 2 * Cs_t[1:-1] + Cs_t[2:]) / cs_scale))
        if len(Cs_t) >= 2 and REGULARIZED_FIT_LAMBDA_CS_MONOTONIC > 0:
            neg_steps = np.minimum(np.diff(Cs_t), 0.0) / cs_scale
            reg_parts.append(np.sqrt(REGULARIZED_FIT_LAMBDA_CS_MONOTONIC) * neg_steps)
        return np.concatenate(reg_parts)

    result = least_squares(
        residual_func,
        x0=np.asarray(p0, dtype=float),
        bounds=(np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)),
        max_nfev=REGULARIZED_FIT_MAX_NFEV
    )

    params = result.x
    n = len(valid_time_idx)
    logD_opt = params[:n]
    Cs_opt = params[n:]
    D_opt = 10 ** logD_opt

    D_vs_time = np.full(n_t, np.nan, dtype=float)
    Cs_vs_time = np.full(n_t, np.nan, dtype=float)
    v_vs_time = np.full(n_t, fixed_v if ade_mode else 0.0, dtype=float)
    fitted_profiles = np.full_like(profiles, np.nan, dtype=float)

    D_vs_time[valid_time_idx] = D_opt
    Cs_vs_time[valid_time_idx] = Cs_opt

    D_plot = D_vs_time.copy()
    Cs_plot = Cs_vs_time.copy()
    if np.sum(np.isfinite(D_plot)) >= 2:
        D_plot = _fill_nan_with_interpolation(D_plot, D_fallback)
    if np.sum(np.isfinite(Cs_plot)) >= 2:
        Cs_plot = _fill_nan_with_interpolation(Cs_plot, Cs_fallback)

    for idx, D_i, Cs_i in zip(valid_time_idx, D_opt, Cs_opt):
        t_s = times_sec[idx]
        if ade_mode:
            fitted_profiles[idx] = ade_profile_model_fixed_v(depth_mm, D_i, Cs_i, fixed_v, t_s)
        else:
            fitted_profiles[idx] = diffusion_profile_model(depth_mm, D_i, Cs_i, t_s)

    return D_vs_time, Cs_vs_time, v_vs_time, fitted_profiles, D_plot, Cs_plot



def build_smooth_cs_curve(times_sec: np.ndarray, control_times_sec: np.ndarray, cs_control_values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(control_times_sec) & np.isfinite(cs_control_values)
    if np.sum(valid) < 2:
        return np.full_like(times_sec, np.nan, dtype=float)
    t_ctrl = np.asarray(control_times_sec[valid], dtype=float)
    cs_ctrl = np.asarray(cs_control_values[valid], dtype=float)
    order = np.argsort(t_ctrl)
    t_ctrl = t_ctrl[order]
    cs_ctrl = cs_ctrl[order]
    if len(np.unique(t_ctrl)) < 2:
        return np.full_like(times_sec, cs_ctrl[0], dtype=float)
    try:
        interp = PchipInterpolator(t_ctrl, cs_ctrl, extrapolate=True)
        return interp(times_sec)
    except Exception:
        return np.interp(times_sec, t_ctrl, cs_ctrl)


def _predict_profiles_from_global_params(depth_mm: np.ndarray, times_sec: np.ndarray, D_global: float, cs_vs_time: np.ndarray, v_global: float = 0.0) -> np.ndarray:
    pred = np.full((len(times_sec), len(depth_mm)), np.nan, dtype=float)
    ade_mode = use_ade_model()

    for i, t_s in enumerate(times_sec):
        if not np.isfinite(t_s) or t_s < MIN_TIME_SECONDS_FOR_FIT or not np.isfinite(cs_vs_time[i]):
            continue
        if ade_mode:
            pred[i] = ade_profile_model_fixed_v(depth_mm, D_global, cs_vs_time[i], v_global, t_s)
        else:
            pred[i] = diffusion_profile_model(depth_mm, D_global, cs_vs_time[i], t_s)
    return pred


def fit_global_spatiotemporal_profiles(depth_mm: np.ndarray, profiles: np.ndarray, times_sec: np.ndarray):
    """
    Global fit across all valid times with:
      - one ROI-level D
      - smooth Cs(t) represented by a small set of control points
      - optional one ROI-level v for advection cases
    """
    n_t = len(times_sec)
    valid_time_idx = np.where(times_sec >= MIN_TIME_SECONDS_FOR_FIT)[0]
    if len(valid_time_idx) < 2:
        nan_prof = np.full_like(profiles, np.nan, dtype=float)
        return np.nan, np.full(n_t, np.nan), np.nan, nan_prof, np.full(0, np.nan), np.full(0, np.nan)

    control_count = int(np.clip(GLOBAL_CS_NUM_CONTROL_POINTS, 2, len(valid_time_idx)))
    control_pick_idx = np.unique(np.linspace(0, len(valid_time_idx) - 1, control_count).astype(int))
    control_time_indices = valid_time_idx[control_pick_idx]
    control_times_sec = times_sec[control_time_indices]

    profile_max = np.nanmax(profiles[valid_time_idx]) if np.any(np.isfinite(profiles[valid_time_idx])) else 1.0
    if not np.isfinite(profile_max) or profile_max <= 0:
        profile_max = 1.0

    cs_init = []
    for idx in control_time_indices:
        prof = profiles[idx]
        prof_finite = prof[np.isfinite(prof)]
        cs_init.append(float(np.nanmax(prof_finite)) if prof_finite.size else profile_max)
    cs_init = np.asarray(cs_init, dtype=float)
    cs_init[~np.isfinite(cs_init)] = profile_max
    cs_init = np.clip(cs_init, 0.0, MAX_CS_FACTOR * profile_max)

    # start from stable late-time median D when possible
    late_D_seed = 1e-3
    try:
        prof_fit = fit_profiles_over_time(depth_mm, profiles, times_sec)
        prof_D = prof_fit[0]
        finite_D = prof_D[np.isfinite(prof_D)]
        if finite_D.size:
            late_D_seed = float(np.nanmedian(finite_D[-min(10, finite_D.size):]))
    except Exception:
        pass
    late_D_seed = float(np.clip(late_D_seed, D_BOUNDS[0] * 10, D_BOUNDS[1] / 10))

    ade_mode = use_ade_model()
    fit_v = ade_mode and GLOBAL_FIT_INCLUDE_VELOCITY
    fixed_v = compute_velocity_mm_s() if (ade_mode and not fit_v) else 0.0

    p0 = [np.log10(late_D_seed)] + cs_init.tolist()
    lb = [np.log10(D_BOUNDS[0])] + [0.0] * len(cs_init)
    ub = [np.log10(D_BOUNDS[1])] + [MAX_CS_FACTOR * profile_max] * len(cs_init)
    if fit_v:
        p0.append(0.0)
        lb.append(V_BOUNDS[0])
        ub.append(V_BOUNDS[1])

    valid_masks = []
    x_fit_list = []
    y_fit_list = []
    fit_time_list = []
    for i in valid_time_idx:
        x_fit, y_fit = select_profile_points_for_fit(depth_mm, profiles[i])
        if len(x_fit) < MIN_VALID_POINTS_FOR_FIT:
            continue
        x_fit_list.append(x_fit)
        y_fit_list.append(y_fit)
        fit_time_list.append(times_sec[i])
        valid_masks.append(i)

    if len(x_fit_list) < 2:
        nan_prof = np.full_like(profiles, np.nan, dtype=float)
        return np.nan, np.full(n_t, np.nan), np.nan, nan_prof, control_times_sec, cs_init

    def residual_func(params):
        logD = params[0]
        Dg = 10 ** logD
        cs_ctrl = np.asarray(params[1:1 + len(control_times_sec)], dtype=float)
        vg = params[-1] if fit_v else fixed_v
        cs_fit_times = build_smooth_cs_curve(np.asarray(fit_time_list, dtype=float), control_times_sec, cs_ctrl)
        res_parts = []
        for x_fit, y_fit, t_s, Cs_t in zip(x_fit_list, y_fit_list, fit_time_list, cs_fit_times):
            if ade_mode:
                y_hat = ade_profile_model_fixed_v(x_fit, Dg, Cs_t, vg, t_s)
            else:
                y_hat = diffusion_profile_model(x_fit, Dg, Cs_t, t_s)
            res_parts.append(y_hat - y_fit)
        return np.concatenate(res_parts) if res_parts else np.array([1e6])

    result = least_squares(residual_func, x0=np.asarray(p0, dtype=float), bounds=(np.asarray(lb), np.asarray(ub)), max_nfev=GLOBAL_FIT_MAX_NFEV)
    params = result.x
    D_global = 10 ** params[0]
    cs_ctrl_opt = np.asarray(params[1:1 + len(control_times_sec)], dtype=float)
    v_global = params[-1] if fit_v else fixed_v
    cs_vs_time = build_smooth_cs_curve(times_sec, control_times_sec, cs_ctrl_opt)
    fitted_profiles = _predict_profiles_from_global_params(depth_mm, times_sec, D_global, cs_vs_time, v_global)

    return D_global, cs_vs_time, v_global, fitted_profiles, control_times_sec, cs_ctrl_opt


def plot_rmse_comparison(times_plot, rmse_per_time, rmse_global, out_path, roi_name, rmse_regularized=None):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_plot, rmse_per_time, marker='o', linewidth=2, label='Per-time fit RMSE')
    ax.plot(times_plot, rmse_global, marker='s', linewidth=2, label='Global fit RMSE')
    if rmse_regularized is not None:
        ax.plot(times_plot, rmse_regularized, marker='^', linewidth=2, label='Temporally regularized fit RMSE')
    ax.set_xlabel(f'Time ({TIME_UNIT})')
    ax.set_ylabel(f'Profile fit RMSE ({CONCENTRATION_UNITS})')
    ax.set_title(f'{roi_display_name}: RMSE Comparison')
    ax.grid(True)
    ax.legend()
    save_figure(fig, out_path)


# ============================================================
# DERIVATIVE-BASED SECONDARY MAPS
# ============================================================

def compute_dcdt(C: np.ndarray, times_sec: np.ndarray) -> np.ndarray:
    return np.gradient(C, times_sec, axis=0)


def compute_spatial_derivatives(C: np.ndarray, dx_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    dCdx = np.gradient(C, dx_mm, axis=1)
    d2Cdx2 = np.gradient(dCdx, dx_mm, axis=1)
    return dCdx, d2Cdx2


def compute_transport_terms_from_fitted_params(
    profiles: np.ndarray,
    times_sec: np.ndarray,
    dx_mm: float,
    D_vs_time: np.ndarray,
    v_vs_time: np.ndarray
):
    dCdx, d2Cdx2 = compute_spatial_derivatives(profiles, dx_mm)

    diffusion_term = np.full_like(profiles, np.nan, dtype=float)
    convection_term = np.full_like(profiles, np.nan, dtype=float)
    total_term = np.full_like(profiles, np.nan, dtype=float)

    for i in range(profiles.shape[0]):
        D_i = D_vs_time[i]
        v_i = v_vs_time[i] if np.isfinite(v_vs_time[i]) else 0.0

        if np.isfinite(D_i):
            diffusion_term[i] = D_i * d2Cdx2[i]
        if np.isfinite(v_i):
            convection_term[i] = -v_i * dCdx[i]

        if np.isfinite(D_i):
            total_term[i] = diffusion_term[i] + convection_term[i]

    diffusion_curve = np.nanmean(np.abs(diffusion_term), axis=1)
    convection_curve = np.nanmean(np.abs(convection_term), axis=1)
    total_curve = np.nanmean(np.abs(total_term), axis=1)

    return diffusion_term, convection_term, total_term, diffusion_curve, convection_curve, total_curve


def compute_diffusive_flux_map_from_fitted_params(
    profiles: np.ndarray,
    dx_mm: float,
    D_vs_time: np.ndarray
):
    """
    Fick's First Law:
        J_diff = -D * dC/dx

    Returns:
        dCdx               : concentration gradient map, shape [time, depth]
        diffusive_flux     : signed diffusive flux map, shape [time, depth]
        diffusive_flux_mag : absolute diffusive flux magnitude map, shape [time, depth]
        mean_flux_mag      : mean |J_diff| over depth, shape [time]
    """
    dCdx, _ = compute_spatial_derivatives(profiles, dx_mm)

    diffusive_flux = np.full_like(profiles, np.nan, dtype=float)

    for i in range(profiles.shape[0]):
        D_i = D_vs_time[i]
        if np.isfinite(D_i):
            diffusive_flux[i] = -D_i * dCdx[i]

    diffusive_flux_mag = np.abs(diffusive_flux)
    mean_flux_mag = np.nanmean(diffusive_flux_mag, axis=1)

    return dCdx, diffusive_flux, diffusive_flux_mag, mean_flux_mag


def _masked_envelope(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    masked = np.ma.masked_invalid(np.asarray(stack, dtype=float))
    low = np.ma.min(masked, axis=0).filled(np.nan)
    high = np.ma.max(masked, axis=0).filled(np.nan)
    return low, high


def compute_diffusive_flux_uncertainty_component_from_std(
    profiles: np.ndarray,
    dx_mm: float,
    D_vs_time: np.ndarray,
    D_std_vs_time: Optional[np.ndarray] = None,
    profiles_std: Optional[np.ndarray] = None,
    z: float = 1.96
) -> Dict[str, np.ndarray]:
    """
    Approximate per-component uncertainty propagation for diffusive flux maps using
    the fitted concentration field and fitted D(t).

    Flux definition:
        J_diff = -D * dC/dx

    The component uncertainty is propagated by constructing low/high envelopes for
    both D(t) and C(x,t), evaluating the four signed-flux combinations, and then
    converting the resulting CI width to an SD map.

    Returns signed and magnitude uncertainty maps.
    """
    profiles = np.asarray(profiles, dtype=float)
    D_vs_time = np.asarray(D_vs_time, dtype=float)

    n_t, n_x = profiles.shape
    nan_map = np.full((n_t, n_x), np.nan, dtype=float)

    if D_std_vs_time is None and profiles_std is None:
        return {
            "signed_ci_low_map": nan_map.copy(),
            "signed_ci_high_map": nan_map.copy(),
            "signed_ci_width_map": nan_map.copy(),
            "signed_std_map": nan_map.copy(),
            "magnitude_ci_low_map": nan_map.copy(),
            "magnitude_ci_high_map": nan_map.copy(),
            "magnitude_ci_width_map": nan_map.copy(),
            "magnitude_std_map": nan_map.copy(),
        }

    if D_std_vs_time is not None:
        D_ci_low, D_ci_high = approx_ci_from_std(D_vs_time, np.asarray(D_std_vs_time, dtype=float), z=z)
    else:
        D_ci_low = D_vs_time.copy()
        D_ci_high = D_vs_time.copy()

    if profiles_std is not None:
        C_ci_low, C_ci_high = approx_ci_from_std(profiles, np.asarray(profiles_std, dtype=float), z=z)
    else:
        C_ci_low = profiles.copy()
        C_ci_high = profiles.copy()

    grad_low, _ = compute_spatial_derivatives(C_ci_low, dx_mm)
    grad_high, _ = compute_spatial_derivatives(C_ci_high, dx_mm)

    signed_candidates = np.stack([
        -D_ci_low[:, None] * grad_low,
        -D_ci_low[:, None] * grad_high,
        -D_ci_high[:, None] * grad_low,
        -D_ci_high[:, None] * grad_high,
    ], axis=0)

    signed_ci_low_map, signed_ci_high_map = _masked_envelope(signed_candidates)
    signed_ci_width_map = signed_ci_high_map - signed_ci_low_map
    signed_std_map = signed_ci_width_map / (2.0 * z)

    magnitude_candidates = np.abs(signed_candidates)
    magnitude_ci_low_map, magnitude_ci_high_map = _masked_envelope(magnitude_candidates)
    magnitude_ci_width_map = magnitude_ci_high_map - magnitude_ci_low_map
    magnitude_std_map = magnitude_ci_width_map / (2.0 * z)

    return {
        "signed_ci_low_map": signed_ci_low_map,
        "signed_ci_high_map": signed_ci_high_map,
        "signed_ci_width_map": signed_ci_width_map,
        "signed_std_map": signed_std_map,
        "magnitude_ci_low_map": magnitude_ci_low_map,
        "magnitude_ci_high_map": magnitude_ci_high_map,
        "magnitude_ci_width_map": magnitude_ci_width_map,
        "magnitude_std_map": magnitude_std_map,
    }


def compute_diffusive_flux_uncertainty_summary_from_maps(
    flux_map: np.ndarray,
    flux_mag_map: np.ndarray,
    fixed_roi_std_map: np.ndarray,
    combined_std_map: np.ndarray,
    z: float = 1.96
) -> Dict[str, np.ndarray]:
    """
    Build depth-averaged uncertainty summaries for the diffusive flux magnitude map.
    These are intended for time-course reporting, while the full maps are saved separately.
    """
    mean_flux_mag = np.nanmean(flux_mag_map, axis=1)
    mean_fixed_roi_std = np.nanmean(fixed_roi_std_map, axis=1)
    mean_combined_std = np.nanmean(combined_std_map, axis=1)

    mean_fixed_roi_ci_low, mean_fixed_roi_ci_high = approx_ci_from_std(mean_flux_mag, mean_fixed_roi_std, z=z)
    mean_combined_ci_low, mean_combined_ci_high = approx_ci_from_std(mean_flux_mag, mean_combined_std, z=z)

    return {
        "mean_flux_mag": mean_flux_mag,
        "mean_fixed_roi_std": mean_fixed_roi_std,
        "mean_fixed_roi_ci_low": mean_fixed_roi_ci_low,
        "mean_fixed_roi_ci_high": mean_fixed_roi_ci_high,
        "mean_combined_std": mean_combined_std,
        "mean_combined_ci_low": mean_combined_ci_low,
        "mean_combined_ci_high": mean_combined_ci_high,
    }




def compute_peclet_vs_time(
    depth_mm: np.ndarray,
    profiles: np.ndarray,
    D_vs_time: np.ndarray,
    v_vs_time: np.ndarray,
    mode: str = PECLET_LENGTH_MODE,
    threshold_fraction: float = PECLET_THRESHOLD_FRACTION,
    min_length_mm: float = PECLET_MIN_LENGTH_MM,
    eps: float = TRANSPORT_EPS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a time-varying Péclet number:
        Pe = |v| L / D

    Returns:
        peclet_vs_time : shape [time]
        length_mm      : characteristic length used at each timepoint [time]
    """
    n_time = profiles.shape[0]
    peclet_vs_time = np.full(n_time, np.nan, dtype=float)
    length_mm = np.full(n_time, np.nan, dtype=float)

    finite_depth = np.asarray(depth_mm, dtype=float)
    finite_depth = finite_depth[np.isfinite(finite_depth)]
    roi_depth_mm = float(np.max(finite_depth) - np.min(finite_depth)) if finite_depth.size else np.nan

    for i in range(n_time):
        D_i = D_vs_time[i] if i < len(D_vs_time) else np.nan
        v_i = v_vs_time[i] if i < len(v_vs_time) else np.nan
        profile_i = profiles[i]

        if not (np.isfinite(D_i) and D_i > 0 and np.isfinite(v_i)):
            continue

        if mode == "penetration_depth":
            valid = np.isfinite(profile_i) & np.isfinite(depth_mm)
            if np.any(valid):
                prof = profile_i[valid]
                dep = depth_mm[valid]
                prof_max = np.nanmax(prof)
                if np.isfinite(prof_max) and prof_max > 0:
                    thresh = threshold_fraction * prof_max
                    above = dep[prof >= thresh]
                    if above.size:
                        length_i = float(np.max(above) - np.min(dep))
                    else:
                        length_i = roi_depth_mm
                else:
                    length_i = roi_depth_mm
            else:
                length_i = roi_depth_mm
        else:
            length_i = roi_depth_mm

        if not np.isfinite(length_i):
            continue

        length_i = max(length_i, min_length_mm)
        length_mm[i] = length_i
        peclet_vs_time[i] = (abs(v_i) * length_i) / max(D_i, eps)

    return peclet_vs_time, length_mm


def compute_convection_fraction_map(
    convection_term: np.ndarray,
    diffusion_term: np.ndarray,
    eps: float = TRANSPORT_EPS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convection fraction = |conv| / (|conv| + |diff|)

    Returns:
        convection_fraction_map      : shape [time, depth]
        convection_fraction_vs_time  : mean convection fraction over depth, shape [time]
    """
    numerator = np.abs(convection_term)
    denominator = numerator + np.abs(diffusion_term) + eps
    convection_fraction_map = numerator / denominator
    convection_fraction_map[~np.isfinite(convection_fraction_map)] = np.nan
    convection_fraction_vs_time = np.nanmean(convection_fraction_map, axis=1)
    return convection_fraction_map, convection_fraction_vs_time


def compute_relative_uncertainty_percent_map(reference_map: np.ndarray, uncertainty_map: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Convert an absolute uncertainty map into a relative uncertainty (%) map."""
    ref = np.asarray(reference_map, dtype=float)
    unc = np.asarray(uncertainty_map, dtype=float)
    rel = 100.0 * unc / np.maximum(np.abs(ref), eps)
    rel[~np.isfinite(rel)] = np.nan
    return rel


def compute_convection_ci_maps_from_profile_ci(
    fitted_profiles_ci_low: np.ndarray,
    fitted_profiles_ci_high: np.ndarray,
    dx_mm: float,
    v_vs_time: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagates fitted-profile confidence bounds through the convection term:
        convection = -v * dC/dx
    """
    dCdx_low, _ = compute_spatial_derivatives(fitted_profiles_ci_low, dx_mm)
    dCdx_high, _ = compute_spatial_derivatives(fitted_profiles_ci_high, dx_mm)

    convection_low_candidate = np.full_like(dCdx_low, np.nan, dtype=float)
    convection_high_candidate = np.full_like(dCdx_high, np.nan, dtype=float)

    for i in range(dCdx_low.shape[0]):
        v_i = v_vs_time[i] if np.isfinite(v_vs_time[i]) else 0.0
        convection_low_candidate[i] = -v_i * dCdx_low[i]
        convection_high_candidate[i] = -v_i * dCdx_high[i]

    convection_ci_low = np.minimum(convection_low_candidate, convection_high_candidate)
    convection_ci_high = np.maximum(convection_low_candidate, convection_high_candidate)
    convection_ci_width = convection_ci_high - convection_ci_low

    convection_ci_low[~np.isfinite(convection_ci_low)] = np.nan
    convection_ci_high[~np.isfinite(convection_ci_high)] = np.nan
    convection_ci_width[~np.isfinite(convection_ci_width)] = np.nan

    return convection_ci_low, convection_ci_high, convection_ci_width


def fit_local_effective_transport_maps(
    depth_mm: np.ndarray,
    profiles: np.ndarray,
    times_sec: np.ndarray,
    window_size: int = LOCAL_EFFECTIVE_MAP_WINDOW_SIZE
):
    """
    Sliding-window analytical fits to estimate local effective transport parameters.

    Returns:
        D_local_map         : local effective diffusion map [time, depth]
        Cs_local_map        : local effective boundary/source concentration map [time, depth]
        v_local_map         : local effective velocity map [time, depth]
        pred_local_map      : center-value fitted concentration map [time, depth]
        rmse_local_map      : local fit RMSE map [time, depth]
    """
    if window_size < 3:
        raise ValueError("LOCAL_EFFECTIVE_MAP_WINDOW_SIZE must be at least 3.")
    if window_size % 2 == 0:
        raise ValueError("LOCAL_EFFECTIVE_MAP_WINDOW_SIZE must be odd.")

    n_t, n_x = profiles.shape
    half_w = window_size // 2

    D_local_map = np.full((n_t, n_x), np.nan, dtype=float)
    Cs_local_map = np.full((n_t, n_x), np.nan, dtype=float)
    v_local_map = np.full((n_t, n_x), np.nan, dtype=float)
    pred_local_map = np.full((n_t, n_x), np.nan, dtype=float)
    rmse_local_map = np.full((n_t, n_x), np.nan, dtype=float)

    ade_mode = use_ade_model()
    fixed_v = compute_velocity_mm_s() if (ade_mode and not FIT_VELOCITY) else np.nan

    for i in range(n_t):
        t_i = times_sec[i]
        if t_i < MIN_TIME_SECONDS_FOR_FIT:
            continue

        profile_i = profiles[i]

        for j in range(n_x):
            j0 = max(0, j - half_w)
            j1 = min(n_x, j + half_w + 1)

            x_win = depth_mm[j0:j1]
            y_win = profile_i[j0:j1]

            if ade_mode:
                if FIT_VELOCITY:
                    fit_res = fit_ade_profile_fit_v_local_map(x_win, y_win, t_i)
                else:
                    fit_res = fit_ade_profile_fixed_v_local_map(x_win, y_win, t_i, fixed_v)
            else:
                fit_res = fit_diffusion_profile_local_map(x_win, y_win, t_i)

            D_local_map[i, j] = fit_res["D_fit"]
            Cs_local_map[i, j] = fit_res["Cs_fit"]
            v_local_map[i, j] = fit_res["v_fit"]
            y_hat = fit_res["y_hat"]

            finite_valid = np.isfinite(y_win) & np.isfinite(y_hat)
            if np.any(finite_valid):
                rmse_local_map[i, j] = np.sqrt(np.mean((y_win[finite_valid] - y_hat[finite_valid]) ** 2))

            center_idx = j - j0
            if np.all(np.isfinite(y_hat)) and 0 <= center_idx < len(y_hat):
                pred_local_map[i, j] = y_hat[center_idx]

    return D_local_map, Cs_local_map, v_local_map, pred_local_map, rmse_local_map


def compute_derived_effective_diffusion_map_from_field(profiles: np.ndarray,
                                                  times_sec: np.ndarray,
                                                  dx_mm: float,
                                                  v_vs_time: Optional[np.ndarray] = None,
                                                  d_bounds: Tuple[float, float] = DERIVED_EFFECTIVE_D_BOUNDS,
                                                  curvature_eps: float = DERIVED_EFFECTIVE_CURVATURE_EPS) -> np.ndarray:
    """
    Derive an effective diffusivity map directly from the full fitted concentration field.

    Pump-off:
        D_eff = (dC/dt) / (d2C/dx2)

    Pump-on:
        D_eff = (dC/dt + v dC/dx) / (d2C/dx2)
    """
    profiles = np.asarray(profiles, dtype=float)
    times_sec = np.asarray(times_sec, dtype=float)

    dCdt = compute_dcdt(profiles, times_sec)
    dCdx, d2Cdx2 = compute_spatial_derivatives(profiles, dx_mm)

    numerator = np.asarray(dCdt, dtype=float).copy()
    if v_vs_time is not None:
        v_arr = np.asarray(v_vs_time, dtype=float)
        if v_arr.ndim == 1 and v_arr.size == profiles.shape[0]:
            numerator = numerator + v_arr[:, None] * dCdx

    D_map = np.full_like(profiles, np.nan, dtype=float)
    valid = np.isfinite(numerator) & np.isfinite(d2Cdx2) & (np.abs(d2Cdx2) > float(curvature_eps))
    D_map[valid] = numerator[valid] / d2Cdx2[valid]

    d_lo, d_hi = d_bounds
    D_map[~np.isfinite(D_map)] = np.nan
    D_map[D_map < d_lo] = np.nan
    D_map[D_map > d_hi] = np.nan
    return D_map


def compute_derived_effective_diffusion_uncertainty_maps(fitted_profiles: np.ndarray,
                                                         times_sec: np.ndarray,
                                                         dx_mm: float,
                                                         v_vs_time: np.ndarray,
                                                         fitted_profiles_ci_low: np.ndarray,
                                                         fitted_profiles_ci_high: np.ndarray,
                                                         fitted_profiles_fixed_roi_ci_low: Optional[np.ndarray] = None,
                                                         fitted_profiles_fixed_roi_ci_high: Optional[np.ndarray] = None,
                                                         fitted_profiles_combined_ci_low: Optional[np.ndarray] = None,
                                                         fitted_profiles_combined_ci_high: Optional[np.ndarray] = None,
                                                         d_bounds: Tuple[float, float] = DERIVED_EFFECTIVE_D_BOUNDS,
                                                         curvature_eps: float = DERIVED_EFFECTIVE_CURVATURE_EPS) -> Dict[str, np.ndarray]:
    center = compute_derived_effective_diffusion_map_from_field(
        fitted_profiles, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
    )
    model_low = compute_derived_effective_diffusion_map_from_field(
        fitted_profiles_ci_low, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
    )
    model_high = compute_derived_effective_diffusion_map_from_field(
        fitted_profiles_ci_high, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
    )

    if fitted_profiles_fixed_roi_ci_low is not None and fitted_profiles_fixed_roi_ci_high is not None:
        fixed_roi_low = compute_derived_effective_diffusion_map_from_field(
            fitted_profiles_fixed_roi_ci_low, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
        )
        fixed_roi_high = compute_derived_effective_diffusion_map_from_field(
            fitted_profiles_fixed_roi_ci_high, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
        )
        fixed_roi_ci_width = np.abs(fixed_roi_high - fixed_roi_low)
        fixed_roi_std = fixed_roi_ci_width / (2.0 * 1.96)
    else:
        fixed_roi_low = np.full_like(center, np.nan, dtype=float)
        fixed_roi_high = np.full_like(center, np.nan, dtype=float)
        fixed_roi_ci_width = np.full_like(center, np.nan, dtype=float)
        fixed_roi_std = np.full_like(center, np.nan, dtype=float)

    if fitted_profiles_combined_ci_low is not None and fitted_profiles_combined_ci_high is not None:
        combined_low = compute_derived_effective_diffusion_map_from_field(
            fitted_profiles_combined_ci_low, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
        )
        combined_high = compute_derived_effective_diffusion_map_from_field(
            fitted_profiles_combined_ci_high, times_sec, dx_mm, v_vs_time, d_bounds=d_bounds, curvature_eps=curvature_eps
        )
        combined_ci_width = np.abs(combined_high - combined_low)
        combined_std = combined_ci_width / (2.0 * 1.96)
    else:
        combined_low = np.full_like(center, np.nan, dtype=float)
        combined_high = np.full_like(center, np.nan, dtype=float)
        combined_ci_width = np.full_like(center, np.nan, dtype=float)
        combined_std = np.full_like(center, np.nan, dtype=float)

    model_ci_width = np.abs(model_high - model_low)
    model_std = model_ci_width / (2.0 * 1.96)

    return {
        "D_map": center,
        "model_fit_std_map": model_std,
        "fixed_roi_std_map": fixed_roi_std,
        "combined_std_map": combined_std,
        "model_fit_ci_width_map": model_ci_width,
        "fixed_roi_ci_width_map": fixed_roi_ci_width,
        "combined_ci_width_map": combined_ci_width,
        "model_fit_ci_low_map": model_low,
        "model_fit_ci_high_map": model_high,
        "fixed_roi_ci_low_map": fixed_roi_low,
        "fixed_roi_ci_high_map": fixed_roi_high,
        "combined_ci_low_map": combined_low,
        "combined_ci_high_map": combined_high,
    }


def compute_pixelwise_apparent_diffusion_map(conc_stack: np.ndarray,
                                             times_sec: np.ndarray,
                                             dx_mm: float,
                                             depth_axis: str):
    dCdt = np.gradient(conc_stack, times_sec, axis=0)

    if depth_axis == "rows":
        dCdx = np.gradient(conc_stack, dx_mm, axis=1)
        d2Cdx2 = np.gradient(dCdx, dx_mm, axis=1)
    else:
        dCdx = np.gradient(conc_stack, dx_mm, axis=2)
        d2Cdx2 = np.gradient(dCdx, dx_mm, axis=2)

    D = np.full_like(conc_stack, np.nan, dtype=float)
    eps = 1e-10
    valid = np.abs(d2Cdx2) > eps
    D[valid] = dCdt[valid] / d2Cdx2[valid]
    D[D <= 0] = np.nan
    D[D > 10] = np.nan
    return D


# ============================================================
# PLOTTING
# ============================================================

def plot_concentration_vs_time(times_plot, mean_conc, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times_plot, mean_conc, marker="o", linewidth=2)
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel(f"Mean concentration ({CONCENTRATION_UNITS})")
    ax.set_title(f"{roi_display_name}: Concentration vs Time")
    ax.grid(True)
    save_figure(fig, out_path)


def plot_parameter_vs_time(times_plot, values, ylabel, title, out_path,
                           std_values=None, ci_low=None, ci_high=None, y_limits=None):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times_plot, values, marker="o", linewidth=2, label="Fit")

    if ci_low is not None and ci_high is not None:
        valid = np.isfinite(times_plot) & np.isfinite(ci_low) & np.isfinite(ci_high)
        if np.any(valid):
            ax.fill_between(times_plot[valid], ci_low[valid], ci_high[valid], alpha=0.2, label="95% CI")

    if std_values is not None:
        std_values = np.asarray(std_values, dtype=float)
        lower = values - std_values
        upper = values + std_values
        valid = np.isfinite(times_plot) & np.isfinite(lower) & np.isfinite(upper)
        if np.any(valid):
            ax.fill_between(times_plot[valid], lower[valid], upper[valid], alpha=0.15, label="±1 SD")

    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend()

    if y_limits is not None:
        ax.set_ylim(*y_limits)

    save_figure(fig, out_path)


def plot_transport_curves(times_plot, diff_curve, conv_curve, total_curve, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_plot, diff_curve, marker="o", linewidth=2, label="Diffusion")
    if conv_curve is not None and np.any(np.isfinite(conv_curve)):
        ax.plot(times_plot, conv_curve, marker="s", linewidth=2, label="Convection")
    if total_curve is not None and np.any(np.isfinite(total_curve)):
        ax.plot(times_plot, total_curve, marker="^", linewidth=2, label="Total")
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel("Mean |transport term|")
    ax.set_title(f"{roi_display_name}: Transport Terms vs Time")
    ax.grid(True)
    ax.legend()
    save_figure(fig, out_path)


def plot_diffusive_flux_curves(times_plot, mean_flux_mag, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times_plot, mean_flux_mag, marker="o", linewidth=2)
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel(r"Mean $|J_{diff}|$")
    ax.set_title(f"{roi_display_name}: Mean Diffusive Flux Magnitude vs Time")
    ax.grid(True)
    save_figure(fig, out_path)


def plot_mean_convection_diffusion_vs_time(times_plot, mean_conv_mag, mean_diff_mag, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(times_plot, mean_conv_mag, marker="s", linewidth=2, label="Convection")
    ax.plot(times_plot, mean_diff_mag, marker="o", linewidth=2, label="Diffusion")
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel("Mean |term|")
    ax.set_title(f"{roi_display_name}: Mean Convection vs Diffusion Magnitude")
    ax.grid(True)
    ax.legend()
    save_figure(fig, out_path)




def plot_convection_fraction_vs_time(times_plot, convection_fraction_vs_time, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times_plot, convection_fraction_vs_time, marker="o", linewidth=2)
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel("Mean convection fraction")
    ax.set_ylim(0, 1)
    ax.set_title(f"{roi_display_name}: Mean Convection Fraction vs Time")
    ax.grid(True)
    save_figure(fig, out_path)


def plot_concentration_depth_time_surface(times_plot, depth_mm, profiles, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    # Make x = depth, y = time, z = concentration
    X, T = np.meshgrid(depth_mm, times_plot, indexing="xy")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        X, T, profiles,
        cmap="viridis",
        edgecolor="none"
    )

    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel(f"Time ({TIME_UNIT})")
    ax.set_zlabel(f"Concentration ({CONCENTRATION_UNITS})")
    ax.set_title(f"{roi_display_name}: Concentration vs Depth vs Time")

    # Choose a view so depth appears on the left and time on the right
    ax.view_init(elev=25, azim=-60)

    fig.colorbar(surf, ax=ax, shrink=0.6, pad=0.1)
    save_figure(fig, out_path)


def plot_row_averaged_map(times_plot, depth_mm, map_data, title, cbar_label, out_path):
    fig, ax = plt.subplots(figsize=(8, 6))

    if PLOT_DEPTH_ZERO_AT_TOP:
        origin = "upper"
        extent = [times_plot[0], times_plot[-1], depth_mm[-1], depth_mm[0]]
    else:
        origin = "lower"
        extent = [times_plot[0], times_plot[-1], depth_mm[0], depth_mm[-1]]

    im = ax.imshow(
        map_data.T,
        aspect="auto",
        origin=origin,
        extent=extent,
        cmap="plasma"
    )
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    save_figure(fig, out_path)


def plot_convection_diffusion_ratio_map(times_plot, depth_mm, convection_term, diffusion_term, out_path, roi_name, eps=TRANSPORT_EPS):
    roi_display_name = format_roi_display_name(roi_name)
    ratio_map = np.abs(convection_term) / (np.abs(diffusion_term) + eps)
    ratio_map[~np.isfinite(ratio_map)] = np.nan

    if PLOT_DEPTH_ZERO_AT_TOP:
        origin = "upper"
        extent = [times_plot[0], times_plot[-1], depth_mm[-1], depth_mm[0]]
    else:
        origin = "lower"
        extent = [times_plot[0], times_plot[-1], depth_mm[0], depth_mm[-1]]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        ratio_map.T,
        aspect="auto",
        origin=origin,
        extent=extent,
        cmap="plasma",
        vmin=0
    )
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel("Depth (mm)")
    ax.set_title(f"{roi_display_name}: Convection-to-Diffusion Ratio Map")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$|v\,\partial C/\partial x| \,/\, |D\,\partial^2 C/\partial x^2|$")
    save_figure(fig, out_path)

    return ratio_map


def plot_pixelwise_map(image2d, title, cbar_label, out_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(image2d, cmap="plasma")
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    save_figure(fig, out_path)


def plot_multi_roi_summary(times_plot, roi_results, out_path):
    fig, ax = plt.subplots(figsize=(9, 6))

    for roi_name, res in roi_results.items():
        ax.plot(times_plot, res["mean_conc"], marker="o", linewidth=2, label=format_roi_display_name(roi_name))

    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel(f"Mean concentration ({CONCENTRATION_UNITS})")
    ax.set_title("Multi-ROI Concentration Comparison")
    ax.grid(True)
    ax.legend()
    save_figure(fig, out_path)


def plot_profile_fit_examples(depth_mm, measured_profiles, fitted_profiles, times_plot, out_path, roi_name,
                              r2_vs_time=None, fitted_profiles_std=None, fitted_profiles_ci_low=None, fitted_profiles_ci_high=None):
    """
    Plots up to 3 representative timepoints and optionally annotates R^2 and uncertainty bands.
    """
    valid_times = np.where(np.any(np.isfinite(fitted_profiles), axis=1))[0]
    if len(valid_times) == 0:
        return

    picks = np.unique(np.linspace(0, len(valid_times) - 1, min(3, len(valid_times))).astype(int))
    chosen = valid_times[picks]

    roi_display_name = format_roi_display_name(roi_name)
    fig, axes = plt.subplots(1, len(chosen), figsize=(5 * len(chosen), 4), squeeze=False)
    axes = axes.ravel()

    for ax, idx in zip(axes, chosen):
        ax.plot(depth_mm, measured_profiles[idx], "o", label="Measured")
        if fitted_profiles_ci_low is not None and fitted_profiles_ci_high is not None:
            low = fitted_profiles_ci_low[idx]
            high = fitted_profiles_ci_high[idx]
            valid = np.isfinite(depth_mm) & np.isfinite(low) & np.isfinite(high)
            if np.any(valid):
                ax.fill_between(depth_mm[valid], low[valid], high[valid], alpha=0.2, label="95% CI")
        if fitted_profiles_std is not None:
            std = fitted_profiles_std[idx]
            lower = fitted_profiles[idx] - std
            upper = fitted_profiles[idx] + std
            valid = np.isfinite(depth_mm) & np.isfinite(lower) & np.isfinite(upper)
            if np.any(valid):
                ax.fill_between(depth_mm[valid], lower[valid], upper[valid], alpha=0.15, label="±1 SD")
        ax.plot(depth_mm, fitted_profiles[idx], "-", linewidth=2, label="Fit")
        ax.set_xlabel("Depth (mm)")
        ax.set_ylabel(f"Concentration ({CONCENTRATION_UNITS})")
        title = f"{roi_display_name}\nTime = {times_plot[idx]:.2f} {TIME_UNIT}"
        if r2_vs_time is not None and idx < len(r2_vs_time) and np.isfinite(r2_vs_time[idx]):
            title += f"\n$R^2$ = {r2_vs_time[idx]:.4f}"
        ax.set_title(title)
        ax.grid(True)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend()

    save_figure(fig, out_path)


def compute_profile_fit_metrics(measured_profiles: np.ndarray, fitted_profiles: np.ndarray):
    residuals = measured_profiles - fitted_profiles
    rmse_vs_time = np.full(measured_profiles.shape[0], np.nan, dtype=float)
    r2_vs_time = np.full(measured_profiles.shape[0], np.nan, dtype=float)

    for i in range(measured_profiles.shape[0]):
        valid = np.isfinite(measured_profiles[i]) & np.isfinite(fitted_profiles[i])
        if np.any(valid):
            y_true = measured_profiles[i, valid]
            y_pred = fitted_profiles[i, valid]
            rmse_vs_time[i] = np.sqrt(np.mean((y_true - y_pred) ** 2))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            if ss_tot > 0:
                r2_vs_time[i] = 1.0 - (ss_res / ss_tot)

    return residuals, rmse_vs_time, r2_vs_time


def plot_profile_fit_rmse(times_plot, rmse_vs_time, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times_plot, rmse_vs_time, marker="o", linewidth=2)
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel(f"Profile fit RMSE ({CONCENTRATION_UNITS})")
    ax.set_title(f"{roi_display_name}: Profile Fit RMSE vs Time")
    ax.grid(True)
    save_figure(fig, out_path)



def plot_profile_fit_r2(times_plot, r2_vs_time, out_path, roi_name):
    roi_display_name = format_roi_display_name(roi_name)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(times_plot, r2_vs_time, marker="o", linewidth=2)
    ax.set_xlabel(f"Time ({TIME_UNIT})")
    ax.set_ylabel(r"Profile fit $R^2$")
    ax.set_title(f"{roi_display_name}: Profile Fit $R^2$ vs Time")
    ax.grid(True)
    save_figure(fig, out_path)

# ============================================================
# CSV EXPORT
# ============================================================

def save_curve_csv(out_path, times_plot, **curves):
    df = pd.DataFrame({"time": times_plot})
    for name, values in curves.items():
        df[name] = values
    df.to_csv(out_path, index=False)


def _safe_nanmean(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmean(arr)) if np.any(np.isfinite(arr)) else np.nan


def _safe_nanmedian(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmedian(arr)) if np.any(np.isfinite(arr)) else np.nan


def _safe_nanmax(arr):
    arr = np.asarray(arr, dtype=float)
    return float(np.nanmax(arr)) if np.any(np.isfinite(arr)) else np.nan


def build_regional_profile_uncertainty_summary(depth_mm: np.ndarray,
                                               fitted_profiles: np.ndarray,
                                               fitted_profiles_ci_low: np.ndarray,
                                               fitted_profiles_ci_high: np.ndarray,
                                               fitted_profiles_std: np.ndarray) -> pd.DataFrame:
    """
    Summarize fitted-profile signal and uncertainty by depth region.
    Regions are split into shallow / mid / deep thirds of the fitted depth range.
    """
    n_depth = int(fitted_profiles.shape[1]) if fitted_profiles.ndim == 2 else 0
    if n_depth == 0:
        return pd.DataFrame()

    ci_width = fitted_profiles_ci_high - fitted_profiles_ci_low
    edges = np.linspace(0, n_depth, 4, dtype=int)

    rows = []
    for region_name, start_idx, end_idx in zip(
        ["shallow", "mid", "deep"], edges[:-1], edges[1:]
    ):
        if end_idx <= start_idx:
            continue

        prof_reg = fitted_profiles[:, start_idx:end_idx]
        ciw_reg = ci_width[:, start_idx:end_idx]
        std_reg = fitted_profiles_std[:, start_idx:end_idx]

        with np.errstate(divide="ignore", invalid="ignore"):
            rel_ci_pct = 100.0 * ciw_reg / np.abs(prof_reg)
            rel_sd_pct = 100.0 * std_reg / np.abs(prof_reg)

        rel_ci_pct[~np.isfinite(rel_ci_pct)] = np.nan
        rel_sd_pct[~np.isfinite(rel_sd_pct)] = np.nan

        depth_start_mm = float(depth_mm[start_idx]) if len(depth_mm) > start_idx else np.nan
        depth_end_mm = float(depth_mm[end_idx - 1]) if len(depth_mm) >= end_idx and end_idx > start_idx else np.nan

        rows.append({
            "region": region_name,
            "depth_start_index": int(start_idx),
            "depth_end_index_inclusive": int(end_idx - 1),
            "n_depth_rows": int(end_idx - start_idx),
            "depth_start_mm": depth_start_mm,
            "depth_end_mm": depth_end_mm,
            "mean_fitted_concentration": _safe_nanmean(prof_reg),
            "median_fitted_concentration": _safe_nanmedian(prof_reg),
            "max_fitted_concentration": _safe_nanmax(prof_reg),
            "mean_ci_width": _safe_nanmean(ciw_reg),
            "median_ci_width": _safe_nanmedian(ciw_reg),
            "max_ci_width": _safe_nanmax(ciw_reg),
            "mean_std": _safe_nanmean(std_reg),
            "median_std": _safe_nanmedian(std_reg),
            "max_std": _safe_nanmax(std_reg),
            "mean_relative_ci_width_percent": _safe_nanmean(rel_ci_pct),
            "median_relative_ci_width_percent": _safe_nanmedian(rel_ci_pct),
            "max_relative_ci_width_percent": _safe_nanmax(rel_ci_pct),
            "mean_relative_std_percent": _safe_nanmean(rel_sd_pct),
            "median_relative_std_percent": _safe_nanmedian(rel_sd_pct),
            "max_relative_std_percent": _safe_nanmax(rel_sd_pct),
        })

    overall_prof = fitted_profiles
    overall_ciw = ci_width
    overall_std = fitted_profiles_std
    with np.errstate(divide="ignore", invalid="ignore"):
        overall_rel_ci_pct = 100.0 * overall_ciw / np.abs(overall_prof)
        overall_rel_sd_pct = 100.0 * overall_std / np.abs(overall_prof)
    overall_rel_ci_pct[~np.isfinite(overall_rel_ci_pct)] = np.nan
    overall_rel_sd_pct[~np.isfinite(overall_rel_sd_pct)] = np.nan

    rows.append({
        "region": "overall",
        "depth_start_index": 0,
        "depth_end_index_inclusive": int(n_depth - 1),
        "n_depth_rows": int(n_depth),
        "depth_start_mm": float(depth_mm[0]) if len(depth_mm) else np.nan,
        "depth_end_mm": float(depth_mm[-1]) if len(depth_mm) else np.nan,
        "mean_fitted_concentration": _safe_nanmean(overall_prof),
        "median_fitted_concentration": _safe_nanmedian(overall_prof),
        "max_fitted_concentration": _safe_nanmax(overall_prof),
        "mean_ci_width": _safe_nanmean(overall_ciw),
        "median_ci_width": _safe_nanmedian(overall_ciw),
        "max_ci_width": _safe_nanmax(overall_ciw),
        "mean_std": _safe_nanmean(overall_std),
        "median_std": _safe_nanmedian(overall_std),
        "max_std": _safe_nanmax(overall_std),
        "mean_relative_ci_width_percent": _safe_nanmean(overall_rel_ci_pct),
        "median_relative_ci_width_percent": _safe_nanmedian(overall_rel_ci_pct),
        "max_relative_ci_width_percent": _safe_nanmax(overall_rel_ci_pct),
        "mean_relative_std_percent": _safe_nanmean(overall_rel_sd_pct),
        "median_relative_std_percent": _safe_nanmedian(overall_rel_sd_pct),
        "max_relative_std_percent": _safe_nanmax(overall_rel_sd_pct),
    })

    return pd.DataFrame(rows)


def flatten_regional_uncertainty_summary(regional_df: pd.DataFrame) -> Dict[str, float]:
    """Flatten regional profile uncertainty summary rows into one summary-row dict."""
    flat = {}
    if regional_df is None or regional_df.empty:
        return flat

    for _, row in regional_df.iterrows():
        region = str(row.get("region", "unknown")).strip().lower().replace(" ", "_")
        for col in regional_df.columns:
            if col == "region":
                continue
            flat[f"regional_{region}_{col}"] = row[col]
    return flat


# ============================================================
# PER-ROI ANALYSIS
# ============================================================

def analyze_single_roi(roi_name: str,
                       roi: Tuple[int, int, int, int],
                       conc_stack: np.ndarray,
                       hu_stack: np.ndarray,
                       times_sec: np.ndarray,
                       times_plot: np.ndarray,
                       dx_mm: float,
                       output_root: str):
    roi_display_name = format_roi_display_name(roi_name)
    roi_folder = os.path.join(output_root, roi_name)
    ensure_dir(roi_folder)
    output_dirs = build_roi_output_dirs(roi_folder)

    r0, r1, c0, c1 = roi
    mean_conc = np.mean(conc_stack[:, r0:r1, c0:c1], axis=(1, 2))

    profiles = compute_depth_profiles(conc_stack, roi, DEPTH_AXIS)
    depth_mm = np.arange(profiles.shape[1]) * dx_mm

    (
        D_vs_time, Cs_vs_time, v_vs_time, fitted_profiles,
        D_std_vs_time, Cs_std_vs_time, v_std_vs_time,
        D_ci_low_vs_time, D_ci_high_vs_time,
        Cs_ci_low_vs_time, Cs_ci_high_vs_time,
        v_ci_low_vs_time, v_ci_high_vs_time,
        fitted_profiles_std, fitted_profiles_ci_low, fitted_profiles_ci_high
    ) = fit_profiles_over_time(
        depth_mm, profiles, times_sec
    )
    residual_profiles, profile_fit_rmse, profile_fit_r2 = compute_profile_fit_metrics(profiles, fitted_profiles)

    estimated_hu_noise_std = estimate_hu_noise_from_deep_region(hu_stack, roi, DEPTH_AXIS) if ENABLE_HU_NOISE_UNCERTAINTY else 0.0
    estimated_conc_noise_std = estimated_hu_noise_std / HU_PER_CONC if HU_PER_CONC not in (0, 0.0) else 0.0
    noise_unc = estimate_noise_uncertainty_via_refits(
        depth_mm, profiles, times_sec,
        sigma_conc=estimated_conc_noise_std,
        n_samples=HU_NOISE_MONTE_CARLO_SAMPLES,
        seed=UNCERTAINTY_RANDOM_SEED
    ) if ENABLE_HU_NOISE_UNCERTAINTY else {
        "D_std": np.full_like(D_vs_time, np.nan),
        "Cs_std": np.full_like(Cs_vs_time, np.nan),
        "v_std": np.full_like(v_vs_time, np.nan),
        "mean_conc_std": np.full_like(mean_conc, np.nan),
        "fitted_profiles_std": np.full_like(fitted_profiles, np.nan),
    }
    roi_unc = estimate_roi_sensitivity_uncertainty(
        conc_stack, roi, times_sec, dx_mm, DEPTH_AXIS, ROI_SENSITIVITY_SHIFTS
    ) if ENABLE_ROI_SENSITIVITY_UNCERTAINTY else {
        "mean_conc_std": np.full_like(mean_conc, np.nan),
        "D_std": np.full_like(D_vs_time, np.nan),
        "Cs_std": np.full_like(Cs_vs_time, np.nan),
        "v_std": np.full_like(v_vs_time, np.nan),
        "fitted_profiles_std": np.full_like(fitted_profiles, np.nan),
        "used_rois": [roi],
    }
    calibration_unc = estimate_calibration_uncertainty_via_refits(
        hu_stack=hu_stack,
        roi=roi,
        times_sec=times_sec,
        dx_mm=dx_mm,
        depth_axis=DEPTH_AXIS,
        hu_per_conc=HU_PER_CONC,
        hu_offset=HU_OFFSET,
        hu_per_conc_std=HU_PER_CONC_STD,
        hu_offset_std=HU_OFFSET_STD
    )

    D_fixed_roi_std_vs_time = combine_uncertainty_terms(D_std_vs_time, noise_unc["D_std"], calibration_unc["D_std"])
    Cs_fixed_roi_std_vs_time = combine_uncertainty_terms(Cs_std_vs_time, noise_unc["Cs_std"], calibration_unc["Cs_std"])
    v_fixed_roi_std_vs_time = combine_uncertainty_terms(v_std_vs_time, noise_unc["v_std"], calibration_unc["v_std"])
    mean_conc_fixed_roi_std = combine_uncertainty_terms(noise_unc["mean_conc_std"], calibration_unc["mean_conc_std"])
    fitted_profiles_fixed_roi_std = combine_uncertainty_terms(fitted_profiles_std, noise_unc["fitted_profiles_std"], calibration_unc["fitted_profiles_std"])

    D_combined_std_vs_time = combine_uncertainty_terms(D_std_vs_time, noise_unc["D_std"], roi_unc["D_std"], calibration_unc["D_std"])
    Cs_combined_std_vs_time = combine_uncertainty_terms(Cs_std_vs_time, noise_unc["Cs_std"], roi_unc["Cs_std"], calibration_unc["Cs_std"])
    v_combined_std_vs_time = combine_uncertainty_terms(v_std_vs_time, noise_unc["v_std"], roi_unc["v_std"], calibration_unc["v_std"])
    mean_conc_combined_std = combine_uncertainty_terms(noise_unc["mean_conc_std"], roi_unc["mean_conc_std"], calibration_unc["mean_conc_std"])
    fitted_profiles_combined_std = combine_uncertainty_terms(fitted_profiles_std, noise_unc["fitted_profiles_std"], roi_unc["fitted_profiles_std"], calibration_unc["fitted_profiles_std"])

    D_fixed_roi_ci_low_vs_time, D_fixed_roi_ci_high_vs_time = approx_ci_from_std(D_vs_time, D_fixed_roi_std_vs_time)
    Cs_fixed_roi_ci_low_vs_time, Cs_fixed_roi_ci_high_vs_time = approx_ci_from_std(Cs_vs_time, Cs_fixed_roi_std_vs_time)
    v_fixed_roi_ci_low_vs_time, v_fixed_roi_ci_high_vs_time = approx_ci_from_std(v_vs_time, v_fixed_roi_std_vs_time)
    mean_conc_fixed_roi_ci_low, mean_conc_fixed_roi_ci_high = approx_ci_from_std(mean_conc, mean_conc_fixed_roi_std)
    fitted_profiles_fixed_roi_ci_low, fitted_profiles_fixed_roi_ci_high = approx_ci_from_std(fitted_profiles, fitted_profiles_fixed_roi_std)

    D_combined_ci_low_vs_time, D_combined_ci_high_vs_time = approx_ci_from_std(D_vs_time, D_combined_std_vs_time)
    Cs_combined_ci_low_vs_time, Cs_combined_ci_high_vs_time = approx_ci_from_std(Cs_vs_time, Cs_combined_std_vs_time)
    v_combined_ci_low_vs_time, v_combined_ci_high_vs_time = approx_ci_from_std(v_vs_time, v_combined_std_vs_time)
    mean_conc_combined_ci_low, mean_conc_combined_ci_high = approx_ci_from_std(mean_conc, mean_conc_combined_std)
    fitted_profiles_combined_ci_low, fitted_profiles_combined_ci_high = approx_ci_from_std(fitted_profiles, fitted_profiles_combined_std)

    if ENABLE_TEMPORALLY_REGULARIZED_FIT:
        D_vs_time_reg, Cs_vs_time_reg, v_vs_time_reg, fitted_profiles_reg, D_vs_time_reg_plot, Cs_vs_time_reg_plot = fit_profiles_over_time_temporally_regularized(
            depth_mm, profiles, times_sec
        )
        residual_profiles_reg, profile_fit_rmse_reg, profile_fit_r2_reg = compute_profile_fit_metrics(profiles, fitted_profiles_reg)
        diffusion_term_reg, convection_term_reg, total_term_reg, diffusion_curve_reg, convection_curve_reg, total_curve_reg =             compute_transport_terms_from_fitted_params(
                fitted_profiles_reg, times_sec, dx_mm, np.where(np.isfinite(D_vs_time_reg), D_vs_time_reg, np.nan), v_vs_time_reg
            )
        _, diffusive_flux_reg, diffusive_flux_mag_reg, mean_flux_mag_reg =             compute_diffusive_flux_map_from_fitted_params(
                fitted_profiles_reg, dx_mm, np.where(np.isfinite(D_vs_time_reg), D_vs_time_reg, np.nan)
            )
    else:
        D_vs_time_reg = np.full_like(times_sec, np.nan, dtype=float)
        Cs_vs_time_reg = np.full_like(times_sec, np.nan, dtype=float)
        v_vs_time_reg = np.full_like(times_sec, np.nan, dtype=float)
        fitted_profiles_reg = np.full_like(profiles, np.nan, dtype=float)
        residual_profiles_reg = np.full_like(profiles, np.nan, dtype=float)
        profile_fit_rmse_reg = np.full_like(times_sec, np.nan, dtype=float)
        D_vs_time_reg_plot = D_vs_time_reg.copy()
        Cs_vs_time_reg_plot = Cs_vs_time_reg.copy()
        diffusion_term_reg = np.full_like(profiles, np.nan, dtype=float)
        convection_term_reg = np.full_like(profiles, np.nan, dtype=float)
        total_term_reg = np.full_like(profiles, np.nan, dtype=float)
        diffusion_curve_reg = np.full_like(times_sec, np.nan, dtype=float)
        convection_curve_reg = np.full_like(times_sec, np.nan, dtype=float)
        total_curve_reg = np.full_like(times_sec, np.nan, dtype=float)
        diffusive_flux_reg = np.full_like(profiles, np.nan, dtype=float)
        diffusive_flux_mag_reg = np.full_like(profiles, np.nan, dtype=float)
        mean_flux_mag_reg = np.full_like(times_sec, np.nan, dtype=float)

    diffusion_term, convection_term, total_term, diffusion_curve, convection_curve, total_curve = \
        compute_transport_terms_from_fitted_params(
            profiles, times_sec, dx_mm, D_vs_time, v_vs_time
        )

    dCdx, diffusive_flux, diffusive_flux_mag, mean_flux_mag = \
        compute_diffusive_flux_map_from_fitted_params(
            profiles, dx_mm, D_vs_time
        )

    flux_model_unc = compute_diffusive_flux_uncertainty_component_from_std(
        fitted_profiles, dx_mm, D_vs_time,
        D_std_vs_time=D_std_vs_time,
        profiles_std=fitted_profiles_std
    )
    flux_noise_unc = compute_diffusive_flux_uncertainty_component_from_std(
        fitted_profiles, dx_mm, D_vs_time,
        D_std_vs_time=noise_unc["D_std"],
        profiles_std=noise_unc["fitted_profiles_std"]
    )
    flux_roi_unc = compute_diffusive_flux_uncertainty_component_from_std(
        fitted_profiles, dx_mm, D_vs_time,
        D_std_vs_time=roi_unc["D_std"],
        profiles_std=roi_unc["fitted_profiles_std"]
    )
    flux_calibration_unc = compute_diffusive_flux_uncertainty_component_from_std(
        fitted_profiles, dx_mm, D_vs_time,
        D_std_vs_time=calibration_unc["D_std"],
        profiles_std=calibration_unc["fitted_profiles_std"]
    )

    diffusive_flux_model_fit_std_map = flux_model_unc["signed_std_map"]
    diffusive_flux_hu_noise_std_map = flux_noise_unc["signed_std_map"]
    diffusive_flux_roi_sensitivity_std_map = flux_roi_unc["signed_std_map"]
    diffusive_flux_calibration_std_map = flux_calibration_unc["signed_std_map"]
    diffusive_flux_fixed_roi_std_map = combine_uncertainty_terms(
        diffusive_flux_model_fit_std_map,
        diffusive_flux_hu_noise_std_map,
        diffusive_flux_calibration_std_map
    )
    diffusive_flux_combined_std_map = combine_uncertainty_terms(
        diffusive_flux_fixed_roi_std_map,
        diffusive_flux_roi_sensitivity_std_map
    )

    diffusive_flux_magnitude_model_fit_std_map = flux_model_unc["magnitude_std_map"]
    diffusive_flux_magnitude_hu_noise_std_map = flux_noise_unc["magnitude_std_map"]
    diffusive_flux_magnitude_roi_sensitivity_std_map = flux_roi_unc["magnitude_std_map"]
    diffusive_flux_magnitude_calibration_std_map = flux_calibration_unc["magnitude_std_map"]
    diffusive_flux_magnitude_fixed_roi_std_map = combine_uncertainty_terms(
        diffusive_flux_magnitude_model_fit_std_map,
        diffusive_flux_magnitude_hu_noise_std_map,
        diffusive_flux_magnitude_calibration_std_map
    )
    diffusive_flux_magnitude_combined_std_map = combine_uncertainty_terms(
        diffusive_flux_magnitude_fixed_roi_std_map,
        diffusive_flux_magnitude_roi_sensitivity_std_map
    )

    diffusive_flux_fixed_roi_ci_low_map, diffusive_flux_fixed_roi_ci_high_map = approx_ci_from_std(
        diffusive_flux, diffusive_flux_fixed_roi_std_map
    )
    diffusive_flux_combined_ci_low_map, diffusive_flux_combined_ci_high_map = approx_ci_from_std(
        diffusive_flux, diffusive_flux_combined_std_map
    )
    diffusive_flux_magnitude_fixed_roi_ci_low_map, diffusive_flux_magnitude_fixed_roi_ci_high_map = approx_ci_from_std(
        diffusive_flux_mag, diffusive_flux_magnitude_fixed_roi_std_map
    )
    diffusive_flux_magnitude_combined_ci_low_map, diffusive_flux_magnitude_combined_ci_high_map = approx_ci_from_std(
        diffusive_flux_mag, diffusive_flux_magnitude_combined_std_map
    )

    diffusive_flux_magnitude_fixed_roi_ci_width_map = (
        diffusive_flux_magnitude_fixed_roi_ci_high_map - diffusive_flux_magnitude_fixed_roi_ci_low_map
    )
    diffusive_flux_magnitude_combined_ci_width_map = (
        diffusive_flux_magnitude_combined_ci_high_map - diffusive_flux_magnitude_combined_ci_low_map
    )
    diffusive_flux_magnitude_fixed_roi_relative_percent_map = compute_relative_uncertainty_percent_map(
        diffusive_flux_mag,
        diffusive_flux_magnitude_fixed_roi_ci_width_map
    )
    diffusive_flux_magnitude_combined_relative_percent_map = compute_relative_uncertainty_percent_map(
        diffusive_flux_mag,
        diffusive_flux_magnitude_combined_ci_width_map
    )

    flux_unc_summary = compute_diffusive_flux_uncertainty_summary_from_maps(
        diffusive_flux,
        diffusive_flux_mag,
        diffusive_flux_magnitude_fixed_roi_std_map,
        diffusive_flux_magnitude_combined_std_map
    )
    mean_flux_mag_fixed_roi_std = flux_unc_summary["mean_fixed_roi_std"]
    mean_flux_mag_fixed_roi_ci_low = flux_unc_summary["mean_fixed_roi_ci_low"]
    mean_flux_mag_fixed_roi_ci_high = flux_unc_summary["mean_fixed_roi_ci_high"]
    mean_flux_mag_combined_std = flux_unc_summary["mean_combined_std"]
    mean_flux_mag_combined_ci_low = flux_unc_summary["mean_combined_ci_low"]
    mean_flux_mag_combined_ci_high = flux_unc_summary["mean_combined_ci_high"]

    peclet_vs_time, peclet_length_mm = compute_peclet_vs_time(
        depth_mm, fitted_profiles, D_vs_time, v_vs_time
    )

    if SAVE_EFFECTIVE_DIFFUSION_MAP:
        profiles_for_local_map = smooth_profiles_for_local_map(profiles)
        D_local_map, Cs_local_map, v_local_map, pred_local_map, local_fit_rmse_map = fit_local_effective_transport_maps(
            depth_mm, profiles_for_local_map, times_sec, window_size=LOCAL_EFFECTIVE_MAP_WINDOW_SIZE
        )
    else:
        D_local_map = np.full_like(profiles, np.nan, dtype=float)
        Cs_local_map = np.full_like(profiles, np.nan, dtype=float)
        v_local_map = np.full_like(profiles, np.nan, dtype=float)
        pred_local_map = np.full_like(profiles, np.nan, dtype=float)
        local_fit_rmse_map = np.full_like(profiles, np.nan, dtype=float)

    if SAVE_DERIVED_EFFECTIVE_DIFFUSION_MAP:
        derived_effective = compute_derived_effective_diffusion_uncertainty_maps(
            fitted_profiles=fitted_profiles,
            times_sec=times_sec,
            dx_mm=dx_mm,
            v_vs_time=v_vs_time,
            fitted_profiles_ci_low=fitted_profiles_ci_low,
            fitted_profiles_ci_high=fitted_profiles_ci_high,
            fitted_profiles_fixed_roi_ci_low=fitted_profiles_fixed_roi_ci_low,
            fitted_profiles_fixed_roi_ci_high=fitted_profiles_fixed_roi_ci_high,
            fitted_profiles_combined_ci_low=fitted_profiles_combined_ci_low,
            fitted_profiles_combined_ci_high=fitted_profiles_combined_ci_high,
            d_bounds=DERIVED_EFFECTIVE_D_BOUNDS,
            curvature_eps=DERIVED_EFFECTIVE_CURVATURE_EPS
        )
        derived_D_map = derived_effective["D_map"]
        derived_D_model_fit_std_map = derived_effective["model_fit_std_map"]
        derived_D_fixed_roi_std_map = derived_effective["fixed_roi_std_map"]
        derived_D_combined_std_map = derived_effective["combined_std_map"]
        derived_D_model_fit_ci_width_map = derived_effective["model_fit_ci_width_map"]
        derived_D_fixed_roi_ci_width_map = derived_effective["fixed_roi_ci_width_map"]
        derived_D_combined_ci_width_map = derived_effective["combined_ci_width_map"]
    else:
        derived_D_map = np.full_like(profiles, np.nan, dtype=float)
        derived_D_model_fit_std_map = np.full_like(profiles, np.nan, dtype=float)
        derived_D_fixed_roi_std_map = np.full_like(profiles, np.nan, dtype=float)
        derived_D_combined_std_map = np.full_like(profiles, np.nan, dtype=float)
        derived_D_model_fit_ci_width_map = np.full_like(profiles, np.nan, dtype=float)
        derived_D_fixed_roi_ci_width_map = np.full_like(profiles, np.nan, dtype=float)
        derived_D_combined_ci_width_map = np.full_like(profiles, np.nan, dtype=float)

    if ENABLE_GLOBAL_SPATIOTEMPORAL_FIT:
        D_global_fit, Cs_global_vs_time, v_global_fit, fitted_profiles_global, cs_control_times_sec, cs_control_values = fit_global_spatiotemporal_profiles(
            depth_mm, profiles, times_sec
        )
        residual_profiles_global, profile_fit_rmse_global, profile_fit_r2_global = compute_profile_fit_metrics(profiles, fitted_profiles_global)
        D_global_vs_time = np.full_like(times_sec, D_global_fit, dtype=float)
        v_global_vs_time = np.full_like(times_sec, v_global_fit, dtype=float)
        diffusion_term_global, convection_term_global, total_term_global, diffusion_curve_global, convection_curve_global, total_curve_global = compute_transport_terms_from_fitted_params(
            fitted_profiles_global, times_sec, dx_mm, D_global_vs_time, v_global_vs_time
        )
        _, diffusive_flux_global, diffusive_flux_mag_global, mean_flux_mag_global = compute_diffusive_flux_map_from_fitted_params(
            fitted_profiles_global, dx_mm, D_global_vs_time
        )
    else:
        D_global_fit = np.nan
        Cs_global_vs_time = np.full_like(times_sec, np.nan, dtype=float)
        v_global_fit = np.nan
        fitted_profiles_global = np.full_like(profiles, np.nan, dtype=float)
        residual_profiles_global = np.full_like(profiles, np.nan, dtype=float)
        profile_fit_rmse_global = np.full_like(times_sec, np.nan, dtype=float)
        cs_control_times_sec = np.array([], dtype=float)
        cs_control_values = np.array([], dtype=float)
        diffusion_term_global = np.full_like(profiles, np.nan, dtype=float)
        convection_term_global = np.full_like(profiles, np.nan, dtype=float)
        total_term_global = np.full_like(profiles, np.nan, dtype=float)
        diffusion_curve_global = np.full_like(times_sec, np.nan, dtype=float)
        convection_curve_global = np.full_like(times_sec, np.nan, dtype=float)
        total_curve_global = np.full_like(times_sec, np.nan, dtype=float)
        diffusive_flux_global = np.full_like(profiles, np.nan, dtype=float)
        diffusive_flux_mag_global = np.full_like(profiles, np.nan, dtype=float)
        mean_flux_mag_global = np.full_like(times_sec, np.nan, dtype=float)

    if not PUMP_ON:
        convection_curve[:] = np.nan
        total_curve = diffusion_curve
        if ENABLE_GLOBAL_SPATIOTEMPORAL_FIT:
            convection_curve_global[:] = np.nan
            total_curve_global = diffusion_curve_global

    plot_concentration_vs_time(
        times_plot, mean_conc,
        roi_output_path(output_dirs, "concentration_vs_time.png"),
        roi_name
    )

    plot_parameter_vs_time(
        times_plot, mean_conc,
        f"Mean concentration ({CONCENTRATION_UNITS})",
        f"{roi_display_name}: Concentration vs Time (fixed ROI uncertainty)",
        roi_output_path(output_dirs, "concentration_vs_time_fixed_roi_uncertainty.png"),
        std_values=mean_conc_fixed_roi_std,
        ci_low=mean_conc_fixed_roi_ci_low,
        ci_high=mean_conc_fixed_roi_ci_high
    )

    plot_parameter_vs_time(
        times_plot, D_vs_time,
        r"Fitted effective diffusivity (mm$^2$/s)",
        f"{roi_display_name}: Effective Diffusivity vs Time",
        roi_output_path(output_dirs, "effective_diffusivity_vs_time.png"),
        std_values=D_std_vs_time,
        ci_low=D_ci_low_vs_time,
        ci_high=D_ci_high_vs_time
    )

    plot_parameter_vs_time(
        times_plot, Cs_vs_time,
        f"Effective fitted boundary concentration, C$_s$ ({CONCENTRATION_UNITS})",
        f"{roi_display_name}: Effective Fitted Boundary Concentration vs Time",
        roi_output_path(output_dirs, "fitted_Cs_vs_time.png"),
        std_values=Cs_std_vs_time,
        ci_low=Cs_ci_low_vs_time,
        ci_high=Cs_ci_high_vs_time,
        y_limits=(15,70)
    )

    plot_parameter_vs_time(
        times_plot, D_vs_time,
        r"Fitted effective diffusivity (mm$^2$/s)",
        f"{roi_display_name}: Effective Diffusivity vs Time (fixed ROI uncertainty)",
        roi_output_path(output_dirs, "effective_diffusivity_vs_time_fixed_roi_uncertainty.png"),
        std_values=D_fixed_roi_std_vs_time,
        ci_low=D_fixed_roi_ci_low_vs_time,
        ci_high=D_fixed_roi_ci_high_vs_time
    )

    plot_parameter_vs_time(
        times_plot, D_vs_time,
        r"Fitted effective diffusivity (mm$^2$/s)",
        f"{roi_display_name}: Effective Diffusivity vs Time (combined uncertainty)",
        roi_output_path(output_dirs, "effective_diffusivity_vs_time_combined_uncertainty.png"),
        std_values=D_combined_std_vs_time,
        ci_low=D_combined_ci_low_vs_time,
        ci_high=D_combined_ci_high_vs_time
    )

    plot_parameter_vs_time(
        times_plot, Cs_vs_time,
        f"Effective fitted boundary concentration, C$_s$ ({CONCENTRATION_UNITS})",
        f"{roi_display_name}: Effective Fitted Boundary Concentration vs Time (fixed ROI uncertainty)",
        roi_output_path(output_dirs, "fitted_Cs_vs_time_fixed_roi_uncertainty.png"),
        std_values=Cs_fixed_roi_std_vs_time,
        ci_low=Cs_fixed_roi_ci_low_vs_time,
        ci_high=Cs_fixed_roi_ci_high_vs_time,
        y_limits=(15,70)
    )

    plot_parameter_vs_time(
        times_plot, Cs_vs_time,
        f"Effective fitted boundary concentration, C$_s$ ({CONCENTRATION_UNITS})",
        f"{roi_display_name}: Effective Fitted Boundary Concentration vs Time (combined uncertainty)",
        roi_output_path(output_dirs, "fitted_Cs_vs_time_combined_uncertainty.png"),
        std_values=Cs_combined_std_vs_time,
        ci_low=Cs_combined_ci_low_vs_time,
        ci_high=Cs_combined_ci_high_vs_time,
        y_limits=(15,70)
    )

    if PUMP_ON:
        plot_parameter_vs_time(
            times_plot, v_vs_time,
            r"Velocity (mm/s)",
            f"{roi_display_name}: Velocity vs Time",
            roi_output_path(output_dirs, "velocity_vs_time.png"),
            std_values=v_std_vs_time,
            ci_low=v_ci_low_vs_time,
            ci_high=v_ci_high_vs_time
        )
        plot_parameter_vs_time(
            times_plot, v_vs_time,
            r"Velocity (mm/s)",
            f"{roi_display_name}: Velocity vs Time (fixed ROI uncertainty)",
            roi_output_path(output_dirs, "velocity_vs_time_fixed_roi_uncertainty.png"),
            std_values=v_fixed_roi_std_vs_time,
            ci_low=v_fixed_roi_ci_low_vs_time,
            ci_high=v_fixed_roi_ci_high_vs_time
        )
        plot_parameter_vs_time(
            times_plot, v_vs_time,
            r"Velocity (mm/s)",
            f"{roi_display_name}: Velocity vs Time (combined uncertainty)",
            roi_output_path(output_dirs, "velocity_vs_time_combined_uncertainty.png"),
            std_values=v_combined_std_vs_time,
            ci_low=v_combined_ci_low_vs_time,
            ci_high=v_combined_ci_high_vs_time
        )

    plot_transport_curves(
        times_plot, diffusion_curve, convection_curve if PUMP_ON else None, total_curve,
        roi_output_path(output_dirs, "transport_terms_vs_time.png"),
        roi_name
    )

    plot_profile_fit_rmse(
        times_plot,
        profile_fit_rmse,
        roi_output_path(output_dirs, "profile_fit_rmse_vs_time.png"),
        roi_name
    )

    if ENABLE_GLOBAL_SPATIOTEMPORAL_FIT:
        plot_profile_fit_rmse(
            times_plot,
            profile_fit_rmse_global,
            roi_output_path(output_dirs, "global_spatiotemporal_profile_fit_rmse_vs_time.png"),
            f"{roi_display_name}: Global Spatiotemporal"
        )
        plot_rmse_comparison(
            times_plot,
            profile_fit_rmse,
            profile_fit_rmse_global,
            roi_output_path(output_dirs, "profile_fit_rmse_comparison.png"),
            roi_name,
            rmse_regularized=profile_fit_rmse_reg if ENABLE_TEMPORALLY_REGULARIZED_FIT else None
        )

    plot_diffusive_flux_curves(
        times_plot,
        mean_flux_mag,
        roi_output_path(output_dirs, "mean_diffusive_flux_magnitude_vs_time.png"),
        roi_name
    )

    if PUMP_ON:
        plot_mean_convection_diffusion_vs_time(
            times_plot,
            convection_curve,
            diffusion_curve,
            roi_output_path(output_dirs, "mean_convection_vs_diffusion_magnitude_vs_time.png"),
            roi_name
        )
        plot_parameter_vs_time(
            times_plot,
            peclet_vs_time,
            "Péclet number, Pe = |v|L/D",
            f"{roi_display_name}: Péclet Number vs Time",
            roi_output_path(output_dirs, "peclet_number_vs_time.png")
        )

    plot_concentration_depth_time_surface(
        times_plot, depth_mm, profiles,
        roi_output_path(output_dirs, "concentration_depth_time_3d.png"),
        roi_name
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles,
        f"{roi_display_name}: Fitted Concentration Profiles",
        f"Concentration ({CONCENTRATION_UNITS})",
        roi_output_path(output_dirs, "fitted_profiles_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_std,
        f"{roi_display_name}: Fitted Profile Standard Deviation Map",
        f"SD ({CONCENTRATION_UNITS})",
        roi_output_path(output_dirs, "fitted_profiles_std_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_fixed_roi_std,
        f"{roi_display_name}: Fitted Profile Fixed-ROI Uncertainty Map",
        f"Fixed-ROI SD ({CONCENTRATION_UNITS})",
        roi_output_path(output_dirs, "fitted_profiles_fixed_roi_std_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_combined_std,
        f"{roi_display_name}: Fitted Profile Combined Uncertainty Map",
        f"Combined SD ({CONCENTRATION_UNITS})",
        roi_output_path(output_dirs, "fitted_profiles_combined_std_map.png")
    )

    fitted_profiles_ci_width = fitted_profiles_ci_high - fitted_profiles_ci_low
    fitted_profiles_fixed_roi_ci_width = fitted_profiles_fixed_roi_ci_high - fitted_profiles_fixed_roi_ci_low
    fitted_profiles_relative_ci_percent_map = compute_relative_uncertainty_percent_map(
        fitted_profiles,
        fitted_profiles_ci_width
    )
    fitted_profiles_fixed_roi_relative_percent_map = compute_relative_uncertainty_percent_map(
        fitted_profiles,
        fitted_profiles_fixed_roi_ci_width
    )
    fitted_profiles_combined_relative_percent_map = compute_relative_uncertainty_percent_map(
        fitted_profiles,
        fitted_profiles_combined_std
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_ci_width,
        f"{roi_display_name}: Fitted Profile 95% CI Width Map",
        f"95% CI width ({CONCENTRATION_UNITS})",
        roi_output_path(output_dirs, "fitted_profiles_ci_width_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_relative_ci_percent_map,
        f"{roi_display_name}: Relative Fitted Profile 95% CI Map",
        "Relative 95% CI (%)",
        roi_output_path(output_dirs, "fitted_profiles_relative_ci_percent_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_fixed_roi_ci_width,
        f"{roi_display_name}: Fitted Profile Fixed-ROI 95% CI Width Map",
        f"Fixed-ROI 95% CI width ({CONCENTRATION_UNITS})",
        roi_output_path(output_dirs, "fitted_profiles_fixed_roi_ci_width_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_fixed_roi_relative_percent_map,
        f"{roi_display_name}: Relative Fitted Profile Fixed-ROI Uncertainty Map",
        "Relative fixed-ROI uncertainty (%)",
        roi_output_path(output_dirs, "fitted_profiles_fixed_roi_relative_percent_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        fitted_profiles_combined_relative_percent_map,
        f"{roi_display_name}: Relative Fitted Profile Combined Uncertainty Map",
        "Relative combined uncertainty (%)",
        roi_output_path(output_dirs, "fitted_profiles_combined_relative_percent_map.png")
    )

    if ENABLE_GLOBAL_SPATIOTEMPORAL_FIT:
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            fitted_profiles_global,
            f"{roi_name}: Global Spatiotemporal Fitted Profiles",
            f"Concentration ({CONCENTRATION_UNITS})",
            roi_output_path(output_dirs, "global_spatiotemporal_fitted_profiles_map.png")
        )
        plot_parameter_vs_time(
            times_plot, Cs_global_vs_time,
            f"Global smooth effective fitted boundary concentration, C$_s$(t) ({CONCENTRATION_UNITS})",
            f"{roi_name}: Global Spatiotemporal Smooth Effective Fitted Boundary Concentration",
            roi_output_path(output_dirs, "global_spatiotemporal_Cs_vs_time.png")
        )
        plot_profile_fit_r2(
            times_plot,
            profile_fit_r2_global,
            roi_output_path(output_dirs, "global_spatiotemporal_profile_fit_r2_vs_time.png"),
            f"{roi_display_name}: Global Spatiotemporal"
        )
        plot_profile_fit_examples(
            depth_mm, profiles, fitted_profiles_global, times_plot,
            roi_output_path(output_dirs, "global_spatiotemporal_profile_fit_examples.png"),
            roi_display_name + " Global",
            r2_vs_time=profile_fit_r2_global
        )

    if ENABLE_TEMPORALLY_REGULARIZED_FIT:
        plot_parameter_vs_time(
            times_plot, D_vs_time_reg_plot,
            r"Regularized fitted effective diffusivity (mm$^2$/s)",
            f"{roi_display_name}: Temporally Regularized Effective Diffusivity vs Time",
            roi_output_path(output_dirs, "temporally_regularized_effective_diffusivity_vs_time.png")
        )
        plot_parameter_vs_time(
            times_plot, Cs_vs_time_reg_plot,
            f"Temporally regularized effective fitted boundary concentration, C$_s$ ({CONCENTRATION_UNITS})",
            f"{roi_display_name}: Temporally Regularized Effective Fitted Boundary Concentration vs Time",
            roi_output_path(output_dirs, "temporally_regularized_fitted_Cs_vs_time.png")
        )
        plot_profile_fit_rmse(
            times_plot,
            profile_fit_rmse_reg,
            roi_output_path(output_dirs, "temporally_regularized_profile_fit_rmse_vs_time.png"),
            f"{roi_display_name}: Temporally Regularized"
        )
        plot_profile_fit_r2(
            times_plot,
            profile_fit_r2_reg,
            roi_output_path(output_dirs, "temporally_regularized_profile_fit_r2_vs_time.png"),
            f"{roi_display_name}: Temporally Regularized"
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            fitted_profiles_reg,
            f"{roi_name}: Temporally Regularized Fitted Profiles",
            f"Concentration ({CONCENTRATION_UNITS})",
            roi_output_path(output_dirs, "temporally_regularized_fitted_profiles_map.png")
        )
        plot_profile_fit_examples(
            depth_mm, profiles, fitted_profiles_reg, times_plot,
            roi_output_path(output_dirs, "temporally_regularized_profile_fit_examples.png"),
            roi_display_name + " Temporally Regularized",
            r2_vs_time=profile_fit_r2_reg
        )
        plot_transport_curves(
            times_plot, diffusion_curve_reg, convection_curve_reg if PUMP_ON else None, total_curve_reg,
            roi_output_path(output_dirs, "temporally_regularized_transport_terms_vs_time.png"),
            roi_display_name + " Temporally Regularized"
        )
        plot_diffusive_flux_curves(
            times_plot,
            mean_flux_mag_reg,
            roi_output_path(output_dirs, "temporally_regularized_mean_diffusive_flux_magnitude_vs_time.png"),
            roi_display_name + " Temporally Regularized"
        )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusion_term,
        f"{roi_display_name}: Row-Averaged Diffusion Term Map",
        "Diffusion term",
        roi_output_path(output_dirs, "row_averaged_diffusion_term_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_mag,
        f"{roi_display_name}: Row-Averaged Diffusive Flux Magnitude Map",
        r"$|J_{diff}| = D |\partial C / \partial x|$",
        roi_output_path(output_dirs, "row_averaged_diffusive_flux_magnitude_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux,
        f"{roi_display_name}: Row-Averaged Signed Diffusive Flux Map",
        r"$J_{diff} = -D \partial C / \partial x$",
        roi_output_path(output_dirs, "row_averaged_signed_diffusive_flux_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_magnitude_fixed_roi_std_map,
        f"{roi_display_name}: Diffusive Flux Magnitude Fixed-ROI Uncertainty Map",
        r"Fixed-ROI SD of $|J_{diff}|$",
        roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_std_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_magnitude_combined_std_map,
        f"{roi_display_name}: Diffusive Flux Magnitude Combined Uncertainty Map",
        r"Combined SD of $|J_{diff}|$",
        roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_std_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_magnitude_fixed_roi_ci_width_map,
        f"{roi_display_name}: Diffusive Flux Magnitude Fixed-ROI 95% CI Width Map",
        r"Fixed-ROI 95% CI width of $|J_{diff}|$",
        roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_ci_width_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_magnitude_combined_ci_width_map,
        f"{roi_display_name}: Diffusive Flux Magnitude Combined 95% CI Width Map",
        r"Combined 95% CI width of $|J_{diff}|$",
        roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_ci_width_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_magnitude_fixed_roi_relative_percent_map,
        f"{roi_display_name}: Relative Diffusive Flux Magnitude Fixed-ROI 95% CI Map",
        "Relative fixed-ROI 95% CI (%)",
        roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_relative_percent_map.png")
    )

    plot_row_averaged_map(
        times_plot,
        depth_mm,
        diffusive_flux_magnitude_combined_relative_percent_map,
        f"{roi_display_name}: Relative Diffusive Flux Magnitude Combined 95% CI Map",
        "Relative combined 95% CI (%)",
        roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_relative_percent_map.png")
    )

    if SAVE_EFFECTIVE_DIFFUSION_MAP:
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            D_local_map,
            f"{roi_display_name}: Local Effective Diffusivity Map",
            r"Local fitted $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "local_effective_diffusivity_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            local_fit_rmse_map,
            f"{roi_display_name}: Local Effective Fit RMSE Map",
            f"Local fit RMSE ({CONCENTRATION_UNITS})",
            roi_output_path(output_dirs, "local_effective_fit_rmse_map.png")
        )

    if SAVE_DERIVED_EFFECTIVE_DIFFUSION_MAP:
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_map,
            f"{roi_display_name}: Derived Effective Diffusivity Map from Full Fitted Field",
            r"Derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_model_fit_std_map,
            f"{roi_display_name}: Derived Effective Diffusivity Model-Fit Uncertainty Map",
            r"Model-fit SD of derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_model_fit_std_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_fixed_roi_std_map,
            f"{roi_display_name}: Derived Effective Diffusivity Fixed-ROI Uncertainty Map",
            r"Fixed-ROI SD of derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_fixed_roi_std_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_combined_std_map,
            f"{roi_display_name}: Derived Effective Diffusivity Combined Uncertainty Map",
            r"Combined SD of derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_combined_std_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_model_fit_ci_width_map,
            f"{roi_display_name}: Derived Effective Diffusivity Model-Fit 95% CI Width Map",
            r"Model-fit 95% CI width of derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_model_fit_ci_width_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_fixed_roi_ci_width_map,
            f"{roi_display_name}: Derived Effective Diffusivity Fixed-ROI 95% CI Width Map",
            r"Fixed-ROI 95% CI width of derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_fixed_roi_ci_width_map.png")
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            derived_D_combined_ci_width_map,
            f"{roi_display_name}: Derived Effective Diffusivity Combined 95% CI Width Map",
            r"Combined 95% CI width of derived $D_{eff}$ (mm$^2$/s)",
            roi_output_path(output_dirs, "derived_effective_diffusivity_combined_ci_width_map.png")
        )

    if PUMP_ON:
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            convection_term,
            f"{roi_display_name}: Row-Averaged Convection Term Map",
            "Convection term",
            roi_output_path(output_dirs, "row_averaged_convection_term_map.png")
        )

        convection_ci_low_map, convection_ci_high_map, convection_ci_width_map = compute_convection_ci_maps_from_profile_ci(
            fitted_profiles_ci_low,
            fitted_profiles_ci_high,
            dx_mm,
            v_vs_time
        )
        convection_relative_ci_percent_map = compute_relative_uncertainty_percent_map(
            convection_term,
            convection_ci_width_map
        )

        plot_row_averaged_map(
            times_plot,
            depth_mm,
            convection_ci_width_map,
            f"{roi_display_name}: Convection Term 95% CI Width Map",
            "95% CI width (convection term)",
            roi_output_path(output_dirs, "convection_term_ci_width_map.png")
        )

        plot_row_averaged_map(
            times_plot,
            depth_mm,
            convection_relative_ci_percent_map,
            f"{roi_display_name}: Relative Convection Term 95% CI Map",
            "Relative 95% CI (%)",
            roi_output_path(output_dirs, "convection_term_relative_ci_percent_map.png")
        )
        convection_diffusion_ratio_map = plot_convection_diffusion_ratio_map(
            times_plot,
            depth_mm,
            convection_term,
            diffusion_term,
            roi_output_path(output_dirs, "convection_to_diffusion_ratio_map.png"),
            roi_name
        )
        convection_fraction_map, convection_fraction_vs_time = compute_convection_fraction_map(
            convection_term,
            diffusion_term
        )
        plot_row_averaged_map(
            times_plot,
            depth_mm,
            convection_fraction_map,
            f"{roi_display_name}: Convection Fraction Map",
            r"$|conv| / (|conv| + |diff|)$",
            roi_output_path(output_dirs, "convection_fraction_map.png")
        )
        plot_convection_fraction_vs_time(
            times_plot,
            convection_fraction_vs_time,
            roi_output_path(output_dirs, "mean_convection_fraction_vs_time.png"),
            roi_name
        )
    else:
        convection_ci_low_map = np.full_like(diffusion_term, np.nan, dtype=float)
        convection_ci_high_map = np.full_like(diffusion_term, np.nan, dtype=float)
        convection_ci_width_map = np.full_like(diffusion_term, np.nan, dtype=float)
        convection_relative_ci_percent_map = np.full_like(diffusion_term, np.nan, dtype=float)
        convection_diffusion_ratio_map = np.full_like(diffusion_term, np.nan, dtype=float)
        convection_fraction_map = np.full_like(diffusion_term, np.nan, dtype=float)
        convection_fraction_vs_time = np.full_like(times_plot, np.nan, dtype=float)

    plot_profile_fit_r2(
        times_plot,
        profile_fit_r2,
        roi_output_path(output_dirs, "profile_fit_r2_vs_time.png"),
        roi_name
    )

    plot_profile_fit_examples(
        depth_mm, profiles, fitted_profiles, times_plot,
        roi_output_path(output_dirs, "profile_fit_examples.png"),
        roi_name,
        r2_vs_time=profile_fit_r2,
        fitted_profiles_std=fitted_profiles_std,
        fitted_profiles_ci_low=fitted_profiles_ci_low,
        fitted_profiles_ci_high=fitted_profiles_ci_high
    )

    if SAVE_PIXELWISE_APPARENT_DIFFUSION_MAP:
        roi_stack = conc_stack[:, r0:r1, c0:c1]
        Dpix = compute_pixelwise_apparent_diffusion_map(roi_stack, times_sec, dx_mm, DEPTH_AXIS)
        Dpix_median = np.nanmedian(Dpix, axis=0)
        plot_pixelwise_map(
            Dpix_median,
            f"{roi_display_name}: Pixel-Wise Median Apparent Effective Diffusivity Map",
            r"Apparent effective diffusivity (mm$^2$/s)",
            roi_output_path(output_dirs, "pixelwise_apparent_effective_diffusivity_map.png")
        )

    save_curve_csv(
        roi_output_path(output_dirs, "curves.csv"),
        times_plot,
        mean_concentration=mean_conc,
        effective_diffusivity=D_vs_time,
        effective_diffusivity_std=D_std_vs_time,
        effective_diffusivity_ci_low=D_ci_low_vs_time,
        effective_diffusivity_ci_high=D_ci_high_vs_time,
        fitted_Cs=Cs_vs_time,
        fitted_Cs_std=Cs_std_vs_time,
        fitted_Cs_ci_low=Cs_ci_low_vs_time,
        fitted_Cs_ci_high=Cs_ci_high_vs_time,
        fitted_velocity=v_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_std=v_std_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_ci_low=v_ci_low_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_ci_high=v_ci_high_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        diffusion_curve=diffusion_curve,
        convection_curve=convection_curve if PUMP_ON else np.full_like(times_plot, np.nan),
        total_curve=total_curve,
        mean_diffusive_flux_magnitude=mean_flux_mag,
        mean_diffusive_flux_magnitude_fixed_roi_std=mean_flux_mag_fixed_roi_std,
        mean_diffusive_flux_magnitude_fixed_roi_ci_low=mean_flux_mag_fixed_roi_ci_low,
        mean_diffusive_flux_magnitude_fixed_roi_ci_high=mean_flux_mag_fixed_roi_ci_high,
        mean_diffusive_flux_magnitude_combined_std=mean_flux_mag_combined_std,
        mean_diffusive_flux_magnitude_combined_ci_low=mean_flux_mag_combined_ci_low,
        mean_diffusive_flux_magnitude_combined_ci_high=mean_flux_mag_combined_ci_high,
        mean_convection_magnitude=convection_curve if PUMP_ON else np.full_like(times_plot, np.nan),
        peclet_number=peclet_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        peclet_characteristic_length_mm=peclet_length_mm if PUMP_ON else np.full_like(times_plot, np.nan),
        convection_to_diffusion_ratio_mean=np.nanmean(convection_diffusion_ratio_map, axis=1) if PUMP_ON else np.full_like(times_plot, np.nan),
        mean_convection_fraction=convection_fraction_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        profile_fit_rmse=profile_fit_rmse,
        profile_fit_r2=profile_fit_r2,
        effective_diffusivity_hu_noise_std=noise_unc["D_std"],
        effective_diffusivity_roi_sensitivity_std=roi_unc["D_std"],
        effective_diffusivity_calibration_std=calibration_unc["D_std"],
        effective_diffusivity_fixed_roi_std=D_fixed_roi_std_vs_time,
        effective_diffusivity_fixed_roi_ci_low=D_fixed_roi_ci_low_vs_time,
        effective_diffusivity_fixed_roi_ci_high=D_fixed_roi_ci_high_vs_time,
        effective_diffusivity_combined_std=D_combined_std_vs_time,
        fitted_Cs_hu_noise_std=noise_unc["Cs_std"],
        fitted_Cs_roi_sensitivity_std=roi_unc["Cs_std"],
        fitted_Cs_calibration_std=calibration_unc["Cs_std"],
        fitted_Cs_fixed_roi_std=Cs_fixed_roi_std_vs_time,
        fitted_Cs_fixed_roi_ci_low=Cs_fixed_roi_ci_low_vs_time,
        fitted_Cs_fixed_roi_ci_high=Cs_fixed_roi_ci_high_vs_time,
        fitted_Cs_combined_std=Cs_combined_std_vs_time,
        fitted_velocity_hu_noise_std=noise_unc["v_std"] if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_roi_sensitivity_std=roi_unc["v_std"] if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_calibration_std=calibration_unc["v_std"] if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_fixed_roi_std=v_fixed_roi_std_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_fixed_roi_ci_low=v_fixed_roi_ci_low_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_fixed_roi_ci_high=v_fixed_roi_ci_high_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        fitted_velocity_combined_std=v_combined_std_vs_time if PUMP_ON else np.full_like(times_plot, np.nan),
        mean_concentration_hu_noise_std=noise_unc["mean_conc_std"],
        mean_concentration_roi_sensitivity_std=roi_unc["mean_conc_std"],
        mean_concentration_calibration_std=calibration_unc["mean_conc_std"],
        mean_concentration_fixed_roi_std=mean_conc_fixed_roi_std,
        mean_concentration_fixed_roi_ci_low=mean_conc_fixed_roi_ci_low,
        mean_concentration_fixed_roi_ci_high=mean_conc_fixed_roi_ci_high,
        mean_concentration_combined_std=mean_conc_combined_std,
        regularized_effective_diffusivity=D_vs_time_reg_plot if ENABLE_TEMPORALLY_REGULARIZED_FIT else np.full_like(times_plot, np.nan),
        regularized_fitted_Cs=Cs_vs_time_reg_plot if ENABLE_TEMPORALLY_REGULARIZED_FIT else np.full_like(times_plot, np.nan),
        regularized_mean_diffusive_flux_magnitude=mean_flux_mag_reg if ENABLE_TEMPORALLY_REGULARIZED_FIT else np.full_like(times_plot, np.nan),
        regularized_profile_fit_rmse=profile_fit_rmse_reg if ENABLE_TEMPORALLY_REGULARIZED_FIT else np.full_like(times_plot, np.nan),
        regularized_profile_fit_r2=profile_fit_r2_reg if ENABLE_TEMPORALLY_REGULARIZED_FIT else np.full_like(times_plot, np.nan),
        global_profile_fit_r2=profile_fit_r2_global if ENABLE_GLOBAL_SPATIOTEMPORAL_FIT else np.full_like(times_plot, np.nan)
    )

    pd.DataFrame(profiles).to_csv(roi_output_path(output_dirs, "measured_profiles_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles).to_csv(roi_output_path(output_dirs, "fitted_profiles_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_std).to_csv(roi_output_path(output_dirs, "fitted_profiles_std_depth_vs_time.csv"), index=False)
    pd.DataFrame(noise_unc["fitted_profiles_std"]).to_csv(roi_output_path(output_dirs, "fitted_profiles_hu_noise_std_depth_vs_time.csv"), index=False)
    pd.DataFrame(roi_unc["fitted_profiles_std"]).to_csv(roi_output_path(output_dirs, "fitted_profiles_roi_sensitivity_std_depth_vs_time.csv"), index=False)
    pd.DataFrame(calibration_unc["fitted_profiles_std"]).to_csv(roi_output_path(output_dirs, "fitted_profiles_calibration_std_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_fixed_roi_std).to_csv(roi_output_path(output_dirs, "fitted_profiles_fixed_roi_std_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_combined_std).to_csv(roi_output_path(output_dirs, "fitted_profiles_combined_std_depth_vs_time.csv"), index=False)
    if SAVE_DERIVED_EFFECTIVE_DIFFUSION_MAP:
        pd.DataFrame(derived_D_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_map.csv"), index=False)
        pd.DataFrame(derived_D_model_fit_std_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_model_fit_std_map.csv"), index=False)
        pd.DataFrame(derived_D_fixed_roi_std_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_fixed_roi_std_map.csv"), index=False)
        pd.DataFrame(derived_D_combined_std_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_combined_std_map.csv"), index=False)
        pd.DataFrame(derived_D_model_fit_ci_width_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_model_fit_ci_width_map.csv"), index=False)
        pd.DataFrame(derived_D_fixed_roi_ci_width_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_fixed_roi_ci_width_map.csv"), index=False)
        pd.DataFrame(derived_D_combined_ci_width_map).to_csv(roi_output_path(output_dirs, "derived_effective_diffusivity_combined_ci_width_map.csv"), index=False)
    pd.DataFrame(fitted_profiles_fixed_roi_ci_low).to_csv(roi_output_path(output_dirs, "fitted_profiles_fixed_roi_ci_low_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_fixed_roi_ci_high).to_csv(roi_output_path(output_dirs, "fitted_profiles_fixed_roi_ci_high_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_combined_ci_low).to_csv(roi_output_path(output_dirs, "fitted_profiles_combined_ci_low_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_combined_ci_high).to_csv(roi_output_path(output_dirs, "fitted_profiles_combined_ci_high_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_ci_low).to_csv(roi_output_path(output_dirs, "fitted_profiles_ci_low_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_ci_high).to_csv(roi_output_path(output_dirs, "fitted_profiles_ci_high_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_ci_width).to_csv(roi_output_path(output_dirs, "fitted_profiles_ci_width_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_fixed_roi_ci_width).to_csv(roi_output_path(output_dirs, "fitted_profiles_fixed_roi_ci_width_depth_vs_time.csv"), index=False)
    pd.DataFrame(fitted_profiles_relative_ci_percent_map).to_csv(roi_output_path(output_dirs, "fitted_profiles_relative_ci_percent_map.csv"), index=False)
    pd.DataFrame(fitted_profiles_fixed_roi_relative_percent_map).to_csv(roi_output_path(output_dirs, "fitted_profiles_fixed_roi_relative_percent_map.csv"), index=False)
    pd.DataFrame(fitted_profiles_combined_relative_percent_map).to_csv(roi_output_path(output_dirs, "fitted_profiles_combined_relative_percent_map.csv"), index=False)
    regional_uncertainty_summary = build_regional_profile_uncertainty_summary(depth_mm, fitted_profiles, fitted_profiles_ci_low, fitted_profiles_ci_high, fitted_profiles_combined_std)
    regional_uncertainty_summary_fixed_roi = build_regional_profile_uncertainty_summary(depth_mm, fitted_profiles, fitted_profiles_fixed_roi_ci_low, fitted_profiles_fixed_roi_ci_high, fitted_profiles_fixed_roi_std)
    regional_uncertainty_summary_fixed_roi.to_csv(roi_output_path(output_dirs, "fitted_profile_regional_uncertainty_summary_fixed_roi.csv"), index=False)
    regional_uncertainty_summary.to_csv(roi_output_path(output_dirs, "fitted_profile_regional_uncertainty_summary.csv"), index=False)
    pd.DataFrame(residual_profiles).to_csv(roi_output_path(output_dirs, "profile_fit_residuals_depth_vs_time.csv"), index=False)
    pd.DataFrame(diffusion_term).to_csv(roi_output_path(output_dirs, "diffusion_term_map.csv"), index=False)
    pd.DataFrame(diffusive_flux).to_csv(roi_output_path(output_dirs, "diffusive_flux_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_mag).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_model_fit_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_model_fit_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_hu_noise_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_hu_noise_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_roi_sensitivity_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_roi_sensitivity_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_calibration_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_calibration_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_fixed_roi_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_fixed_roi_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_combined_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_combined_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_fixed_roi_ci_low_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_fixed_roi_ci_low_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_fixed_roi_ci_high_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_fixed_roi_ci_high_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_combined_ci_low_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_combined_ci_low_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_combined_ci_high_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_combined_ci_high_map.csv"), index=False)

    pd.DataFrame(diffusive_flux_magnitude_model_fit_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_model_fit_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_hu_noise_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_hu_noise_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_roi_sensitivity_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_roi_sensitivity_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_calibration_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_calibration_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_fixed_roi_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_combined_std_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_std_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_fixed_roi_ci_low_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_ci_low_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_fixed_roi_ci_high_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_ci_high_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_combined_ci_low_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_ci_low_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_combined_ci_high_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_ci_high_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_fixed_roi_ci_width_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_ci_width_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_combined_ci_width_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_ci_width_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_fixed_roi_relative_percent_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_fixed_roi_relative_percent_map.csv"), index=False)
    pd.DataFrame(diffusive_flux_magnitude_combined_relative_percent_map).to_csv(roi_output_path(output_dirs, "diffusive_flux_magnitude_combined_relative_percent_map.csv"), index=False)

    if ENABLE_TEMPORALLY_REGULARIZED_FIT:
        pd.DataFrame(fitted_profiles_reg).to_csv(roi_output_path(output_dirs, "temporally_regularized_fitted_profiles_depth_vs_time.csv"), index=False)
        pd.DataFrame(residual_profiles_reg).to_csv(roi_output_path(output_dirs, "temporally_regularized_profile_fit_residuals_depth_vs_time.csv"), index=False)
        pd.DataFrame(diffusion_term_reg).to_csv(roi_output_path(output_dirs, "temporally_regularized_diffusion_term_map.csv"), index=False)
        pd.DataFrame(diffusive_flux_reg).to_csv(roi_output_path(output_dirs, "temporally_regularized_diffusive_flux_map.csv"), index=False)
        pd.DataFrame(diffusive_flux_mag_reg).to_csv(roi_output_path(output_dirs, "temporally_regularized_diffusive_flux_magnitude_map.csv"), index=False)
        pd.DataFrame({
            'time_plot': times_plot,
            'time_seconds': times_sec,
            'regularized_effective_diffusivity_mm2_s': D_vs_time_reg,
            'regularized_effective_diffusivity_plot_mm2_s': D_vs_time_reg_plot,
            'regularized_fitted_Cs': Cs_vs_time_reg,
            'regularized_fitted_Cs_plot': Cs_vs_time_reg_plot,
            'regularized_fitted_velocity_mm_s': v_vs_time_reg,
            'regularized_profile_fit_rmse': profile_fit_rmse_reg,
            'regularized_profile_fit_r2': profile_fit_r2_reg,
            'regularized_mean_diffusive_flux_magnitude': mean_flux_mag_reg,
            'regularized_diffusion_curve': diffusion_curve_reg,
            'regularized_convection_curve': convection_curve_reg,
            'regularized_total_curve': total_curve_reg
        }).to_csv(roi_output_path(output_dirs, "temporally_regularized_fit_summary.csv"), index=False)

    if ENABLE_GLOBAL_SPATIOTEMPORAL_FIT:
        pd.DataFrame(fitted_profiles_global).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_fitted_profiles_depth_vs_time.csv"), index=False)
        pd.DataFrame(residual_profiles_global).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_profile_fit_residuals_depth_vs_time.csv"), index=False)
        pd.DataFrame(diffusion_term_global).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_diffusion_term_map.csv"), index=False)
        pd.DataFrame(diffusive_flux_global).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_diffusive_flux_map.csv"), index=False)
        pd.DataFrame(diffusive_flux_mag_global).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_diffusive_flux_magnitude_map.csv"), index=False)
        pd.DataFrame({
            'time_plot': times_plot,
            'time_seconds': times_sec,
            'global_smooth_Cs': Cs_global_vs_time,
            'global_profile_fit_rmse': profile_fit_rmse_global,
            'global_profile_fit_r2': profile_fit_r2_global,
            'global_effective_diffusivity_mm2_s': np.full_like(times_plot, D_global_fit, dtype=float),
            'global_fitted_velocity_mm_s': np.full_like(times_plot, v_global_fit, dtype=float),
            'global_mean_diffusive_flux_magnitude': mean_flux_mag_global,
            'global_diffusion_curve': diffusion_curve_global,
            'global_convection_curve': convection_curve_global,
            'global_total_curve': total_curve_global
        }).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_fit_summary.csv"), index=False)
        pd.DataFrame({
            'control_time_seconds': cs_control_times_sec,
            'control_time_plot': normalize_time(cs_control_times_sec, TIME_UNIT),
            'Cs_control_value': cs_control_values
        }).to_csv(roi_output_path(output_dirs, "global_spatiotemporal_Cs_control_points.csv"), index=False)

    if SAVE_EFFECTIVE_DIFFUSION_MAP:
        pd.DataFrame(D_local_map).to_csv(roi_output_path(output_dirs, "local_effective_diffusivity_map.csv"), index=False)
        pd.DataFrame(Cs_local_map).to_csv(roi_output_path(output_dirs, "local_effective_Cs_map.csv"), index=False)
        pd.DataFrame(pred_local_map).to_csv(roi_output_path(output_dirs, "local_effective_predicted_profile_map.csv"), index=False)
        pd.DataFrame(local_fit_rmse_map).to_csv(roi_output_path(output_dirs, "local_effective_fit_rmse_map.csv"), index=False)
        if PUMP_ON:
            pd.DataFrame(v_local_map).to_csv(roi_output_path(output_dirs, "local_effective_velocity_map.csv"), index=False)

    if PUMP_ON:
        pd.DataFrame(convection_term).to_csv(roi_output_path(output_dirs, "convection_term_map.csv"), index=False)
        pd.DataFrame(convection_ci_low_map).to_csv(roi_output_path(output_dirs, "convection_term_ci_low_map.csv"), index=False)
        pd.DataFrame(convection_ci_high_map).to_csv(roi_output_path(output_dirs, "convection_term_ci_high_map.csv"), index=False)
        pd.DataFrame(convection_ci_width_map).to_csv(roi_output_path(output_dirs, "convection_term_ci_width_map.csv"), index=False)
        pd.DataFrame(convection_relative_ci_percent_map).to_csv(roi_output_path(output_dirs, "convection_term_relative_ci_percent_map.csv"), index=False)
        pd.DataFrame(convection_diffusion_ratio_map).to_csv(roi_output_path(output_dirs, "convection_to_diffusion_ratio_map.csv"), index=False)
        pd.DataFrame(convection_fraction_map).to_csv(roi_output_path(output_dirs, "convection_fraction_map.csv"), index=False)

    fit_table = pd.DataFrame({
        "time_plot": times_plot,
        "time_seconds": times_sec,
        "effective_diffusivity_mm2_s": D_vs_time,
        "effective_diffusivity_std_mm2_s": D_std_vs_time,
        "effective_diffusivity_ci_low_mm2_s": D_ci_low_vs_time,
        "effective_diffusivity_ci_high_mm2_s": D_ci_high_vs_time,
        "effective_diffusivity_hu_noise_std_mm2_s": noise_unc["D_std"],
        "effective_diffusivity_roi_sensitivity_std_mm2_s": roi_unc["D_std"],
        "effective_diffusivity_calibration_std_mm2_s": calibration_unc["D_std"],
        "effective_diffusivity_fixed_roi_std_mm2_s": D_fixed_roi_std_vs_time,
        "effective_diffusivity_fixed_roi_ci_low_mm2_s": D_fixed_roi_ci_low_vs_time,
        "effective_diffusivity_fixed_roi_ci_high_mm2_s": D_fixed_roi_ci_high_vs_time,
        "effective_diffusivity_combined_std_mm2_s": D_combined_std_vs_time,
        "effective_diffusivity_combined_ci_low_mm2_s": D_combined_ci_low_vs_time,
        "effective_diffusivity_combined_ci_high_mm2_s": D_combined_ci_high_vs_time,
        "fitted_Cs": Cs_vs_time,
        "fitted_Cs_std": Cs_std_vs_time,
        "fitted_Cs_hu_noise_std": noise_unc["Cs_std"],
        "fitted_Cs_roi_sensitivity_std": roi_unc["Cs_std"],
        "fitted_Cs_calibration_std": calibration_unc["Cs_std"],
        "fitted_Cs_fixed_roi_std": Cs_fixed_roi_std_vs_time,
        "fitted_Cs_fixed_roi_ci_low": Cs_fixed_roi_ci_low_vs_time,
        "fitted_Cs_fixed_roi_ci_high": Cs_fixed_roi_ci_high_vs_time,
        "fitted_Cs_combined_std": Cs_combined_std_vs_time,
        "fitted_Cs_combined_ci_low": Cs_combined_ci_low_vs_time,
        "fitted_Cs_combined_ci_high": Cs_combined_ci_high_vs_time,
        "fitted_Cs_ci_low": Cs_ci_low_vs_time,
        "fitted_Cs_ci_high": Cs_ci_high_vs_time,
        "fitted_velocity_mm_s": v_vs_time,
        "fitted_velocity_std_mm_s": v_std_vs_time,
        "fitted_velocity_hu_noise_std_mm_s": noise_unc["v_std"],
        "fitted_velocity_roi_sensitivity_std_mm_s": roi_unc["v_std"],
        "fitted_velocity_calibration_std_mm_s": calibration_unc["v_std"],
        "fitted_velocity_ci_low_mm_s": v_ci_low_vs_time,
        "fitted_velocity_ci_high_mm_s": v_ci_high_vs_time,
        "fitted_velocity_fixed_roi_std_mm_s": v_fixed_roi_std_vs_time,
        "fitted_velocity_fixed_roi_ci_low_mm_s": v_fixed_roi_ci_low_vs_time,
        "fitted_velocity_fixed_roi_ci_high_mm_s": v_fixed_roi_ci_high_vs_time,
        "fitted_velocity_combined_std_mm_s": v_combined_std_vs_time,
        "fitted_velocity_combined_ci_low_mm_s": v_combined_ci_low_vs_time,
        "fitted_velocity_combined_ci_high_mm_s": v_combined_ci_high_vs_time,
        "peclet_number": peclet_vs_time,
        "peclet_characteristic_length_mm": peclet_length_mm,
        "mean_convection_fraction": convection_fraction_vs_time,
        "profile_fit_rmse": profile_fit_rmse,
        "profile_fit_r2": profile_fit_r2,
        "mean_diffusive_flux_magnitude_fixed_roi_std": mean_flux_mag_fixed_roi_std,
        "mean_diffusive_flux_magnitude_fixed_roi_ci_low": mean_flux_mag_fixed_roi_ci_low,
        "mean_diffusive_flux_magnitude_fixed_roi_ci_high": mean_flux_mag_fixed_roi_ci_high,
        "mean_diffusive_flux_magnitude_combined_std": mean_flux_mag_combined_std,
        "mean_diffusive_flux_magnitude_combined_ci_low": mean_flux_mag_combined_ci_low,
        "mean_diffusive_flux_magnitude_combined_ci_high": mean_flux_mag_combined_ci_high,
        "regularized_effective_diffusivity_mm2_s": D_vs_time_reg,
        "regularized_effective_diffusivity_plot_mm2_s": D_vs_time_reg_plot,
        "regularized_fitted_Cs": Cs_vs_time_reg,
        "regularized_fitted_Cs_plot": Cs_vs_time_reg_plot,
        "regularized_fitted_velocity_mm_s": v_vs_time_reg,
        "regularized_profile_fit_rmse": profile_fit_rmse_reg,
        "regularized_profile_fit_r2": profile_fit_r2_reg,
        "global_effective_diffusivity_mm2_s": np.full_like(times_sec, D_global_fit, dtype=float),
        "global_smooth_Cs": Cs_global_vs_time,
        "global_fitted_velocity_mm_s": np.full_like(times_sec, v_global_fit, dtype=float),
        "global_profile_fit_rmse": profile_fit_rmse_global,
        "global_profile_fit_r2": profile_fit_r2_global,
        "mean_concentration_hu_noise_std": noise_unc["mean_conc_std"],
        "mean_concentration_roi_sensitivity_std": roi_unc["mean_conc_std"],
        "mean_concentration_calibration_std": calibration_unc["mean_conc_std"],
        "mean_concentration_fixed_roi_std": mean_conc_fixed_roi_std,
        "mean_concentration_fixed_roi_ci_low": mean_conc_fixed_roi_ci_low,
        "mean_concentration_fixed_roi_ci_high": mean_conc_fixed_roi_ci_high,
        "mean_concentration_combined_std": mean_conc_combined_std,
        "mean_concentration_combined_ci_low": mean_conc_combined_ci_low,
        "mean_concentration_combined_ci_high": mean_conc_combined_ci_high,
        "estimated_hu_noise_std": np.full_like(times_sec, estimated_hu_noise_std, dtype=float),
        "estimated_concentration_noise_std": np.full_like(times_sec, estimated_conc_noise_std, dtype=float)
    })
    fit_table.to_csv(roi_output_path(output_dirs, "fit_parameters_vs_time.csv"), index=False)
    with open(roi_output_path(output_dirs, "uncertainty_metadata.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Estimated HU noise std from deep ROI region: {estimated_hu_noise_std:.6g} HU\n")
        fh.write(f"Estimated concentration noise std from deep ROI region: {estimated_conc_noise_std:.6g} {CONCENTRATION_UNITS}\n")
        fh.write(f"HU noise Monte Carlo samples: {HU_NOISE_MONTE_CARLO_SAMPLES}\n")
        fh.write(f"ROI sensitivity shifts: {ROI_SENSITIVITY_SHIFTS}\n")
        fh.write(f"ROI sensitivity variants used: {roi_unc.get('used_rois', [roi])}\n")
        fh.write(f"HU_PER_CONC_STD input: {HU_PER_CONC_STD}\n")
        fh.write(f"HU_OFFSET_STD input: {HU_OFFSET_STD}\n")
        fh.write("Calibration uncertainty estimated by ±1 SD perturbation refits of HU_PER_CONC and HU_OFFSET.\n")
        fh.write("Fixed-ROI uncertainty outputs are now saved for D(t), C_s(t), v(t), mean concentration, fitted profiles, derived effective diffusivity maps, and regional fitted-profile summaries. Fixed-ROI means model fit + HU noise + calibration, excluding ROI sensitivity.\n")
        fh.write("Diffusive flux uncertainty maps use fixed-ROI propagation from D(t) and fitted-profile uncertainty components (model fit, HU noise, calibration), with combined maps also including ROI sensitivity.\n")

    return {
        "roi": roi,
        "mean_conc": mean_conc,
        "profiles": profiles,
        "fitted_profiles": fitted_profiles,
        "diffusion_curve": diffusion_curve,
        "convection_curve": convection_curve,
        "total_curve": total_curve,
        "diffusive_flux": diffusive_flux,
        "diffusive_flux_mag": diffusive_flux_mag,
        "diffusive_flux_magnitude_fixed_roi_std_map": diffusive_flux_magnitude_fixed_roi_std_map,
        "diffusive_flux_magnitude_combined_std_map": diffusive_flux_magnitude_combined_std_map,
        "diffusive_flux_magnitude_fixed_roi_ci_width_map": diffusive_flux_magnitude_fixed_roi_ci_width_map,
        "diffusive_flux_magnitude_combined_ci_width_map": diffusive_flux_magnitude_combined_ci_width_map,
        "diffusive_flux_magnitude_fixed_roi_relative_percent_map": diffusive_flux_magnitude_fixed_roi_relative_percent_map,
        "diffusive_flux_magnitude_combined_relative_percent_map": diffusive_flux_magnitude_combined_relative_percent_map,
        "mean_flux_mag": mean_flux_mag,
        "mean_flux_mag_fixed_roi_std": mean_flux_mag_fixed_roi_std,
        "mean_flux_mag_fixed_roi_ci_low": mean_flux_mag_fixed_roi_ci_low,
        "mean_flux_mag_fixed_roi_ci_high": mean_flux_mag_fixed_roi_ci_high,
        "mean_flux_mag_combined_std": mean_flux_mag_combined_std,
        "mean_flux_mag_combined_ci_low": mean_flux_mag_combined_ci_low,
        "mean_flux_mag_combined_ci_high": mean_flux_mag_combined_ci_high,
        "mean_convection_mag": convection_curve,
        "fitted_profiles_ci_width": fitted_profiles_ci_width,
        "fitted_profiles_relative_ci_percent_map": fitted_profiles_relative_ci_percent_map,
        "fitted_profiles_combined_relative_percent_map": fitted_profiles_combined_relative_percent_map,
        "convection_ci_low_map": convection_ci_low_map,
        "convection_ci_high_map": convection_ci_high_map,
        "convection_ci_width_map": convection_ci_width_map,
        "convection_relative_ci_percent_map": convection_relative_ci_percent_map,
        "convection_diffusion_ratio_map": convection_diffusion_ratio_map,
        "convection_fraction_map": convection_fraction_map,
        "convection_fraction_vs_time": convection_fraction_vs_time,
        "peclet_vs_time": peclet_vs_time,
        "peclet_length_mm": peclet_length_mm,
        "D_local_map": D_local_map,
        "Cs_local_map": Cs_local_map,
        "v_local_map": v_local_map,
        "pred_local_map": pred_local_map,
        "local_fit_rmse_map": local_fit_rmse_map,
        "profile_fit_rmse": profile_fit_rmse,
        "profile_fit_r2": profile_fit_r2,
        "residual_profiles": residual_profiles,
        "D_vs_time": D_vs_time,
        "D_std_vs_time": D_std_vs_time,
        "D_ci_low_vs_time": D_ci_low_vs_time,
        "D_ci_high_vs_time": D_ci_high_vs_time,
        "D_fixed_roi_ci_low_vs_time": D_fixed_roi_ci_low_vs_time,
        "D_fixed_roi_ci_high_vs_time": D_fixed_roi_ci_high_vs_time,
        "D_combined_ci_low_vs_time": D_combined_ci_low_vs_time,
        "D_combined_ci_high_vs_time": D_combined_ci_high_vs_time,
        "D_noise_std_vs_time": noise_unc["D_std"],
        "D_roi_sensitivity_std_vs_time": roi_unc["D_std"],
        "D_calibration_std_vs_time": calibration_unc["D_std"],
        "D_fixed_roi_std_vs_time": D_fixed_roi_std_vs_time,
        "D_combined_std_vs_time": D_combined_std_vs_time,
        "Cs_vs_time": Cs_vs_time,
        "Cs_std_vs_time": Cs_std_vs_time,
        "Cs_ci_low_vs_time": Cs_ci_low_vs_time,
        "Cs_ci_high_vs_time": Cs_ci_high_vs_time,
        "Cs_fixed_roi_ci_low_vs_time": Cs_fixed_roi_ci_low_vs_time,
        "Cs_fixed_roi_ci_high_vs_time": Cs_fixed_roi_ci_high_vs_time,
        "Cs_combined_ci_low_vs_time": Cs_combined_ci_low_vs_time,
        "Cs_combined_ci_high_vs_time": Cs_combined_ci_high_vs_time,
        "Cs_noise_std_vs_time": noise_unc["Cs_std"],
        "Cs_roi_sensitivity_std_vs_time": roi_unc["Cs_std"],
        "Cs_calibration_std_vs_time": calibration_unc["Cs_std"],
        "Cs_fixed_roi_std_vs_time": Cs_fixed_roi_std_vs_time,
        "Cs_combined_std_vs_time": Cs_combined_std_vs_time,
        "v_vs_time": v_vs_time,
        "v_std_vs_time": v_std_vs_time,
        "v_ci_low_vs_time": v_ci_low_vs_time,
        "v_ci_high_vs_time": v_ci_high_vs_time,
        "v_fixed_roi_ci_low_vs_time": v_fixed_roi_ci_low_vs_time,
        "v_fixed_roi_ci_high_vs_time": v_fixed_roi_ci_high_vs_time,
        "v_combined_ci_low_vs_time": v_combined_ci_low_vs_time,
        "v_combined_ci_high_vs_time": v_combined_ci_high_vs_time,
        "v_noise_std_vs_time": noise_unc["v_std"],
        "v_roi_sensitivity_std_vs_time": roi_unc["v_std"],
        "v_calibration_std_vs_time": calibration_unc["v_std"],
        "v_fixed_roi_std_vs_time": v_fixed_roi_std_vs_time,
        "v_combined_std_vs_time": v_combined_std_vs_time,
        "fitted_profiles_std": fitted_profiles_std,
        "fitted_profiles_ci_low": fitted_profiles_ci_low,
        "fitted_profiles_ci_high": fitted_profiles_ci_high,
        "fitted_profiles_fixed_roi_std": fitted_profiles_fixed_roi_std,
        "fitted_profiles_fixed_roi_ci_low": fitted_profiles_fixed_roi_ci_low,
        "fitted_profiles_fixed_roi_ci_high": fitted_profiles_fixed_roi_ci_high,
        "fitted_profiles_reg": fitted_profiles_reg,
        "profile_fit_rmse_reg": profile_fit_rmse_reg,
        "D_vs_time_reg": D_vs_time_reg,
        "Cs_vs_time_reg": Cs_vs_time_reg,
        "v_vs_time_reg": v_vs_time_reg,
        "fitted_profiles_global": fitted_profiles_global,
        "profile_fit_rmse_global": profile_fit_rmse_global,
        "D_global_fit": D_global_fit,
        "Cs_global_vs_time": Cs_global_vs_time,
        "v_global_fit": v_global_fit,
        "mean_conc_hu_noise_std": noise_unc["mean_conc_std"],
        "mean_conc_roi_sensitivity_std": roi_unc["mean_conc_std"],
        "mean_conc_calibration_std": calibration_unc["mean_conc_std"],
        "mean_conc_fixed_roi_std": mean_conc_fixed_roi_std,
        "mean_conc_fixed_roi_ci_low": mean_conc_fixed_roi_ci_low,
        "mean_conc_fixed_roi_ci_high": mean_conc_fixed_roi_ci_high,
        "mean_conc_combined_std": mean_conc_combined_std,
        "mean_conc_combined_ci_low": mean_conc_combined_ci_low,
        "mean_conc_combined_ci_high": mean_conc_combined_ci_high,
        "estimated_hu_noise_std": estimated_hu_noise_std,
        "estimated_concentration_noise_std": estimated_conc_noise_std,
        "regional_uncertainty_summary": regional_uncertainty_summary.to_dict(orient="records"),
        "regional_uncertainty_summary_fixed_roi": regional_uncertainty_summary_fixed_roi.to_dict(orient="records"),
        "derived_D_map": derived_D_map,
        "derived_D_model_fit_std_map": derived_D_model_fit_std_map,
        "derived_D_fixed_roi_std_map": derived_D_fixed_roi_std_map,
        "derived_D_combined_std_map": derived_D_combined_std_map,
        "derived_D_model_fit_ci_width_map": derived_D_model_fit_ci_width_map,
        "derived_D_fixed_roi_ci_width_map": derived_D_fixed_roi_ci_width_map,
        "derived_D_combined_ci_width_map": derived_D_combined_ci_width_map,
        "D_global": nan_safe_median(D_vs_time),
        "Cs_global": nan_safe_median(Cs_vs_time),
        "v_global": nan_safe_median(v_vs_time)
    }


# ============================================================
# RUN METADATA
# ============================================================

def _make_json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    return str(value)


def collect_user_settings_metadata() -> Dict[str, object]:
    settings = {}
    for name, value in globals().items():
        if not name.isupper():
            continue
        if name.startswith('__'):
            continue
        settings[name] = _make_json_safe(value)
    return settings


def save_run_metadata(output_folder: str, metadata: Dict[str, object]) -> str:
    metadata_path = os.path.join(output_folder, 'run_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(_make_json_safe(metadata), f, indent=2, sort_keys=True)
    return metadata_path


# ============================================================
# MAIN
# ============================================================

def main():
    ensure_dir(OUTPUT_FOLDER)

    run_metadata = {
        "script_name": os.path.basename(__file__) if "__file__" in globals() else "Analysis_Main_with_noise_and_roi_and_calibrated_uncertainty.py",
        "script_path": os.path.abspath(__file__) if "__file__" in globals() else None,
        "run_timestamp_local": datetime.now().isoformat(timespec="seconds"),
        "output_folder_absolute": os.path.abspath(OUTPUT_FOLDER),
        "user_settings": collect_user_settings_metadata(),
    }

    print("Loading DICOM files...")
    frames = load_dicom_frames(DICOM_FOLDER)
    groups = group_frames_by_slice(frames)

    slice_locations = sorted(groups.keys())

    print("\nAvailable slice locations:")
    for i, sl in enumerate(slice_locations):
        print(f"[{i}] SliceLocation = {sl:.4f} | time points = {len(groups[sl])}")

    user_input = input("\nEnter slice index OR exact SliceLocation value: ").strip()

    selected_slice = None
    try:
        idx = int(user_input)
        selected_slice = slice_locations[idx]
    except ValueError:
        selected_slice = float(user_input)
        nearest = np.argmin(np.abs(np.array(slice_locations) - selected_slice))
        selected_slice = slice_locations[nearest]

    selected_frames = groups[selected_slice]

    if SKIP_INITIAL_FRAMES > 0:
        if len(selected_frames) <= SKIP_INITIAL_FRAMES:
            raise RuntimeError(
                f"Not enough frames after skipping the first {SKIP_INITIAL_FRAMES} frame(s)."
            )
        selected_frames = selected_frames[SKIP_INITIAL_FRAMES:]

    if len(selected_frames) < 2:
        raise RuntimeError("Need at least 2 time points for transport analysis.")

    print(f"\nSelected SliceLocation: {selected_slice:.4f}")
    print(f"Number of time points after skipping initial frames: {len(selected_frames)}")
    print(f"Initial frames skipped: {SKIP_INITIAL_FRAMES}")

    run_metadata["selected_slice_location"] = float(selected_slice)
    run_metadata["num_timepoints"] = int(len(selected_frames))
    run_metadata["skipped_initial_frames"] = int(SKIP_INITIAL_FRAMES)

    times_sec = np.array([fr.time_seconds for fr in selected_frames], dtype=float)
    times_sec = times_sec - times_sec[0]   # make time relative to first frame
    times_plot = normalize_time(times_sec, TIME_UNIT)

    hu_stack = np.stack([fr.image_hu for fr in selected_frames], axis=0)
    row_spacing_mm = selected_frames[0].row_spacing_mm
    col_spacing_mm = selected_frames[0].col_spacing_mm
    dx_mm = get_depth_spacing_mm(row_spacing_mm, col_spacing_mm, DEPTH_AXIS)

    run_metadata["frame_geometry"] = {
        "hu_stack_shape": list(hu_stack.shape),
        "row_spacing_mm": float(row_spacing_mm),
        "col_spacing_mm": float(col_spacing_mm),
        "depth_spacing_mm": float(dx_mm),
        "time_seconds_relative": times_sec.tolist(),
        "time_plot_values": times_plot.tolist(),
    }

    reference_image = hu_stack[0]

    if ROI_SELECTION_MODE == "interactive":
        named_rois = collect_named_rois(reference_image)
    elif ROI_SELECTION_MODE == "manual_list":
        named_rois = get_named_rois_from_settings(reference_image)
    elif ROI_SELECTION_MODE == "manual_prompt":
        named_rois = collect_named_rois_manual_input(reference_image)
    else:
        raise ValueError("ROI_SELECTION_MODE must be 'interactive', 'manual_list', or 'manual_prompt'")

    if not named_rois:
        raise RuntimeError("No ROIs were provided.")

    saved_roi_json_path = save_selected_rois_json(OUTPUT_FOLDER, named_rois)

    run_metadata["selected_rois"] = [
        {
            "roi_name": roi_name,
            "r0": int(roi[0]),
            "r1": int(roi[1]),
            "c0": int(roi[2]),
            "c1": int(roi[3]),
        }
        for roi_name, roi in named_rois
    ]

    show_all_rois_overlay(
        reference_image,
        named_rois,
        os.path.join(OUTPUT_FOLDER, "all_rois_overlay.png")
    )

    # Absolute-concentration fitting branch: no baseline subtraction (disabled in this absolute-concentration version)
    baseline_image = None

    conc_stack = hu_to_concentration(hu_stack, baseline_image, HU_PER_CONC, HU_OFFSET)
    conc_stack = smooth_stack(conc_stack)

    roi_results = {}
    summary_rows = []
    post_5min_mask = np.asarray(times_sec) >= 300.0
    pre_5min_mask = np.asarray(times_sec) < 300.0

    def _subset_by_mask(values, mask):
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 0:
            return np.array([arr], dtype=float)
        if arr.shape[0] != mask.shape[0]:
            return arr
        return arr[mask]

    for roi_name, roi in named_rois:
        print(f"\nAnalyzing {roi_name}: {roi}")
        res = analyze_single_roi(
            roi_name=roi_name,
            roi=roi,
            conc_stack=conc_stack,
            hu_stack=hu_stack,
            times_sec=times_sec,
            times_plot=times_plot,
            dx_mm=dx_mm,
            output_root=OUTPUT_FOLDER
        )
        roi_results[roi_name] = res

        D_pre_5min = _subset_by_mask(res["D_vs_time"], pre_5min_mask)
        D_post_5min = _subset_by_mask(res["D_vs_time"], post_5min_mask)
        D_fit_std_post_5min = _subset_by_mask(res["D_std_vs_time"], post_5min_mask)
        D_fixed_roi_std_post_5min = _subset_by_mask(res["D_fixed_roi_std_vs_time"], post_5min_mask)
        D_combined_std_post_5min = _subset_by_mask(res["D_combined_std_vs_time"], post_5min_mask)
        Cs_pre_5min = _subset_by_mask(res["Cs_vs_time"], pre_5min_mask)
        Cs_post_5min = _subset_by_mask(res["Cs_vs_time"], post_5min_mask)
        Cs_fit_std_post_5min = _subset_by_mask(res["Cs_std_vs_time"], post_5min_mask)
        Cs_fixed_roi_std_post_5min = _subset_by_mask(res["Cs_fixed_roi_std_vs_time"], post_5min_mask)
        Cs_combined_std_post_5min = _subset_by_mask(res["Cs_combined_std_vs_time"], post_5min_mask)
        v_pre_5min = _subset_by_mask(res["v_vs_time"], pre_5min_mask)
        v_post_5min = _subset_by_mask(res["v_vs_time"], post_5min_mask)
        v_fit_std_post_5min = _subset_by_mask(res["v_std_vs_time"], post_5min_mask)
        v_fixed_roi_std_post_5min = _subset_by_mask(res["v_fixed_roi_std_vs_time"], post_5min_mask)
        v_combined_std_post_5min = _subset_by_mask(res["v_combined_std_vs_time"], post_5min_mask)
        mean_conc_pre_5min = _subset_by_mask(res["mean_conc"], pre_5min_mask)
        mean_conc_post_5min = _subset_by_mask(res["mean_conc"], post_5min_mask)
        mean_conc_fixed_roi_std_post_5min = _subset_by_mask(res["mean_conc_fixed_roi_std"], post_5min_mask)
        mean_conc_combined_std_post_5min = _subset_by_mask(res["mean_conc_combined_std"], post_5min_mask)
        diffusion_curve_post_5min = _subset_by_mask(res["diffusion_curve"], post_5min_mask)
        convection_curve_post_5min = _subset_by_mask(res["convection_curve"], post_5min_mask)
        total_curve_post_5min = _subset_by_mask(res["total_curve"], post_5min_mask)
        mean_flux_mag_post_5min = _subset_by_mask(res["mean_flux_mag"], post_5min_mask)
        peclet_post_5min = _subset_by_mask(res["peclet_vs_time"], post_5min_mask)
        convection_fraction_post_5min = _subset_by_mask(res["convection_fraction_vs_time"], post_5min_mask)

        regional_uncertainty_summary_df = pd.DataFrame(res.get("regional_uncertainty_summary", []))
        regional_uncertainty_summary_fixed_roi_df = pd.DataFrame(res.get("regional_uncertainty_summary_fixed_roi", []))
        regional_summary_flat = flatten_regional_uncertainty_summary(regional_uncertainty_summary_df)
        regional_summary_fixed_roi_flat = flatten_regional_uncertainty_summary(regional_uncertainty_summary_fixed_roi_df)

        summary_row = {
            "roi_name": roi_name,
            "r0": roi[0],
            "r1": roi[1],
            "c0": roi[2],
            "c1": roi[3],
            "global_fit_D_mm2_s": res["D_global_fit"],
            "median_per_timepoint_D_mm2_s": res["D_global"],
            "mean_per_timepoint_D_mm2_s": nan_safe_mean(res["D_vs_time"]),
            "std_per_timepoint_D_mm2_s": nan_safe_std(res["D_vs_time"]),
            "iqr_pre_5min_D_mm2_s": nan_safe_iqr(D_pre_5min),
            "mad_pre_5min_D_mm2_s": nan_safe_mad(D_pre_5min),
            "median_post_5min_D_mm2_s": nan_safe_median(D_post_5min),
            "mean_post_5min_D_mm2_s": nan_safe_mean(D_post_5min),
            "std_post_5min_D_mm2_s": nan_safe_std(D_post_5min),
            "iqr_post_5min_D_mm2_s": nan_safe_iqr(D_post_5min),
            "mad_post_5min_D_mm2_s": nan_safe_mad(D_post_5min),
            "mean_D_fit_std_mm2_s": nan_safe_mean(res["D_std_vs_time"]),
            "mean_D_fixed_roi_std_mm2_s": nan_safe_mean(res["D_fixed_roi_std_vs_time"]),
            "mean_D_combined_std_mm2_s": nan_safe_mean(res["D_combined_std_vs_time"]),
            "mean_post_5min_D_fit_std_mm2_s": nan_safe_mean(D_fit_std_post_5min),
            "mean_post_5min_D_fixed_roi_std_mm2_s": nan_safe_mean(D_fixed_roi_std_post_5min),
            "mean_post_5min_D_combined_std_mm2_s": nan_safe_mean(D_combined_std_post_5min),
            "global_fit_Cs": nan_safe_mean(res["Cs_global_vs_time"]),
            "median_per_timepoint_Cs": res["Cs_global"],
            "iqr_pre_5min_Cs": nan_safe_iqr(Cs_pre_5min),
            "mad_pre_5min_Cs": nan_safe_mad(Cs_pre_5min),
            "median_post_5min_Cs": nan_safe_median(Cs_post_5min),
            "mean_post_5min_Cs": nan_safe_mean(Cs_post_5min),
            "std_post_5min_Cs": nan_safe_std(Cs_post_5min),
            "iqr_post_5min_Cs": nan_safe_iqr(Cs_post_5min),
            "mad_post_5min_Cs": nan_safe_mad(Cs_post_5min),
            "mean_Cs_fit_std": nan_safe_mean(res["Cs_std_vs_time"]),
            "mean_Cs_fixed_roi_std": nan_safe_mean(res["Cs_fixed_roi_std_vs_time"]),
            "mean_Cs_combined_std": nan_safe_mean(res["Cs_combined_std_vs_time"]),
            "mean_post_5min_Cs_fit_std": nan_safe_mean(Cs_fit_std_post_5min),
            "mean_post_5min_Cs_fixed_roi_std": nan_safe_mean(Cs_fixed_roi_std_post_5min),
            "mean_post_5min_Cs_combined_std": nan_safe_mean(Cs_combined_std_post_5min),
            "global_fit_velocity_mm_s": res["v_global_fit"],
            "median_per_timepoint_velocity_mm_s": res["v_global"],
            "std_per_timepoint_velocity_mm_s": nan_safe_std(res["v_vs_time"]),
            "iqr_pre_5min_velocity_mm_s": nan_safe_iqr(v_pre_5min),
            "mad_pre_5min_velocity_mm_s": nan_safe_mad(v_pre_5min),
            "median_post_5min_velocity_mm_s": nan_safe_median(v_post_5min),
            "mean_post_5min_velocity_mm_s": nan_safe_mean(v_post_5min),
            "std_post_5min_velocity_mm_s": nan_safe_std(v_post_5min),
            "iqr_post_5min_velocity_mm_s": nan_safe_iqr(v_post_5min),
            "mad_post_5min_velocity_mm_s": nan_safe_mad(v_post_5min),
            "mean_velocity_fit_std_mm_s": nan_safe_mean(res["v_std_vs_time"]),
            "mean_velocity_fixed_roi_std_mm_s": nan_safe_mean(res["v_fixed_roi_std_vs_time"]),
            "mean_velocity_combined_std_mm_s": nan_safe_mean(res["v_combined_std_vs_time"]),
            "mean_post_5min_velocity_fit_std_mm_s": nan_safe_mean(v_fit_std_post_5min),
            "mean_post_5min_velocity_fixed_roi_std_mm_s": nan_safe_mean(v_fixed_roi_std_post_5min),
            "mean_post_5min_velocity_combined_std_mm_s": nan_safe_mean(v_combined_std_post_5min),
            "mean_concentration_over_time": nan_safe_mean(res["mean_conc"]),
            "iqr_pre_5min_mean_concentration": nan_safe_iqr(mean_conc_pre_5min),
            "mad_pre_5min_mean_concentration": nan_safe_mad(mean_conc_pre_5min),
            "mean_concentration_post_5min": nan_safe_mean(mean_conc_post_5min),
            "mean_concentration_fixed_roi_std": nan_safe_mean(res["mean_conc_fixed_roi_std"]),
            "mean_concentration_combined_std": nan_safe_mean(res["mean_conc_combined_std"]),
            "mean_post_5min_concentration_fixed_roi_std": nan_safe_mean(mean_conc_fixed_roi_std_post_5min),
            "mean_post_5min_concentration_combined_std": nan_safe_mean(mean_conc_combined_std_post_5min),
            "iqr_post_5min_mean_concentration": nan_safe_iqr(mean_conc_post_5min),
            "mad_post_5min_mean_concentration": nan_safe_mad(mean_conc_post_5min),
            "mean_diffusion_curve": nan_safe_mean(res["diffusion_curve"]),
            "mean_diffusion_curve_post_5min": nan_safe_mean(diffusion_curve_post_5min),
            "mean_convection_curve": nan_safe_mean(res["convection_curve"]),
            "mean_convection_curve_post_5min": nan_safe_mean(convection_curve_post_5min),
            "mean_total_curve": nan_safe_mean(res["total_curve"]),
            "mean_total_curve_post_5min": nan_safe_mean(total_curve_post_5min),
            "mean_diffusive_flux_magnitude": nan_safe_mean(res["mean_flux_mag"]),
            "mean_diffusive_flux_magnitude_post_5min": nan_safe_mean(mean_flux_mag_post_5min),
            "mean_peclet_number": nan_safe_mean(res["peclet_vs_time"]),
            "mean_peclet_number_post_5min": nan_safe_mean(peclet_post_5min),
            "mean_convection_fraction": nan_safe_mean(res["convection_fraction_vs_time"]),
            "mean_convection_fraction_post_5min": nan_safe_mean(convection_fraction_post_5min),
            "median_local_effective_D_mm2_s": nan_safe_median(res["D_local_map"]),
        }
        summary_row.update(regional_summary_flat)
        summary_row.update({f"{k}_fixed_roi": v for k, v in regional_summary_fixed_roi_flat.items()})
        summary_rows.append(summary_row)

    plot_multi_roi_summary(
        times_plot,
        roi_results,
        os.path.join(OUTPUT_FOLDER, "multi_roi_concentration_comparison.png")
    )

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = os.path.join(OUTPUT_FOLDER, "multi_roi_summary.csv")
    summary_txt_path = os.path.join(OUTPUT_FOLDER, "multi_roi_summary.txt")
    summary_df.to_csv(summary_csv_path, index=False)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        for _, row in summary_df.iterrows():
            f.write("=" * 60 + "\n")
            roi_name = row.get("roi_name", "Unknown ROI")
            f.write(f"ROI = {roi_name}\n")
            f.write("=" * 60 + "\n\n")
            for col in summary_df.columns:
                value = row[col]
                f.write(f"{col} = {value}\n")
            f.write("\n")

    combined_df = pd.DataFrame({"time": times_plot})
    for roi_name, res in roi_results.items():
        combined_df[f"{roi_name}_mean_conc"] = res["mean_conc"]
        combined_df[f"{roi_name}_mean_conc_hu_noise_std"] = res["mean_conc_hu_noise_std"]
        combined_df[f"{roi_name}_mean_conc_roi_sensitivity_std"] = res["mean_conc_roi_sensitivity_std"]
        combined_df[f"{roi_name}_mean_conc_calibration_std"] = res["mean_conc_calibration_std"]
        combined_df[f"{roi_name}_mean_conc_fixed_roi_std"] = res["mean_conc_fixed_roi_std"]
        combined_df[f"{roi_name}_mean_conc_fixed_roi_ci_low"] = res["mean_conc_fixed_roi_ci_low"]
        combined_df[f"{roi_name}_mean_conc_fixed_roi_ci_high"] = res["mean_conc_fixed_roi_ci_high"]
        combined_df[f"{roi_name}_mean_conc_combined_std"] = res["mean_conc_combined_std"]
        combined_df[f"{roi_name}_mean_conc_combined_ci_low"] = res["mean_conc_combined_ci_low"]
        combined_df[f"{roi_name}_mean_conc_combined_ci_high"] = res["mean_conc_combined_ci_high"]
        combined_df[f"{roi_name}_effective_diffusivity"] = res["D_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_std"] = res["D_std_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_hu_noise_std"] = res["D_noise_std_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_roi_sensitivity_std"] = res["D_roi_sensitivity_std_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_calibration_std"] = res["D_calibration_std_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_fixed_roi_std"] = res["D_fixed_roi_std_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_fixed_roi_ci_low"] = res["D_fixed_roi_ci_low_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_fixed_roi_ci_high"] = res["D_fixed_roi_ci_high_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_combined_std"] = res["D_combined_std_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_combined_ci_low"] = res["D_combined_ci_low_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_combined_ci_high"] = res["D_combined_ci_high_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_ci_low"] = res["D_ci_low_vs_time"]
        combined_df[f"{roi_name}_effective_diffusivity_ci_high"] = res["D_ci_high_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs"] = res["Cs_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_std"] = res["Cs_std_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_hu_noise_std"] = res["Cs_noise_std_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_roi_sensitivity_std"] = res["Cs_roi_sensitivity_std_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_calibration_std"] = res["Cs_calibration_std_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_fixed_roi_std"] = res["Cs_fixed_roi_std_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_fixed_roi_ci_low"] = res["Cs_fixed_roi_ci_low_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_fixed_roi_ci_high"] = res["Cs_fixed_roi_ci_high_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_combined_std"] = res["Cs_combined_std_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_combined_ci_low"] = res["Cs_combined_ci_low_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_combined_ci_high"] = res["Cs_combined_ci_high_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_ci_low"] = res["Cs_ci_low_vs_time"]
        combined_df[f"{roi_name}_fitted_Cs_ci_high"] = res["Cs_ci_high_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity"] = res["v_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_std"] = res["v_std_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_hu_noise_std"] = res["v_noise_std_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_roi_sensitivity_std"] = res["v_roi_sensitivity_std_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_calibration_std"] = res["v_calibration_std_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_fixed_roi_std"] = res["v_fixed_roi_std_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_fixed_roi_ci_low"] = res["v_fixed_roi_ci_low_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_fixed_roi_ci_high"] = res["v_fixed_roi_ci_high_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_combined_std"] = res["v_combined_std_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_combined_ci_low"] = res["v_combined_ci_low_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_combined_ci_high"] = res["v_combined_ci_high_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_ci_low"] = res["v_ci_low_vs_time"]
        combined_df[f"{roi_name}_fitted_velocity_ci_high"] = res["v_ci_high_vs_time"]
        combined_df[f"{roi_name}_diffusion_curve"] = res["diffusion_curve"]
        combined_df[f"{roi_name}_convection_curve"] = res["convection_curve"]
        combined_df[f"{roi_name}_total_curve"] = res["total_curve"]
        combined_df[f"{roi_name}_mean_diffusive_flux_magnitude"] = res["mean_flux_mag"]
        combined_df[f"{roi_name}_peclet_number"] = res["peclet_vs_time"]
        combined_df[f"{roi_name}_peclet_characteristic_length_mm"] = res["peclet_length_mm"]
        combined_df[f"{roi_name}_mean_convection_fraction"] = res["convection_fraction_vs_time"]

    combined_df.to_csv(os.path.join(OUTPUT_FOLDER, "multi_roi_timecourse_comparison.csv"), index=False)

    run_metadata["output_files"] = {
        "all_rois_overlay_png": os.path.abspath(os.path.join(OUTPUT_FOLDER, "all_rois_overlay.png")),
        "multi_roi_concentration_comparison_png": os.path.abspath(os.path.join(OUTPUT_FOLDER, "multi_roi_concentration_comparison.png")),
        "multi_roi_summary_csv": os.path.abspath(summary_csv_path),
        "multi_roi_summary_txt": os.path.abspath(summary_txt_path),
        "multi_roi_timecourse_comparison_csv": os.path.abspath(os.path.join(OUTPUT_FOLDER, "multi_roi_timecourse_comparison.csv")),
        "selected_rois_for_rerun_json": os.path.abspath(saved_roi_json_path) if saved_roi_json_path else None,
    }
    metadata_path = save_run_metadata(OUTPUT_FOLDER, run_metadata)

    print("\nFinished.")
    print(f"Outputs saved to: {os.path.abspath(OUTPUT_FOLDER)}")
    print("\nTop-level outputs:")
    print(" - all_rois_overlay.png")
    print(" - multi_roi_concentration_comparison.png")
    print(" - multi_roi_summary.csv")
    print(" - multi_roi_summary.txt")
    print(" - multi_roi_timecourse_comparison.csv")
    if saved_roi_json_path:
        print(" - selected_rois_for_rerun.json")
    print(" - run_metadata.json")
    print(f"Metadata saved to: {metadata_path}")
    print("\nPer ROI:")
    for roi_name, _ in named_rois:
        print(f" - {roi_name}/")


if __name__ == "__main__":
    main()