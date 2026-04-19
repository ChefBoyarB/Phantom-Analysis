# Support both "python Comparison_Analysis/..." and package imports from repo root.
try:
    from .Pump_Profile_Comparison_Figure_Generator_Engine import main
except ImportError:
    from Pump_Profile_Comparison_Figure_Generator_Engine import main


if __name__ == "__main__":
    raise SystemExit(main())


r'''
import os
import glob
import zipfile
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# USER SETTINGS
# ============================================================

# These inputs can be either:
#   1) a .zip results folder exported from your analysis, or
#   2) an already-extracted results folder on disk.
PUMP_OFF_INPUT = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\GAD_Swollen_No_Pressure_135kvp"
PUMP_ON_INPUT  = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\GAD_Pressure_Test_Global_Shared_V"
OUTPUT_DIR     = r"C:\Users\brend\OneDrive - University of Toronto\VS Code Files\Results\Results_Paper_1\Pump_Analysis_GAD_0P_Vs_GAD_5.5P"

# If True, compare depth as a 0->1 fraction of each ROI height.
# This is recommended when the ROIs come from different chambers or
# have different numbers of depth samples.
USE_NORMALIZED_DEPTH = True

# Number of depth points used after interpolation to the common depth grid.
COMMON_DEPTH_POINTS = 200

# Each profile is normalized to its own peak for the main figure.
NORMALIZE_EACH_PROFILE_TO_OWN_PEAK = True

# Times (minutes) to show in the matched-profile panel.
PROFILE_TIMES_MIN = [0.5, 10.0, 20.0]

# Colormap limits for normalized maps and normalized difference map.
NORM_MAP_VMIN = 0.0
NORM_MAP_VMAX = 1.0
DIFF_MAP_ABS_LIM = 0.30

# Optional: if your fitted outputs skip the first few frames, you can set
# a minimum time here to remove very early unmatched frames.
MIN_TIME_MIN = 0.0

# ============================================================
# HELPERS
# ============================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def unzip_to_temp(zip_path: str, temp_root: str) -> str:
    name = Path(zip_path).stem.replace(" ", "_").replace("(", "").replace(")", "")
    out_dir = os.path.join(temp_root, name)
    ensure_dir(out_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    return out_dir


def resolve_input_to_folder(input_path: str, temp_root: str) -> tuple[str, str]:
    """
    Returns:
        folder_path : extracted folder or original folder
        input_kind  : "zip" or "folder"
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if os.path.isdir(input_path):
        return input_path, "folder"

    if zipfile.is_zipfile(input_path):
        return unzip_to_temp(input_path, temp_root), "zip"

    raise ValueError(
        "Input must be either a .zip file or an extracted results folder. "
        f"Received: {input_path}"
    )


def find_single_file(root: str, pattern: str) -> str:
    matches = glob.glob(os.path.join(root, "**", pattern), recursive=True)
    if not matches:
        raise FileNotFoundError(f"Could not find {pattern} under {root}")
    # Prefer measured profiles and summaries closest to the deepest folder.
    matches = sorted(matches, key=lambda p: (p.count(os.sep), len(p)))
    return matches[-1]


def load_dataset_from_input(input_path: str, temp_root: str, label: str) -> dict:
    extracted, input_kind = resolve_input_to_folder(input_path, temp_root)

    profiles_csv = find_single_file(extracted, "measured_profiles_depth_vs_time.csv")
    params_csv = find_single_file(extracted, "fit_parameters_vs_time.csv")

    profiles = pd.read_csv(profiles_csv)
    params = pd.read_csv(params_csv)

    if "time_seconds" in params.columns:
        time_s = params["time_seconds"].to_numpy(dtype=float)
    elif "time_plot" in params.columns:
        # fallback: assume minutes if only time_plot exists
        time_s = params["time_plot"].to_numpy(dtype=float) * 60.0
    else:
        raise ValueError(f"No time column found in {params_csv}")

    # measured profiles are [time, depth]
    prof = profiles.to_numpy(dtype=float)

    # guard against a row-count mismatch
    n = min(len(time_s), prof.shape[0])
    time_s = time_s[:n]
    prof = prof[:n, :]

    return {
        "label": label,
        "input_path": input_path,
        "input_kind": input_kind,
        "extracted_dir": extracted,
        "profiles_csv": profiles_csv,
        "params_csv": params_csv,
        "time_s": time_s,
        "time_min": time_s / 60.0,
        "profiles_raw": prof,
        "n_time": prof.shape[0],
        "n_depth": prof.shape[1],
    }


def build_depth_axis(n_depth: int, use_normalized_depth: bool = True) -> np.ndarray:
    if use_normalized_depth:
        if n_depth == 1:
            return np.array([0.0], dtype=float)
        return np.linspace(0.0, 1.0, n_depth)
    return np.arange(n_depth, dtype=float)


def interpolate_profiles_to_common_depth(profiles: np.ndarray,
                                         old_depth: np.ndarray,
                                         new_depth: np.ndarray) -> np.ndarray:
    out = np.full((profiles.shape[0], len(new_depth)), np.nan, dtype=float)
    for i in range(profiles.shape[0]):
        y = np.asarray(profiles[i], dtype=float)
        valid = np.isfinite(old_depth) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue
        out[i] = np.interp(new_depth, old_depth[valid], y[valid])
    return out


def normalize_profiles_to_peak(profiles: np.ndarray) -> np.ndarray:
    out = np.asarray(profiles, dtype=float).copy()
    for i in range(out.shape[0]):
        prof = out[i]
        m = np.nanmax(prof) if np.any(np.isfinite(prof)) else np.nan
        if np.isfinite(m) and m > 0:
            out[i] = prof / m
    return out


def align_by_nearest_time(off_time_min: np.ndarray,
                          on_time_min: np.ndarray,
                          max_delta_min: float = 0.11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build matched time indices using nearest neighbor matching.
    Default max_delta_min=0.11 is a little above one 10 s frame (~0.1667 min would
    be more permissive), but here we keep it fairly strict.
    """
    off_idx = []
    on_idx = []
    matched_t = []

    for i, t in enumerate(off_time_min):
        j = int(np.argmin(np.abs(on_time_min - t)))
        dt = abs(on_time_min[j] - t)
        if dt <= max_delta_min:
            off_idx.append(i)
            on_idx.append(j)
            matched_t.append(0.5 * (t + on_time_min[j]))

    return np.asarray(off_idx, dtype=int), np.asarray(on_idx, dtype=int), np.asarray(matched_t, dtype=float)


def select_target_time_indices(matched_times_min: np.ndarray, targets_min: list[float]) -> list[int]:
    idxs = []
    for tgt in targets_min:
        idxs.append(int(np.argmin(np.abs(matched_times_min - tgt))))
    return idxs


def center_of_mass_depth(norm_profiles: np.ndarray, depth: np.ndarray) -> np.ndarray:
    com = np.full(norm_profiles.shape[0], np.nan, dtype=float)
    for i in range(norm_profiles.shape[0]):
        p = np.asarray(norm_profiles[i], dtype=float)
        valid = np.isfinite(depth) & np.isfinite(p)
        if np.sum(valid) < 2:
            continue
        pv = p[valid]
        dv = depth[valid]
        s = np.sum(pv)
        if s > 0:
            com[i] = np.sum(dv * pv) / s
    return com


def deep_tail_fraction(norm_profiles: np.ndarray, depth: np.ndarray, deep_start: float = 0.67) -> np.ndarray:
    frac = np.full(norm_profiles.shape[0], np.nan, dtype=float)
    mask = depth >= deep_start
    if not np.any(mask):
        return frac
    for i in range(norm_profiles.shape[0]):
        p = np.asarray(norm_profiles[i], dtype=float)
        if not np.any(np.isfinite(p)):
            continue
        total = np.nansum(p)
        deep = np.nansum(p[mask])
        if total > 0:
            frac[i] = deep / total
    return frac


def save_dataframe(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)



def add_reference_time_lines(ax, matched_times_min: np.ndarray, reference_times_min: list[float], color: str = "white"):
    """Draw dashed vertical lines at the nearest matched times for readability."""
    if matched_times_min is None or len(matched_times_min) == 0:
        return
    used = []
    for target in reference_times_min:
        idx = int(np.argmin(np.abs(matched_times_min - target)))
        t_line = float(matched_times_min[idx])
        if any(abs(t_line - u) < 1e-9 for u in used):
            continue
        used.append(t_line)
        ax.axvline(t_line, color=color, linestyle="--", linewidth=1.3, alpha=0.95, zorder=5)


def make_main_4panel_figure(matched_times_min: np.ndarray,
                            depth_common: np.ndarray,
                            off_norm: np.ndarray,
                            on_norm: np.ndarray,
                            diff_norm: np.ndarray,
                            profile_time_idxs: list[int],
                            out_path: str,
                            use_normalized_depth: bool = True):
    fig = plt.figure(figsize=(15.5, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.34)

    y_label = "Normalized depth (0=top, 1=bottom)" if use_normalized_depth else "Depth index"
    x_label = "Time (min)"

    # Panel A
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(off_norm.T,
                     aspect="auto",
                     origin="upper",
                     extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
                     vmin=NORM_MAP_VMIN,
                     vmax=NORM_MAP_VMAX)
    ax1.set_title("A. PUMP OFF: NORMALIZED MEASURED CONCENTRATION MAP")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    add_reference_time_lines(ax1, matched_times_min, PROFILE_TIMES_MIN, color="white")
    cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Normalized concentration")

    # Panel B
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(on_norm.T,
                     aspect="auto",
                     origin="upper",
                     extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
                     vmin=NORM_MAP_VMIN,
                     vmax=NORM_MAP_VMAX)
    ax2.set_title("B. PUMP ON: NORMALIZED MEASURED CONCENTRATION MAP")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel(y_label)
    add_reference_time_lines(ax2, matched_times_min, PROFILE_TIMES_MIN, color="white")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("Normalized concentration")

    # Panel C
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(diff_norm.T,
                     aspect="auto",
                     origin="upper",
                     extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
                     vmin=-DIFF_MAP_ABS_LIM,
                     vmax=DIFF_MAP_ABS_LIM)
    ax3.set_title("C. NORMALIZED DIFFERENCE MAP (PUMP ON − PUMP OFF)")
    ax3.set_xlabel(x_label)
    ax3.set_ylabel(y_label)
    add_reference_time_lines(ax3, matched_times_min, PROFILE_TIMES_MIN, color="white")
    cbar3 = fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    cbar3.set_label("Δ normalized concentration")

    # Panel D with 3 subplots
    subgs = gs[1, 1].subgridspec(1, 3, wspace=0.52)
    titles = ["EARLY", "MID", "LATE"]
    for k, idx in enumerate(profile_time_idxs):
        ax = fig.add_subplot(subgs[0, k])
        t = matched_times_min[idx]
        ax.plot(off_norm[idx], depth_common, linewidth=2, label="Pump off")
        ax.plot(on_norm[idx], depth_common, linewidth=2, label="Pump on")
        ax.set_ylim(depth_common.max(), depth_common.min())
        ax.set_title(f"D{k+1}. {titles[k]}\n{t:.2f} min")
        ax.set_xlabel("Normalized concentration")
        if k == 0:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if k == 2:
            ax.legend(loc="lower right", frameon=False)

    fig.suptitle("MEASURED PROFILE COMPARISON: PUMP OFF VS PUMP ON", fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.91)
    fig.subplots_adjust(wspace=0.30)
    fig.subplots_adjust(top=0.91)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_metrics_figure(matched_times_min: np.ndarray,
                        com_off: np.ndarray, com_on: np.ndarray,
                        deep_off: np.ndarray, deep_on: np.ndarray,
                        out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    axes[0].plot(matched_times_min, com_off, linewidth=2, label="Pump off")
    axes[0].plot(matched_times_min, com_on, linewidth=2, label="Pump on")
    axes[0].set_title("CENTER-OF-MASS DEPTH VS TIME")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Normalized depth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False)

    axes[1].plot(matched_times_min, deep_off, linewidth=2, label="Pump off")
    axes[1].plot(matched_times_min, deep_on, linewidth=2, label="Pump on")
    axes[1].set_title("DEEP-TAIL FRACTION VS TIME")
    axes[1].set_xlabel("Time (min)")
    axes[1].set_ylabel("Fraction of profile in deep region")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.suptitle("SHAPE-BASED METRICS FROM NORMALIZED MEASURED PROFILES", fontsize=14.5)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_supplementary_absolute_figure(matched_times_min: np.ndarray,
                                       depth_common: np.ndarray,
                                       off_abs: np.ndarray,
                                       on_abs: np.ndarray,
                                       profile_time_idxs: list[int],
                                       out_path: str,
                                       use_normalized_depth: bool = True):
    fig = plt.figure(figsize=(15.5, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.34)

    y_label = "Normalized depth (0=top, 1=bottom)" if use_normalized_depth else "Depth index"

    vmax = np.nanmax([np.nanmax(off_abs), np.nanmax(on_abs)])
    vmin = 0.0

    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(off_abs.T,
                     aspect="auto",
                     origin="upper",
                     extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
                     vmin=vmin, vmax=vmax)
    ax1.set_title("PUMP OFF: ABSOLUTE MEASURED CONCENTRATION MAP")
    ax1.set_xlabel("Time (min)")
    ax1.set_ylabel(y_label)
    add_reference_time_lines(ax1, matched_times_min, PROFILE_TIMES_MIN, color="white")
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04).set_label("Concentration")

    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(on_abs.T,
                     aspect="auto",
                     origin="upper",
                     extent=[matched_times_min.min(), matched_times_min.max(), depth_common.max(), depth_common.min()],
                     vmin=vmin, vmax=vmax)
    ax2.set_title("PUMP ON: ABSOLUTE MEASURED CONCENTRATION MAP")
    ax2.set_xlabel("Time (min)")
    ax2.set_ylabel(y_label)
    add_reference_time_lines(ax2, matched_times_min, PROFILE_TIMES_MIN, color="white")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04).set_label("Concentration")

    subgs = gs[1, :].subgridspec(1, 3, wspace=0.46)
    titles = ["EARLY", "MID", "LATE"]
    for k, idx in enumerate(profile_time_idxs):
        ax = fig.add_subplot(subgs[0, k])
        t = matched_times_min[idx]
        ax.plot(off_abs[idx], depth_common, linewidth=2, label="Pump off")
        ax.plot(on_abs[idx], depth_common, linewidth=2, label="Pump on")
        ax.set_ylim(depth_common.max(), depth_common.min())
        ax.set_title(f"{titles[k]} ABSOLUTE PROFILES\n{t:.2f} MIN")
        ax.set_xlabel("Concentration")
        if k == 0:
            ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        if k == 2:
            ax.legend(loc="lower right", frameon=False)

    fig.suptitle("SUPPLEMENTARY ABSOLUTE MEASURED-PROFILE COMPARISON", fontsize=16, y=0.99)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ensure_dir(OUTPUT_DIR)

    temp_root = tempfile.mkdtemp(prefix="pump_profile_compare_")
    try:
        off = load_dataset_from_input(PUMP_OFF_INPUT, temp_root, "Pump off")
        on = load_dataset_from_input(PUMP_ON_INPUT, temp_root, "Pump on")

        # Apply time filter if requested
        off_keep = off["time_min"] >= MIN_TIME_MIN
        on_keep = on["time_min"] >= MIN_TIME_MIN

        off_time = off["time_min"][off_keep]
        on_time = on["time_min"][on_keep]
        off_prof_raw = off["profiles_raw"][off_keep]
        on_prof_raw = on["profiles_raw"][on_keep]

        off_depth = build_depth_axis(off_prof_raw.shape[1], use_normalized_depth=USE_NORMALIZED_DEPTH)
        on_depth = build_depth_axis(on_prof_raw.shape[1], use_normalized_depth=USE_NORMALIZED_DEPTH)
        depth_common = np.linspace(0.0 if USE_NORMALIZED_DEPTH else 0.0,
                                   1.0 if USE_NORMALIZED_DEPTH else float(min(off_prof_raw.shape[1]-1, on_prof_raw.shape[1]-1)),
                                   COMMON_DEPTH_POINTS)

        off_abs_common = interpolate_profiles_to_common_depth(off_prof_raw, off_depth, depth_common)
        on_abs_common = interpolate_profiles_to_common_depth(on_prof_raw, on_depth, depth_common)

        if NORMALIZE_EACH_PROFILE_TO_OWN_PEAK:
            off_norm_common = normalize_profiles_to_peak(off_abs_common)
            on_norm_common = normalize_profiles_to_peak(on_abs_common)
        else:
            off_norm_common = off_abs_common.copy()
            on_norm_common = on_abs_common.copy()

        off_idx, on_idx, matched_t = align_by_nearest_time(off_time, on_time, max_delta_min=0.17)
        if len(matched_t) < 3:
            raise RuntimeError("Not enough matched timepoints found between the two datasets.")

        off_abs_matched = off_abs_common[off_idx]
        on_abs_matched = on_abs_common[on_idx]
        off_norm_matched = off_norm_common[off_idx]
        on_norm_matched = on_norm_common[on_idx]
        diff_norm = on_norm_matched - off_norm_matched

        # Shape metrics
        com_off = center_of_mass_depth(off_norm_matched, depth_common)
        com_on = center_of_mass_depth(on_norm_matched, depth_common)
        deep_off = deep_tail_fraction(off_norm_matched, depth_common, deep_start=0.67)
        deep_on = deep_tail_fraction(on_norm_matched, depth_common, deep_start=0.67)

        # Summary CSV
        summary = pd.DataFrame({
            "matched_time_min": matched_t,
            "center_of_mass_depth_pump_off": com_off,
            "center_of_mass_depth_pump_on": com_on,
            "center_of_mass_depth_difference_on_minus_off": com_on - com_off,
            "deep_tail_fraction_pump_off": deep_off,
            "deep_tail_fraction_pump_on": deep_on,
            "deep_tail_fraction_difference_on_minus_off": deep_on - deep_off,
        })
        save_dataframe(summary, os.path.join(OUTPUT_DIR, "normalized_profile_shape_metrics.csv"))

        # Early/mid/late profile picks
        profile_time_idxs = select_target_time_indices(matched_t, PROFILE_TIMES_MIN)

        make_main_4panel_figure(
            matched_times_min=matched_t,
            depth_common=depth_common,
            off_norm=off_norm_matched,
            on_norm=on_norm_matched,
            diff_norm=diff_norm,
            profile_time_idxs=profile_time_idxs,
            out_path=os.path.join(OUTPUT_DIR, "main_4panel_normalized_measured_profile_comparison.png"),
            use_normalized_depth=USE_NORMALIZED_DEPTH
        )

        make_metrics_figure(
            matched_times_min=matched_t,
            com_off=com_off,
            com_on=com_on,
            deep_off=deep_off,
            deep_on=deep_on,
            out_path=os.path.join(OUTPUT_DIR, "normalized_profile_shape_metrics_vs_time.png")
        )

        make_supplementary_absolute_figure(
            matched_times_min=matched_t,
            depth_common=depth_common,
            off_abs=off_abs_matched,
            on_abs=on_abs_matched,
            profile_time_idxs=profile_time_idxs,
            out_path=os.path.join(OUTPUT_DIR, "supplementary_absolute_measured_profile_comparison.png"),
            use_normalized_depth=USE_NORMALIZED_DEPTH
        )

        # Save aligned matrices too, so you can replot quickly later
        save_dataframe(pd.DataFrame(off_abs_matched), os.path.join(OUTPUT_DIR, "pump_off_absolute_profiles_aligned.csv"))
        save_dataframe(pd.DataFrame(on_abs_matched), os.path.join(OUTPUT_DIR, "pump_on_absolute_profiles_aligned.csv"))
        save_dataframe(pd.DataFrame(off_norm_matched), os.path.join(OUTPUT_DIR, "pump_off_normalized_profiles_aligned.csv"))
        save_dataframe(pd.DataFrame(on_norm_matched), os.path.join(OUTPUT_DIR, "pump_on_normalized_profiles_aligned.csv"))
        save_dataframe(pd.DataFrame(diff_norm), os.path.join(OUTPUT_DIR, "normalized_difference_profiles_aligned.csv"))
        save_dataframe(pd.DataFrame({"matched_time_min": matched_t}), os.path.join(OUTPUT_DIR, "matched_times_min.csv"))
        save_dataframe(pd.DataFrame({"common_depth_axis": depth_common}), os.path.join(OUTPUT_DIR, "common_depth_axis.csv"))

        print("Done.")
        print(f"Output directory: {OUTPUT_DIR}")

    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
'''
