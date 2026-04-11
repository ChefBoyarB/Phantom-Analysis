# Comparison Analysis Workflow

This folder contains scripts that compare outputs from completed analysis runs.

## Main Scripts

### `Compare_Profile_Fit_Examples.py`
Use this to compare fitted concentration-depth profiles across multiple tracers and multiple samples per tracer.

- This is now a config-driven workflow
- You run it with `--config`
- The active comparison logic lives in `Compare_Profile_Fit_Examples_Engine.py`
- `Compare_Profile_Fit_Examples.py` is the normal entry point you should run

Run from the repo root:

```powershell
& '.\.venv\Scripts\python.exe' Comparison_Analysis\Compare_Profile_Fit_Examples.py --config configs\comparison\profile_fit_examples_vis_vs_gad.json
```

Another ready-to-edit example for the current `n = 2` VIS-vs-GAD comparison workflow is:

```powershell
& '.\.venv\Scripts\python.exe' Comparison_Analysis\Compare_Profile_Fit_Examples.py --config configs\comparison\profile_fit_examples_vis320_gad_n2.json
```

### `Compare_Maps.py`
Use this to compare concentration, flux, and diffusivity maps across multiple tracers and multiple samples per tracer.

- This is now a config-driven workflow
- You run it with `--config`
- The active comparison logic lives in `Compare_Maps_Engine.py`
- `Compare_Maps.py` is the normal entry point you should run

Run from the repo root:

```powershell
& '.\.venv\Scripts\python.exe' Comparison_Analysis\Compare_Maps.py --config configs\comparison\map_comparison_vis320_gad_n2.json
```

### Other comparison scripts

- `Compare_Time_Course_CSV_Plots.py`
- `Pump_Profile_Comparison_Figure_Generator.py`

## Profile-Fit Comparison Workflow

Use this script after you have already analyzed each dataset individually in `Data_Analysis/`.

The normal flow is:

1. Run `Analysis_Main_Config_First.py` once per dataset
2. Keep each dataset output folder
3. Add those output folders to a comparison config in `configs/comparison/`
4. Run `Compare_Profile_Fit_Examples.py --config ...`

## How The Config Works

The example config is:

- `configs/comparison/profile_fit_examples_vis_vs_gad.json`
- `configs/comparison/profile_fit_examples_vis320_gad_n2.json`

Important top-level fields:

- `results_source`
- `out_folder`
- `figure_title`
- `target_times_min`
- `report_windows`
- `stats_alpha`
- `stats_metrics`
- `tracers`

Each tracer entry defines:

- `name`
- `label`
- `color`
- `marker`
- `roi_folder`
- `dx_mm`
- `samples`

Each sample entry defines:

- `sample_id`
- `run_rel`

If needed, a sample can also override:

- `roi_folder`
- `dx_mm`
- `run_path`

Use `run_rel` when the sample folder is under a shared `results_source`.
Use `run_path` when you want to point directly to an output folder somewhere else.

Use a sample-level `dx_mm` override when runs in the same tracer group do not share the same pixel spacing. This is the recommended pattern when mixing runs that were reconstructed with different FOV or pixel size.

### Late-time windows

Use `report_windows` when you want summary outputs for windows like post-5-min.

Example:

```json
"report_windows": [
  {
    "name": "post_5min",
    "min_time_min": 5.0,
    "max_time_min": null
  }
]
```

This writes both generic window summary CSVs and window-specific files such as `post_5min_...csv`.

## Uncertainty Definitions

For profile uncertainty outputs, the script now follows the same split used by `Analysis_Main_Engine.py`.

### Combined uncertainty includes:

- ROI sensitivity uncertainty
- model-fit uncertainty
- calibration uncertainty
- HU-noise uncertainty

### Fixed-ROI uncertainty includes:

- model-fit uncertainty
- calibration uncertainty
- HU-noise uncertainty

Fixed-ROI uncertainty does **not** include ROI sensitivity uncertainty.

The audit JSON records these definitions and whether the comparison script used engine-written uncertainty CSVs or recomputed the same definitions from the component maps.

## Adding More Samples

If you have multiple samples for the same tracer, keep one tracer block and add more sample entries under it.

Example pattern:

```json
{
  "name": "VIS320",
  "label": "VIS 320",
  "color": "#1f77b4",
  "marker": "o",
  "roi_folder": "VIS_320",
  "dx_mm": 0.166,
  "samples": [
    { "sample_id": "VIS320_01", "run_rel": "VIS320_Run_01" },
    { "sample_id": "VIS320_02", "run_rel": "VIS320_Run_02" },
    { "sample_id": "VIS320_03", "run_rel": "VIS320_Run_03" }
  ]
}
```

Do the same for `GAD` or any other tracer.

## What The Script Writes

The script writes figure outputs under:

- `profile_fit_examples/no_uncertainty/`
- `profile_fit_examples/model_fit_only/`
- `profile_fit_examples/combined_95CI/`
- `profile_fit_examples/combined_1SD/`
- `profile_fit_examples/fixedROI_95CI/`
- `profile_fit_examples/fixedROI_1SD/`

It also writes summary tables under:

- `profile_fit_examples/summaries/per_sample_target_time_metrics.csv`
- `profile_fit_examples/summaries/per_sample_all_time_metrics.csv`
- `profile_fit_examples/summaries/tracer_group_metric_summary.csv`
- `profile_fit_examples/summaries/pairwise_significance_tests.csv`
- `profile_fit_examples/summaries/omnibus_significance_tests.csv`
- `profile_fit_examples/summaries/window_per_sample_metric_summary.csv`
- `profile_fit_examples/summaries/window_tracer_group_metric_summary.csv`
- `profile_fit_examples/summaries/window_pairwise_significance_tests.csv`
- `profile_fit_examples/summaries/window_omnibus_significance_tests.csv`

If you define a window named `post_5min`, it also writes:

- `profile_fit_examples/summaries/post_5min_per_sample_metric_summary.csv`
- `profile_fit_examples/summaries/post_5min_tracer_group_metric_summary.csv`
- `profile_fit_examples/summaries/post_5min_pairwise_significance_tests.csv`
- `profile_fit_examples/summaries/post_5min_omnibus_significance_tests.csv`

And it writes an audit trail under:

- `profile_fit_examples/audit/comparison_audit.json`
- `profile_fit_examples/audit/comparison_audit.txt`

## How To Read The Stats Outputs

### `per_sample_target_time_metrics.csv`
One row per sample per target time.

Use this when you want to inspect the raw matched values that feed the comparison.

### `tracer_group_metric_summary.csv`
Grouped summary by tracer and target time.

This gives you:

- `n`
- mean
- standard deviation
- SEM
- min
- max

### `pairwise_significance_tests.csv`
Pairwise Welch t-tests between tracers at each target time and for each metric.

- `p_value_raw` is the uncorrected p-value
- `p_value_holm` is the Holm-corrected p-value
- `significant = true` means the corrected p-value is below `alpha`

If each tracer has fewer than 2 samples, the script will still write the file, but it will note that significance testing is not valid yet.

### `omnibus_significance_tests.csv`
One-way ANOVA across tracers for each target time and metric.

This is only meaningful when you have 3 or more tracer groups and enough samples in each group.

If you are comparing only two tracers, the pairwise Welch t-test table is the main significance output to read.

## Practical Guidance

- Match profiles by physical time in minutes, not by frame index.
- Compare depth on the common `mm` grid created by the script, not by raw pixel row.
- Treat `profile_fit_r2` mainly as a QC metric. For that field, the main goal is high values for both tracers, not necessarily a significant difference.
- For tracer separation, the most informative metrics are usually `effective_diffusivity_mm2_s`, `fitted_profile_auc_concentration_x_mm`, and sometimes `fitted_Cs`.

## Best Practice

Use this rule:

- changing which datasets are being compared: edit the comparison config
- changing how the comparison is computed or plotted: edit `Compare_Profile_Fit_Examples_Engine.py`

That keeps study selection separate from method changes.

## Short Version

Use this for the profile-fit comparison workflow:

```powershell
& '.\.venv\Scripts\python.exe' Comparison_Analysis\Compare_Profile_Fit_Examples.py --config configs\comparison\your_comparison_config.json
```

## Map Comparison Workflow

Use this after the per-dataset outputs already exist in `Data_Analysis/`.

The ready-to-edit example config is:

- `configs/comparison/map_comparison_vis320_gad_n2.json`

The top-level structure mirrors the profile-fit comparison:

- `out_folder`
- `figure_title`
- `target_times_min`
- `target_time_map_key`
- `stats_alpha`
- `stats_metrics`
- `report_windows`
- `tracers`

Each tracer entry defines:

- `name`
- `label`
- `color`
- `marker`
- `samples`

Each sample can use one of:

- `run_rel`
- `run_path`
- `analysis_config`

`analysis_config` is the new shortcut for map comparison. When you use it, the script can infer:

- the analysis output folder from `settings.output_folder`
- `dx_mm` from `run_metadata.json`
- `roi_folder` when the analysis run has exactly one selected ROI

If needed, a sample can still override:

- `sample_id`
- `roi_folder`
- `dx_mm`

The map comparison writes outputs under:

- `map_comparisons/per_timepoint/`
- `map_comparisons/temporally_regularized/`
- `map_comparisons/secondary/`
- `map_comparisons/summaries/`
- `map_comparisons/audit/`

The main summary tables are:

- `map_comparisons/summaries/per_sample_target_time_map_metrics.csv`
- `map_comparisons/summaries/per_sample_all_time_map_metrics.csv`
- `map_comparisons/summaries/tracer_group_map_metric_summary.csv`
- `map_comparisons/summaries/pairwise_map_significance_tests.csv`
- `map_comparisons/summaries/omnibus_map_significance_tests.csv`

Windowed late-time summaries follow the same pattern as the profile-fit workflow.

Use this for the map comparison workflow:

```powershell
& '.\.venv\Scripts\python.exe' Comparison_Analysis\Compare_Maps.py --config configs\comparison\your_map_comparison_config.json
```
