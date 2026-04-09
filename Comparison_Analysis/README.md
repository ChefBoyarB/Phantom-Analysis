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

### Other comparison scripts

- `Compare_Maps.py`
- `Compare_Time_Course_CSV_Plots.py`
- `Pump_Profile_Comparison_Figure_Generator.py`

Those scripts are still separate workflows and are not yet using the new multi-sample profile-fit config format.

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

Important top-level fields:

- `results_source`
- `out_folder`
- `figure_title`
- `target_times_min`
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
- `profile_fit_examples/summaries/tracer_group_metric_summary.csv`
- `profile_fit_examples/summaries/pairwise_significance_tests.csv`
- `profile_fit_examples/summaries/omnibus_significance_tests.csv`

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

If each tracer has fewer than 2 samples, the script will still write the file, but it will note that significance testing is not valid yet.

### `omnibus_significance_tests.csv`
One-way ANOVA across tracers for each target time and metric.

This is only meaningful when you have 3 or more tracer groups and enough samples in each group.

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
