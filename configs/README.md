# Configs Folder

This folder holds reusable JSON configs for both single-run analysis and multi-run comparison workflows.

## Layout

- `analysis/`
- `comparison/`

The repo currently uses lowercase folder names:

- `configs/analysis/old/`
- `configs/analysis/diffusion/`

## `analysis/`

Use `configs/analysis/` with `Data_Analysis/Analysis_Main_Config_First.py`.

Current examples include:

- legacy-style templates in `configs/analysis/old/`
- diffusion study runs in `configs/analysis/diffusion/`
- current accepted diffusion configs such as `GAD_Run_1.json` through `GAD_Run_3.json` and `VIS320_Run_1.json` through `VIS320_Run_3.json`

Temporary comparison-only sample configs also live under `configs/analysis/diffusion/`, including:

- `GAD_Run_2_shift_down1_sample.json`
- `VIS320_Run_2_shift_up1_sample.json`

Those temporary sample configs are helpful for isolated checks, but they should not replace the main accepted run configs.

## `comparison/`

Use `configs/comparison/` with the scripts in `Comparison_Analysis/`.

Current examples include:

- `profile_fit_examples_vis_vs_gad.json`
- `profile_fit_examples_vis320_gad_n3.json`
- `map_comparison_vis320_gad_n2.json`
- `map_comparison_vis320_gad_n3.json`

The `n3` configs are the current three-samples-per-tracer VIS320-vs-GAD examples. The `n2` map config is kept as a smaller comparison example.

## Best Practice

Use this rule:

- changing which dataset or run is analyzed: edit a file under `configs/analysis/`
- changing which completed runs are compared: edit a file under `configs/comparison/`
- changing the method itself: edit the corresponding engine code in `Data_Analysis/` or `Comparison_Analysis/`
