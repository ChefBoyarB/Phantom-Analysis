# Temporary Outputs

This folder contains temporary local outputs created during troubleshooting, reruns, and isolated comparison checks.

## Current Contents

- `GAD_Run_2_shift_down1_sample/`
- `GAD_Run_2_shift_down1_sample_rerun/`
- `VIS320_Run_2_shift_up1_sample/`

These folders correspond to temporary sample runs used for comparison or ROI-sensitivity checks rather than the accepted main study outputs.

## How To Treat This Folder

- use it for temporary validation work
- do not treat these folders as canonical manuscript results
- prefer the accepted results folders under your main paper-results location for final comparisons and tables
- document any temporary-run purpose in the matching config `description` or `notes`

If a temporary rerun becomes important enough to keep as a reusable analysis setup, promote the settings into a named config under `configs/analysis/diffusion/` rather than relying on the output folder alone.
