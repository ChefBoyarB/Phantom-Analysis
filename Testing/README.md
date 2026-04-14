# Testing Folder

This folder holds lightweight smoke-test outputs and temporary validation assets for the comparison workflows.

## Current Contents

- `ProfileFitExamplesSmoke/`
- `ProfileFitExamplesSmokeLateWindow/`
- `Outputs/`
- `temp_map_compare_config.json`

## What These Are For

`ProfileFitExamplesSmoke/` and `ProfileFitExamplesSmokeLateWindow/` contain saved `profile_fit_examples/` output trees that are useful for quick regression checks after changing the comparison engine.

`Outputs/Map_Compare_Smoke/` is the scratch output area for map-comparison smoke runs.

`temp_map_compare_config.json` is a temporary map-comparison config used for validation work. It is useful for quick checks, but it should not be treated as a canonical study config.

## How To Treat This Folder

- use it for smoke tests and quick sanity checks
- do not treat these outputs as manuscript-ready results
- prefer `configs/comparison/` for reusable study configs
- prefer the main paper results folders for accepted analysis outputs

If you rerun a smoke test after changing comparison logic, this folder is the right place to inspect whether the expected output structure and summary tables are still being written.
