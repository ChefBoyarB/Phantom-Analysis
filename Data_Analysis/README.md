# Data Analysis Workflow

This folder now has **3 Python files**, but only **2 are meant to be run directly**.

## Which File Should I Run?

### 1. `Analysis_Main_Config_First.py`
Use this for **normal work going forward**.

- This is the **recommended workflow**
- It requires a JSON config file
- You do **not** edit the Python settings block for each run

Run from the repo root:

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Config_First.py --config configs\analysis\old\vis320_no_pressure_135kvp.json
```

### 2. `Analysis_Main_Legacy.py`
Use this only if you want the **older workflow**.

- You edit the hardcoded settings inside `Analysis_Main_Engine.py`
- Then you run the legacy launcher

Run from the repo root:

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Legacy.py
```

### 3. `Analysis_Main_Engine.py`
This is the **shared analysis engine**.

- Both launchers use this file
- You usually **do not run this file directly**
- Future updates to the analysis logic should be made here

## Recommended Everyday Workflow

Use the **config-first** workflow for new datasets.

### Step 1. Copy the closest existing config

Current config folders:

- `configs/analysis/old/`
- `configs/analysis/diffusion/`

Current example configs include:

- `configs/analysis/old/gad_pressure_135kvp.json`
- `configs/analysis/old/gad_no_pressure_135kvp.json`
- `configs/analysis/old/vis320_no_pressure_135kvp.json`
- `configs/analysis/diffusion/GAD_Run_1.json`
- `configs/analysis/diffusion/GAD_Run_2.json`
- `configs/analysis/diffusion/GAD_Run_3.json`
- `configs/analysis/diffusion/VIS320_Run_1.json`
- `configs/analysis/diffusion/VIS320_Run_2.json`
- `configs/analysis/diffusion/VIS320_Run_3.json`

Temporary comparison-only sample configs also live under `configs/analysis/diffusion/`:

- `configs/analysis/diffusion/GAD_Run_2_shift_down1_sample.json`
- `configs/analysis/diffusion/VIS320_Run_2_shift_up1_sample.json`

Those temporary sample configs are useful for isolated comparison checks, but they should not be treated as canonical run templates.

For a new run:

1. Copy the closest config
2. Rename it
3. Save it in the subfolder that best matches its role
4. Edit only the values that should change for the new dataset

Example:

- copy `configs/analysis/old/vis320_no_pressure_135kvp.json`
- rename it to something like `configs/analysis/diffusion/VIS320_Run_4.json`

### Step 2. Edit the config file

The most common fields you will change are:

- `dicom_folder`
- `output_folder`
- `pump_on`
- `fit_velocity`
- `convection_method`
- `hu_per_conc`
- `hu_offset`
- `hu_per_conc_std`
- `hu_offset_std`
- `roi_selection_mode`
- `manual_named_rois`
- `save_selected_rois_json`
- `skip_initial_frames`

You do **not** need to change every field for every dataset.

### Step 3. Run the config-first launcher

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Config_First.py --config configs\analysis\diffusion\your_new_config.json
```

## Legacy Workflow

If you want to use the older style:

1. Open `Analysis_Main_Engine.py`
2. Edit the hardcoded settings near the top of the file
3. Run:

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Legacy.py
```

This still uses the same analysis logic as the config-first workflow.

## Important Logic

### If I change the analysis code later, what happens?

If you update fitting logic, uncertainty logic, plotting, or output writing in `Analysis_Main_Engine.py`, those updates will be used by:

- `Analysis_Main_Config_First.py`
- `Analysis_Main_Legacy.py`

That is because both launchers call the same shared engine file.

### If I add a new setting later, what happens?

If you add or rename a setting in `Analysis_Main_Engine.py`, you may also need to update your JSON config files so they stay aligned with the code.

## Best Practice

Use this rule:

- **Changing datasets**: edit or create a config file
- **Changing analysis behavior**: edit `Analysis_Main_Engine.py`

That keeps dataset-specific settings separate from method changes.

## ROI Guidance

Try to keep ROI selection as consistent as possible within the same tracer or experiment group.

Use this rule:

- keep one default ROI size for each tracer or tracer-condition group when possible
- prefer shifting the ROI to match the same physical gel region rather than resizing it
- only resize when needed to exclude artifacts, ruptures, missing gel, or edge problems
- if resizing is necessary, make the smallest change that solves the problem
- document meaningful ROI deviations in the config `description` or `notes`

For example, for your VIS workflow, a reasonable template is to keep the same VIS ROI size across comparable datasets and allow small vertical or lateral shifts when gel height or alignment changes between runs.

The goal is not to force identical coordinates in every dataset. The goal is to keep the ROI as similar as possible in physical meaning while still excluding clearly invalid regions.

### ROI selection modes

The analysis engine currently supports three ROI workflows:

- `interactive`: draw ROIs with the mouse during the run
- `manual_list`: reuse exact ROI coordinates already saved in the config
- `manual_prompt`: type ROI coordinates into the terminal during the run

For locked reruns, `manual_list` is the usual choice.

### Reusing saved ROIs

If `save_selected_rois_json` is enabled, each run writes `selected_rois_for_rerun.json` into the output folder.

Use this file when you want to:

- confirm which ROI coordinates were actually used in a completed run
- copy those coordinates into `manual_named_rois` for a locked rerun
- keep ROI definitions consistent across repeated analyses

## Output Checking

When a run finishes, the most useful files for checking whether two runs match are:

- `multi_roi_summary.csv`
- `multi_roi_timecourse_comparison.csv`
- `selected_rois_for_rerun.json`
- `run_metadata.json`

`run_metadata.json` will usually differ in timestamp, output path, script path, and config information even when the actual numerical analysis results are identical.

## Short Version

Use this for normal runs:

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Config_First.py --config configs\analysis\diffusion\some_config.json
```

Use this only for old-style manual runs:

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Legacy.py
```
