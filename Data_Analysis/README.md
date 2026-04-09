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
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Config_First.py --config configs\analysis\vis320_no_pressure_135kvp.json
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

Current example configs:

- `configs/analysis/gad_pressure_135kvp.json`
- `configs/analysis/gad_no_pressure_135kvp.json`
- `configs/analysis/vis320_no_pressure_135kvp.json`

For a new run:

1. Copy the closest config
2. Rename it
3. Edit only the values that should change for the new dataset

Example:

- copy `vis320_no_pressure_135kvp.json`
- rename to `vis320_no_pressure_repeat2.json`

### Step 2. Edit the config file

The most common fields you will change are:

- `dicom_folder`
- `output_folder`
- `pump_on`
- `convection_method`
- `hu_per_conc`
- `hu_offset`
- `hu_per_conc_std`
- `hu_offset_std`
- `manual_named_rois`
- `skip_initial_frames`

You do **not** need to change every field for every dataset.

### Step 3. Run the config-first launcher

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Config_First.py --config configs\analysis\your_new_config.json
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
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Config_First.py --config configs\analysis\some_config.json
```

Use this only for old-style manual runs:

```powershell
& '.\.venv\Scripts\python.exe' Data_Analysis\Analysis_Main_Legacy.py
```
