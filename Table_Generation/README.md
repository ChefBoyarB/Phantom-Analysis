# Table Generation Workflow

This folder contains paper-facing table builders that read completed analysis outputs and write formatted support tables.

## Main Scripts

### `Create_Tracer_Stacked_Main_Table.py`
Builds the stacked main comparison table for one VIS run and one GAD run.

### `Create_EarlyMidLate_Profile_Concentration_Support_Table.py`
Builds an early/mid/late support table for fitted concentration-depth profiles.

### `Create_EarlyMidLate_Flux_Support_Table.py`
Builds an early/mid/late support table for diffusive flux profiles.

## How These Scripts Are Configured

These scripts currently use a top-of-file `USER INPUT` block rather than a JSON config.

Before running a script, update the source settings near the top of the file:

- `VIS_SOURCE`
- `GAD_SOURCE`
- `VIS_RUN_REL` and `GAD_RUN_REL` when the source points to a shared results root
- tracer labels and ROI folder names
- `VIS_DX_MM` and `GAD_DX_MM` when the script needs depth spacing
- target times or late-time window settings
- `OUT_FOLDER`
- output toggles such as `WRITE_CSV`, `WRITE_MARKDOWN`, `WRITE_HTML`, `WRITE_LATEX`, and `WRITE_DOCX`

The source paths can point to either:

- a single completed run folder or zip that already contains `multi_roi_summary.csv` and ROI subfolders
- a shared results root, together with the matching `*_RUN_REL` value

## Running From The Repo Root

After editing the `USER INPUT` block, run the script you want:

```powershell
& '.\.venv\Scripts\python.exe' Table_Generation\Create_Tracer_Stacked_Main_Table.py
```

```powershell
& '.\.venv\Scripts\python.exe' Table_Generation\Create_EarlyMidLate_Profile_Concentration_Support_Table.py
```

```powershell
& '.\.venv\Scripts\python.exe' Table_Generation\Create_EarlyMidLate_Flux_Support_Table.py
```

## Outputs

Depending on the write toggles you enable, the scripts can write:

- CSV
- Markdown
- HTML
- LaTeX `.tex`
- DOCX

## Best Practice

Use this rule:

- changing which completed runs feed a table: edit the `USER INPUT` block in the table script
- changing how the table is computed or formatted: edit the Python logic in that script

These table builders are more paper-specific than the main analysis and comparison workflows, so a small amount of script-level editing is expected.
