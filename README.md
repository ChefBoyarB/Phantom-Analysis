# Phantom Analysis Codes

This repo contains scripts for analyzing tracer/contrast transport datasets, comparing result outputs, and generating paper-ready support tables.

## Main Folders

### `Data_Analysis/`
Runs the main single-dataset transport analysis workflow.

See:

- [Data_Analysis/README.md](./Data_Analysis/README.md)

That README explains:

- the config-first workflow
- the legacy workflow
- which analysis file to run
- how to create and use configs
- how the current analysis configs are organized into subfolders

### `Comparison_Analysis/`
Contains scripts that compare outputs across completed analysis runs.

Examples include:

- map comparisons
- time-course comparisons
- profile-fit comparison figures
- pump on vs pump off comparisons

See:

- [Comparison_Analysis/README.md](./Comparison_Analysis/README.md)

That README explains:

- the config-driven profile-fit comparison workflow
- how to compare multiple samples per tracer
- which comparison config file to edit
- which CSV/statistics outputs are written

Current example comparison configs include:

- `configs/comparison/profile_fit_examples_vis_vs_gad.json`
- `configs/comparison/profile_fit_examples_vis320_gad_n2.json`

### `Table_Generation/`
Contains scripts that generate support tables and manuscript-style outputs from completed analysis results.

These scripts can write outputs such as:

- CSV
- Markdown
- HTML
- LaTeX `.tex`
- DOCX

## Dependencies

Install the Python dependencies from the repo root:

```powershell
py -m pip install -r requirements.txt
```

Or use the project virtual environment:

```powershell
& '.\.venv\Scripts\python.exe' -m pip install -r requirements.txt
```

## Recommended Workflow

For new dataset analysis work:

1. Use the config-first analysis workflow in `Data_Analysis/`
2. Create or copy a config under `configs/analysis/`
   Current analysis configs are organized into subfolders such as `Old/` and `Diffusion/`
3. Run the analysis using `Analysis_Main_Config_First.py`
4. Create or update a comparison config in `configs/comparison/` when you want to compare multiple analyzed runs
5. Use `Comparison_Analysis/` and `Table_Generation/` after the per-dataset analysis outputs are generated

For example, the current four-run VIS-vs-GAD profile-fit comparison uses:

- `configs/comparison/profile_fit_examples_vis320_gad_n2.json`

## Notes

- `Analysis_Main_Engine.py` is now the shared analysis engine behind the config-first and legacy launchers.
- `configs/analysis/` holds reusable dataset/run settings for the config-first workflow and is currently organized into subfolders such as `Old/` and `Diffusion/`.
- `configs/comparison/` holds reusable settings for comparison workflows such as multi-sample profile-fit comparisons.
- Generated output folders are not the same thing as source code; keep code changes and result folders conceptually separate when using Git.
