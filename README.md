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

### `Comparison_Analysis/`
Contains scripts that compare outputs across completed analysis runs.

Examples include:

- map comparisons
- time-course comparisons
- profile-fit comparison figures
- pump on vs pump off comparisons

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
2. Create or copy a config in `configs/analysis/`
3. Run the analysis using `Analysis_Main_Config_First.py`
4. Use `Comparison_Analysis/` and `Table_Generation/` after the per-dataset analysis outputs are generated

## Notes

- `Analysis_Main_Engine.py` is now the shared analysis engine behind the config-first and legacy launchers.
- `configs/analysis/` holds reusable dataset/run settings for the config-first workflow.
- Generated output folders are not the same thing as source code; keep code changes and result folders conceptually separate when using Git.
