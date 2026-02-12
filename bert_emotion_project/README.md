# BERT Emotion Classification (Phase 0)

Project skeleton for fine-tuning BERT on an emotion classification dataset.

Objective
--------
- Build a clean, modular, production-ready project for BERT fine-tuning.

Architecture Philosophy
-----------------------
- Single `src/` package for code; clear separation of `data/`, `models/`, `docs/`, and `reports/`.
- Configuration centralized in `src/config.py`.
- Small, testable modules that avoid heavy I/O at import time.

Phase-based Development
-----------------------
Phase 0 focuses on project structure, configuration, and reproducible utilities. Training,
model forward pass, and data integration will be implemented in subsequent phases.

Environment setup (Phase 1)
---------------------------
Create an isolated Python virtual environment before installing dependencies. Do NOT use your global Python.

If Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If Mac / Linux (bash/zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:
- The repo includes a `scripts/phase1_data_prep.py` script to load the dataset, perform EDA,
  create a stratified validation split, compute percentile-based `MAX_LENGTH`, and save plots
  to `reports/figures/`.
- Run the script after activating the virtual environment:

```bash
python scripts/phase1_data_prep.py
```

The script will generate `data/processed/` artifacts and `docs/PHASE_1_DATA_SUMMARY.md` with
dataset statistics. See that file for detailed results after running.
