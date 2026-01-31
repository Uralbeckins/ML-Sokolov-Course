# Agent Guidelines for ML Sokolov Course Repository

## Project Overview

This repository contains machine learning course materials, including Jupyter notebooks with homework assignments, seminar materials, and lecture PDFs. The primary focus is on linear models, classification, regression, and text analysis. The codebase consists entirely of Jupyter notebooks (`.ipynb` files) and large datasets (CSV files).

There are no traditional Python modules (`.py` files), test suites, or build systems configured. However, the following guidelines will help agents work effectively with the notebooks and maintain consistency.

## Build, Lint, and Test Commands

### Current State
No build, lint, or test commands are defined in the repository. There is no `pyproject.toml`, `setup.py`, `requirements.txt`, or CI configuration.

### Suggested Commands (if tools are added)
If you need to introduce linting/testing for Python code extracted from notebooks, consider these standard commands:

```bash
# Install development dependencies
pip install black isort ruff pytest nbqa

# Format Python code with black
black .

# Sort imports with isort
isort .

# Lint with ruff
ruff check .

# Run pytest (if test files exist)
pytest

# Run a single test
pytest path/to/test_file.py::test_function_name

# Lint notebooks with nbqa
nbqa black notebook.ipynb
nbqa isort notebook.ipynb
nbqa ruff notebook.ipynb
```

### Notebook Execution
To run a specific notebook:

```bash
jupyter notebook notebook.ipynb
# or
jupyter nbconvert --execute --to notebook notebook.ipynb --output executed.ipynb
```

## Code Style Guidelines

### Imports
- Follow standard scientific Python import order:
  1. Standard library imports
  2. Third-party imports (numpy, pandas, scikit-learn, matplotlib)
  3. Local imports (none in this repo)
- Use common aliases:
  ```python
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from sklearn import preprocessing, model_selection, metrics
  ```
- Import specific functions/classes when they are used frequently:
  ```python
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import roc_auc_score
  ```
- Avoid wildcard imports (`from module import *`) except for `%matplotlib inline` magic.

### Formatting
- Follow PEP 8 guidelines (79‑character line limit optional for notebooks).
- Use 4 spaces per indentation level (Jupyter default).
- Use descriptive variable names in **snake_case** (e.g., `feature_matrix`, `target_vector`).
- Use **UPPER_CASE** for constants.
- Use **CamelCase** for class definitions (if any).
- Use spaces around operators (`=`, `+`, `-`, `*`, `/`, `==`, etc.) and after commas.

### Naming Conventions
- **Variables**: `lowercase_with_underscores`
- **Functions**: `lowercase_with_underscores`
- **Classes**: `CamelCase`
- **Constants**: `UPPER_CASE_WITH_UNDERSCORES`
- **Private** (module‑internal) identifiers: prefix with `_` (e.g., `_helper_function`).

### Error Handling
- Prefer explicit error checking with `try`‑`except` blocks when dealing with file I/O, data loading, or API calls.
- Use informative error messages that help debug the issue.
- Avoid bare `except:` clauses; catch specific exceptions (`ValueError`, `KeyError`, `FileNotFoundError`, etc.).
- Use `logging` or `print` statements for debugging (not for production code).

### Type Hints
Although notebooks rarely use type hints, they are encouraged for clarity in any `.py` files that may be created:
```python
from typing import List, Tuple, Optional

def load_data(path: str) -> Tuple[pd.DataFrame, pd.Series]:
    ...
```

## Notebook‑Specific Conventions

### Cell Order
1. **Import cell** at the top (all imports together).
2. **Configuration cell** (e.g., `pd.set_option`, `%matplotlib inline`).
3. **Data loading** and preprocessing cells.
4. **Exploratory data analysis** (visualizations, summary statistics).
5. **Model training** and evaluation cells.
6. **Results** and discussion cells.

### Markdown Cells
- Use markdown for headings, lists, and explanations.
- Headings should follow the notebook’s hierarchical structure (`#`, `##`, `###`).
- Include clear problem statements and step‑by‑step instructions.

### Code Cells
- Keep cells reasonably short; each cell should perform a single logical step.
- Avoid extremely long output (use `.head()`, `.sample()`, or `display()`).
- Use comments sparingly; prefer self‑explanatory variable names and markdown explanations.

### Magic Commands
- Use `%matplotlib inline` for inline plots.
- Use `%load_ext` for extensions (e.g., `%load_ext autoreload`).
- Avoid shell magic (`!`) unless necessary (e.g., downloading data).

## Data Handling

### Data Location
Large datasets are stored in `data/` (currently ignored by `.gitignore`). The repository already contains `Train_rev1.csv` and `Test_rev1.csv` (≈2 GB total). Do not commit additional large files.

### Data Loading Patterns
```python
import pandas as pd
df = pd.read_csv('data/Train_rev1.csv')
```
- Use `pd.read_csv` with appropriate arguments (`sep`, `encoding`, `low_memory=False`) as needed.
- For memory‑efficient processing, consider `dtype` specification or chunking.

### Data Preprocessing
- Document any transformations (scaling, encoding, imputation) in markdown cells.
- Use scikit‑learn pipelines when possible for reproducibility.

## Additional Tools

### Environment Management
Although no environment file is provided, typical dependencies include:
- numpy
- pandas
- scikit‑learn
- matplotlib
- seaborn (optional)
- nltk (for text processing)
- torch (for neural networks)

Create a `requirements.txt` if needed:
```txt
numpy>=1.21
pandas>=1.3
scikit-learn>=1.0
matplotlib>=3.5
jupyter>=1.0
```

### Version Control
- The repository uses Git with a `.gitignore` that excludes `data/` and the large CSV files.
- Commit messages should be descriptive (e.g., “Add homework 4 solution”, “Fix data loading bug”).

### Cursor / Copilot Rules
No repository‑specific Cursor (`.cursor/rules/`) or Copilot (`.github/copilot‑instructions.md`) rules are present. Follow the general guidelines above.

## Summary for Agents

1. **No existing build/lint/test commands** – you may suggest adding them if the project evolves.
2. **Code style** follows standard Python scientific stack conventions.
3. **Notebooks** are the primary artifact; keep them well‑organized and executable.
4. **Data** is large; avoid committing it.
5. **Imports** follow the `np`, `pd`, `plt` aliases and scikit‑learn sub‑module imports.
6. **Error handling** should be explicit and informative.

Use these guidelines when creating, modifying, or reviewing code in this repository.