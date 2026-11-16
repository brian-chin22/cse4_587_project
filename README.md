# Car Insurance Claim Prediction — CSE 487/587

Team: Elijah Sagaran, Pranjal Singh, and Brian Chin.

This project studies which driver and vehicle characteristics are associated with car insurance claims and predicts whether a driver files a claim (`OUTCOME = 1`). It progresses from local data analysis and modeling to a Spark and Databricks pipeline.

## Repository layout

| Path | Contents |
|---|---|
| `Car_Insurance_Claim.csv` | Shared dataset: 10,000 records and 19 columns |
| `phase_1/` | Exploratory analysis, figures, assignment requirements, notebook, and report |
| `phase_2/` | Modeling and interpretation, evaluation figures, assignment requirements, notebook, and report |
| `phase_3/` | Current Spark/Databricks notebooks, split assignments, supporting script, and team instructions |
| `phase_3_earlier_full_draft/` | Earlier Bronze/Silver/Gold implementations for reference; not the current run sequence |
| `requirements.txt` | Dependencies for the local Python workflow |

Local environments, caches, notebook checkpoints, build artifacts, submission ZIPs, and LaTeX source files (`*.tex`) are excluded through `.gitignore`. Report PDFs and figures are retained.

## Phase 3 overview and current status

The intended workflow is:

```text
Source CSV → Bronze (raw) → Silver (cleaned) → Gold (features and summaries)
                                                      ↓
Split lookup → shared train/validation/test assignments → Spark ML models
```

| Notebook or file | Role | Current status |
|---|---|---|
| `00_setup.ipynb` | Shared names, storage setup, Delta validation, split loading, and evaluation helpers | Code present |
| `01_bronze_layer.ipynb` | Preserve source records in Delta | TODO skeleton |
| `02_silver_layer.ipynb` | Clean and standardize records | TODO skeleton |
| `03_gold_layer.ipynb` | Create model features and reporting summaries | TODO skeleton |
| `03b_delta_tables.ipynb` | Persist split assignments and read back all tables for validation | Code present; depends on completed layers |
| `04a_mllib_logistic_regression.ipynb` | Train, tune, evaluate, save, and reload logistic regression | Code present; depends on completed layers |
| `04x_mllib_model_template.ipynb` | Starting point for additional models | Template |
| `src/make_split.py` | Generate the Phase 2 split lookup locally | Implemented; lookup included |

The current notebooks have no saved execution outputs. The pipeline is not yet complete or verified end to end in Databricks. Additional model notebooks and `report.pdf` remain to be added. The assignment requires at least two models for CSE 487 or three for CSE 587.

See the [Phase 3 README](phase_3/README.md) for ownership, helper functions, and table contracts. The Git-folder workflow below provides an alternative to its manual notebook import instructions.

## Open only Phase 3 in Databricks

All phases can remain in this repository. Use sparse checkout to include Phase 3 in your Databricks Git folder:

1. In Databricks, open **Settings → Linked accounts → Add Git credential**, select GitHub, and link an account with access to this repository.
2. Open **Workspace → Create → Git folder**.
3. Enter the repository URL:

   ```text
   https://github.com/brian-chin22/cse4_587_project.git
   ```

4. Enable **Sparse checkout mode** and enter this cone pattern:

   ```text
   phase_3
   ```

5. Create the Git folder and open `phase_3/notebooks/`.

Cone-mode sparse checkout includes the selected directory recursively and files directly at the repository root. The root CSV therefore remains available; the other phase directories are excluded. It does not remove them from GitHub.

Keep all current notebooks together: their `%run ./00_setup` cells depend on that relative location.

References: [Connect a Git provider](https://docs.databricks.com/aws/en/repos/get-access-tokens-from-git-provider) and [Databricks sparse checkout](https://docs.databricks.com/aws/en/repos/git-operations-with-repos/#configure-sparse-checkout-mode).

## Prepare data in Databricks

Git synchronizes project files into the workspace. The pipeline reads its CSV inputs from Unity Catalog volumes, so copy the files there before running the layers.

Create a Python notebook alongside `00_setup` and run this as its first cell:

```python
%run ./00_setup
```

In a second cell, replace `YOUR_EMAIL` and the Git folder name with your actual absolute workspace path:

```python
from pathlib import Path
import shutil

repo_root = Path("/Workspace/Users/YOUR_EMAIL/cse4_587_project")
copies = [
    (repo_root / "Car_Insurance_Claim.csv", Path(CLAIMS_CSV)),
    (repo_root / "phase_3/data/split_assignment.csv", Path(SPLIT_CSV)),
]

for source, destination in copies:
    assert source.is_file(), f"Missing source file: {source}"

for source, destination in copies:
    shutil.copyfile(source, destination)
    print(f"Copied {source.name} to {destination}")

require_source_files(CLAIMS_CSV, SPLIT_CSV)
```

This copies both files into the volumes created by setup, overwriting existing copies. Run it initially and after changes to the input files. The workspace path is user-specific; this preparation snippet must be verified in your workspace.

Default locations configured in `00_setup`:

| Resource | Location |
|---|---|
| Catalog and schema | `workspace.car_insurance` |
| Source files | `/Volumes/workspace/car_insurance/raw/` |
| Saved models | `/Volumes/workspace/car_insurance/models/` |
| Tables | `claims_bronze`, `claims_silver`, `claims_gold_features`, `claims_gold_summary`, `claims_model_split` within the schema above |

If your catalog differs, update `CATALOG` in `00_setup`.

## Run Phase 3

After completing the TODOs and preparing the inputs, run each notebook from top to bottom:

1. `01_bronze_layer`
2. `02_silver_layer`
3. `03_gold_layer`
4. `03b_delta_tables`
5. `04a_mllib_logistic_regression` and additional completed model notebooks

Each notebook loads `00_setup` automatically. Layer writes replace their current table contents. Coordinate runs when using the shared table names.

Preprocessing that learns statistics, such as imputation and scaling, belongs in the model pipeline and is fitted on training data only. The shared split contains 6,400 training, 1,600 validation, and 2,000 test drivers. Preserve IDs through the data layers so the split lookup can be joined correctly.

## Local Python setup

From a dedicated clone of this repository:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For regenerating only the split lookup, the required packages are `pandas` and `scikit-learn`. Run from the repository root:

```bash
python phase_3/src/make_split.py \
  Car_Insurance_Claim.csv \
  phase_3/data/split_assignment.csv
```

The script reproduces the earlier scikit-learn split using seed `123`; preserve source row order when regenerating it. It overwrites the lookup and prints split sizes and claim rates. The Databricks notebooks require the Spark/Databricks runtime and do not run as ordinary local Python scripts.

## Team Git workflow

1. Work in a dedicated local checkout or a personal Databricks Git folder and development branch.
2. Commit and push changes to GitHub; merge reviewed work into the shared branch.
3. Pull the desired branch in Databricks before running updated notebooks.
4. Recopy changed input CSVs and rerun affected pipeline stages as needed.

Each member should operate Git in their own Databricks Git folder. Shared catalog tables still require coordination even when notebook checkouts are separate. Pulling code does not rerun notebooks or rebuild tables.

## Submission

Complete and execute the notebooks, add the Phase 3 report, and package the required deliverables according to the assignment PDF in `phase_3/`. A GitHub push does not submit the project to UBlearn. Generated ZIPs remain local and are excluded from version control.
