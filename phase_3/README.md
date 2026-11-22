# Phase 3: Car Insurance Claim Prediction on Spark and Databricks

Predicting whether a driver files a car insurance claim (`OUTCOME = 1`), rebuilt from the Phase 1 and 2 Pandas / scikit-learn workflow as a Bronze → Silver → Gold Delta pipeline with Spark MLlib models.

## Who owns what

| Member | Tasks | Files |
|---|---|---|
| Elijah | Bronze and Silver layers, one MLlib model (decision tree) | `01_bronze_layer`, `02_silver_layer`, `04c_mllib_Decision_Tree` |
| Pranjal | Gold layer, one MLlib model (random forest) | `03_gold_layer`, `04b_mllib_Random_Forest` |
| Brian | Delta tables, one MLlib model (logistic regression) | `00_setup`, `03b_delta_tables`, `04a_mllib_logistic_regression`, `src/make_split.py` |

## File structure

```
phase_3/
├── README.md
├── report.pdf                              All      project report
├── data/
│   └── split_assignment.csv                Brian    ID → train / validation / test, from make_split.py
├── src/
│   └── make_split.py                       Brian    reproduces the Phase 2 scikit-learn split
└── notebooks/
    ├── 00_setup.ipynb                      Brian    shared names and helpers; run by every notebook
    ├── 01_bronze_layer.ipynb               Elijah   raw CSV → claims_bronze
    ├── 02_silver_layer.ipynb               Elijah   claims_bronze → claims_silver
    ├── 03_gold_layer.ipynb                 Pranjal  claims_silver → claims_gold_features, claims_gold_summary
    ├── 03b_delta_tables.ipynb              Brian    split table + read-back validation of all tables
    ├── 04a_mllib_logistic_regression.ipynb Brian    logistic regression
    ├── 04b_mllib_Random_Forest.ipynb       Pranjal  random forest
    └── 04c_mllib_Decision_Tree.ipynb       Elijah   decision tree
```

**Differences from the suggested structure in the assignment**, which allows another layout if it is explained:

- `00_setup` holds the table names and the Delta write and validation helpers, so the three of us write to the same tables in the same way. Every notebook runs it in its first code cell with `%run ./00_setup`.
- `03b_delta_tables` is the Task 6 evidence: it reads every table back from the catalog and verifies it.
- `04_mllib_models` is split into one notebook per model (`04a` logistic regression, `04b` random forest, `04c` decision tree) so each member can work and rerun independently.

## How we work together

1. **Edit only your own notebooks.** Everyone works in the shared `phase_3` folder of the Databricks workspace.
2. **Use the names from `00_setup`** (`BRONZE_TABLE`, `SILVER_TABLE`, `GOLD_FEATURES_TABLE`, `CLAIMS_CSV`, ...) instead of typing table names or paths. If something in `00_setup` needs to change, tell Brian, because it affects every notebook.
3. **End each layer notebook with `write_delta(...)` and `validate_tables(...)`.** `validate_tables` reads the table back and stops the notebook if it does not meet the contract below.
4. **Work order.** Each stage reads the previous stage's table: Elijah (Bronze, Silver) → Pranjal (Gold) → the three model notebooks. Tell the next person when your table is written.
5. **Models use the shared split and metrics.** `load_model_splits()` returns the Phase 2 train, validation and test drivers, and `evaluate_classifier(...)` computes the metrics the same way for all three models.
6. Before submission, each owner runs their notebook from top to bottom and exports it as `.ipynb`.

## Table contract

What each notebook must produce for the next one to work. `validate_tables` checks the "must contain" and "key" columns automatically.

| Table | Written by | Must contain | Key | Notes |
|---|---|---|---|---|
| `claims_bronze` | `01_bronze_layer` | The 19 source columns, original names and values | | 10,000 rows, one per source record. Malformed records are kept and flagged |
| `claims_silver` | `02_silver_layer` | The 19 source columns | `ID` | Numbers as numeric types, categories as text. No imputation, scaling or encoding |
| `claims_gold_features` | `03_gold_layer` | `ID`, `OUTCOME` | `ID` | Every other column is used as a model input. Only row-level features |
| `claims_gold_summary` | `03_gold_layer` | (free design) | | At least one aggregation tied to the project question |
| `claims_model_split` | `03b_delta_tables` | `ID`, `split` | `ID` | Phase 2 assignment: 6,400 train, 1,600 validation, 2,000 test |

**Why Silver and Gold must not impute, scale or encode.** Phase 2 fitted those steps on training drivers only. Doing them on all 10,000 rows would use test drivers to prepare the training data. Each model notebook does them inside its Spark ML pipeline, fitted on the training split.

**Why rows should not be removed without reporting it.** Phase 2 kept all 10,000 drivers. Every removed row changes the test set, and `03b_delta_tables` reports how many Phase 2 drivers are missing from each split.

## Models

All three are Phase 2 models rebuilt with Spark MLlib (`spark.ml`). Each is one Spark ML `Pipeline` (preprocessing and classifier) fitted on the 6,400 Phase 2 training drivers, tuned on the 1,600 validation drivers and scored once on the 2,000 test drivers. Every notebook reports accuracy, precision, recall, F1 and ROC-AUC for the claim class, plus the confusion matrix, through the shared `evaluate_classifier`, and compares them with the Phase 2 report.

| Notebook | Model | Preprocessing (fitted on train) | Tuned on validation | Saved to |
|---|---|---|---|---|
| `04a_mllib_logistic_regression` | `LogisticRegression` (L2) | `StringIndexer`, `OneHotEncoder`, median `Imputer`, `MinMaxScaler` | Regularization: 7 values of `C`, by accuracy | `/Volumes/workspace/car_insurance/models/logistic_regression` |
| `04b_mllib_Random_Forest` | `RandomForestClassifier` (300 trees) | `StringIndexer`, median `Imputer` | Maximum depth 4, 6, 8, 10, by ROC-AUC | `/Volumes/workspace/car_insurance/models/random_forest` |
| `04c_mllib_Decision_Tree` | `DecisionTreeClassifier` | `StringIndexer`, median `Imputer` | Maximum depth 3, 5, 8, 10 × minimum rows per node 1, 10, 25, by ROC-AUC | `/Volumes/workspace/car_insurance/models/decision_tree` |

Each notebook saves the fitted pipeline, loads it back and checks that the reloaded copy gives the same test predictions. To use a saved model elsewhere:

```python
from pyspark.ml import PipelineModel
model = PipelineModel.load("/Volumes/workspace/car_insurance/models/random_forest")
predictions = model.transform(drivers)   # same columns as claims_gold_features
```

## Workspace and table locations

- **Platform:** Databricks Free Edition, serverless compute, Unity Catalog.
- **Workspace folder:** `phase_3`
- **Catalog / schema:** `workspace.car_insurance` (tables are `workspace.car_insurance.<table name>`)
- **Source files (volume):** `/Volumes/workspace/car_insurance/raw/`
- **Saved models (volume):** `/Volumes/workspace/car_insurance/models/` with one folder per model: `logistic_regression`, `random_forest`, `decision_tree`

| Table | Full name | Rows |
|---|---|---|
| Bronze | `workspace.car_insurance.claims_bronze` | 10,000 |
| Silver | `workspace.car_insurance.claims_silver` | 10,000 |
| Gold features | `workspace.car_insurance.claims_gold_features` | 10,000 |
| Gold summary | `workspace.car_insurance.claims_gold_summary` | One row per group (claim rate by segment) |
| Split | `workspace.car_insurance.claims_model_split` | 10,000 |

If the catalog is not named `workspace`, change `CATALOG` in `00_setup`; nothing else needs editing.

## Data access

- **Dataset:** `Car_Insurance_Claim.csv`, 10,000 drivers, 19 columns, about 1.3 MB. It is the file used in Phases 1 and 2: <https://raw.githubusercontent.com/brian-chin22/cse4_587_project/refs/heads/main/Car_Insurance_Claim.csv>
- **Split lookup:** `data/split_assignment.csv`. To regenerate it locally (needs `pandas` and `scikit-learn`):

  ```
  python src/make_split.py Car_Insurance_Claim.csv data/split_assignment.csv
  ```

  Spark cannot reproduce scikit-learn's shuffle, so the pipeline loads this lookup to train and test on the same drivers as Phase 2.

## How to run

One-time setup:

1. In Databricks, create a workspace folder named `phase_3` and import all notebooks from `notebooks/` into it (right-click the folder → **Import**). They must stay in the same folder, because each one runs `00_setup` by relative path.
2. Open `03b_delta_tables`, attach **Serverless** compute and choose **Run all**. The first run creates the schema and the two volumes, then stops and asks for the source files.
3. Upload `Car_Insurance_Claim.csv` and `data/split_assignment.csv`: **Catalog** → `workspace` → `car_insurance` → `raw` → **Upload to this volume**.

Run order, each with **Run all**:

1. `01_bronze_layer`
2. `02_silver_layer`
3. `03_gold_layer`
4. `03b_delta_tables`
5. `04a_mllib_logistic_regression`, `04b_mllib_Random_Forest`, `04c_mllib_Decision_Tree` (any order, after `03b_delta_tables`)

`00_setup` is never run directly. Every notebook overwrites its own tables, so the pipeline can be rerun from the top at any time.

## Dependencies

- Databricks serverless compute. The model notebooks need serverless **environment version 4 or later**, which enables Spark ML. No extra libraries are installed.
- `src/make_split.py` runs locally with Python 3, `pandas` and `scikit-learn`.

## Team access

Team members are added under **Settings → Identity and access → Users**, and the `phase_3` folder is shared with edit permission from its **Share** menu. The workspace owner then runs the grants below once per member, in the SQL editor:

```sql
-- The schema and volumes must exist before they can be shared.
CREATE SCHEMA IF NOT EXISTS workspace.car_insurance;
CREATE VOLUME IF NOT EXISTS workspace.car_insurance.raw;
CREATE VOLUME IF NOT EXISTS workspace.car_insurance.models;

-- Open the catalog. 00_setup runs CREATE SCHEMA IF NOT EXISTS, which needs CREATE SCHEMA.
GRANT USE CATALOG, CREATE SCHEMA ON CATALOG workspace TO `<member email>`;
-- Read and write tables, upload source files to `raw`, save models to `models`.
GRANT ALL PRIVILEGES ON SCHEMA workspace.car_insurance TO `<member email>`;
-- Replace tables another member created (every notebook overwrites its own tables).
GRANT MANAGE ON SCHEMA workspace.car_insurance TO `<member email>`;
```

No passwords, tokens or other credentials are stored in this submission.
