"""Reproduce the Phase 2 train / validation / test split as an ID -> split lookup.

Spark's randomSplit cannot reproduce scikit-learn's train_test_split, so the
Phase 3 pipeline loads this lookup instead of re-splitting. That keeps the same
2,000 test drivers as Phase 2, which is what makes the model comparison fair.

The two calls below are copied from the Phase 2 notebook (Main Function cell):
  1. 80/20 random split of all drivers, random_state=123
  2. 80/20 split of the training drivers, stratified on OUTCOME, random_state=123

Run locally (pandas + scikit-learn only):
    python make_split.py path/to/Car_Insurance_Claim.csv path/to/split_assignment.csv
"""

import sys

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 123


def build_split(data: pd.DataFrame) -> pd.DataFrame:
    outcome = data["OUTCOME"]
    features = data.drop(columns=["OUTCOME"])

    X_train, X_test, y_train, _ = train_test_split(
        features, outcome, test_size=0.2, random_state=SEED
    )
    X_fit, X_val, _, _ = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )

    split = pd.Series("", index=data.index, name="split")
    split.loc[X_fit.index] = "train"
    split.loc[X_val.index] = "validation"
    split.loc[X_test.index] = "test"
    return pd.DataFrame({"ID": data["ID"], "split": split})


if __name__ == "__main__":
    source_csv, output_csv = sys.argv[1], sys.argv[2]
    data = pd.read_csv(source_csv)
    assignment = build_split(data)

    assert assignment["ID"].is_unique, "ID must be unique to act as the join key"
    assert (assignment["split"] != "").all(), "every driver needs a split"

    assignment.to_csv(output_csv, index=False)

    report = assignment.assign(OUTCOME=data["OUTCOME"]).groupby("split")["OUTCOME"]
    print(report.agg(drivers="size", claims="sum", claim_rate="mean").round(4))
