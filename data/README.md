# Data

The raw datasets are not included in this repository. They are public, but their licences belong to the original providers. The sources below are the ones used in the dissertation.

Place the downloaded files in `data/raw/`, or point the `HDIOD_DATA_DIR` environment variable to another folder.

## Pima Indians Diabetes

- The dissertation used the Kaggle copy `uciml/pima-indians-diabetes-database` (file `diabetes.csv`), which is no longer available on Kaggle.
- The same 768-record dataset is available on OpenML: https://www.openml.org/d/37
- If `diabetes.csv` is not in `data/raw/`, the loader downloads it from OpenML automatically, so no file is needed.

## Thyroid and Annthyroid (UCI Machine Learning Repository)

- Source: https://archive.ics.uci.edu/dataset/102/thyroid+disease
- Files: `ann-train.data` (Thyroid), `ann-train.data` and `ann-test.data` (Annthyroid)

## Arrhythmia (UCI Machine Learning Repository)

- Source: https://archive.ics.uci.edu/dataset/5/arrhythmia
- File: `arrhythmia.data`

## Vertebral Column (UCI Machine Learning Repository)

- Source: https://archive.ics.uci.edu/dataset/212/vertebral+column
- Downloaded automatically with `ucimlrepo`, so no file is needed:

```python
from ucimlrepo import fetch_ucirepo
vertebral = fetch_ucirepo(id=212)
```

## WPBC: Breast Cancer Wisconsin Prognostic (UCI Machine Learning Repository)

- Source: https://archive.ics.uci.edu/dataset/16/breast+cancer+wisconsin+prognostic
- Downloaded automatically with `ucimlrepo`, so no file is needed:

```python
from ucimlrepo import fetch_ucirepo
wpbc = fetch_ucirepo(id=16)
```
