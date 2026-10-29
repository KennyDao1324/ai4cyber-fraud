# AI4Cyber ML Training Run Log

This log is generated from the model card and dataset profile manifest.

## Run identity

| Item | Value |
| --- | --- |
| Log generated | 2026-10-29T16:57:35+07:00 |
| Architecture | ML-only |
| Selection rule | Highest validation PR-AUC; test data not used for selection |
| Random seed | 42 |
| Development period | Months 0-5 (0-4 train; 5 validation) |
| Held-out test period | Months 6-7 |

## Environment

| Component | Version |
| --- | ---: |
| Python | 3.12.6 |
| pandas | 2.3.3 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| FastAPI | 0.162.0 |
| Matplotlib | 3.11.2 |

## Source-data verification

- Kaggle dataset: `sgpjesus/bank-account-fraud-dataset-neurips-2022`
- Version: 2 (2023-11-29T11:32:12.037Z)
- Licence: CC BY-NC-SA 4.0
- Source: https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022

| Dataset | Rows | Columns | Fraud rate | Documented `-1` sentinels | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Base | 1,000,000 | 32 | 1.1029% | 973,183 | `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` |
| Variant I | 1,000,000 | 32 | 1.1029% | 965,874 | `48c637a255d1fa4515da4286ccb99251412a6d905937ec7c108838ddfc1674bd` |
| Variant II | 1,000,000 | 32 | 1.1029% | 1,007,388 | `60bcd971cf28779abb183c3286990cff9a87275bd8b83d9f6054a396c95cc736` |
| Variant III | 1,000,000 | 34 | 1.1030% | 1,015,308 | `64693e549cff7ef802cf043fc7c937839e5929966e33a0852953b3f4c4321339` |
| Variant IV | 1,000,000 | 32 | 1.1030% | 1,015,027 | `c6cddebbad34fa262a278deeb7985e7b669a64308d26e480d8eb85bfd08dac75` |
| Variant V | 1,000,000 | 34 | 1.1030% | 1,014,540 | `470899bbef97c32a100528d63e863e1b1036df9501df6ef8140e7ec13b7365b4` |

Per-column dtypes, raw null counts, sentinel counts, monthly counts, and differences from Base are stored in `outputs/profiling/`.

## Data preparation and leakage controls

- Documented `-1` sentinels are converted to missing values before preprocessing.
- Median numeric imputation and most-frequent categorical imputation are fitted on training rows only.
- Categorical fields use one-hot encoding with unknown-category handling.
- Class imbalance is handled through estimator class weights; no pre-split resampling is used.
- Target, month, protected diagnostics, `device_fraud_count`, `x1`, and `x2` are excluded from predictors.
- Sampling is stratified by month and label with fixed seed 42.
- Thresholds and model winners use validation data only.
- Full checklist: `docs/LEAKAGE_CHECKLIST.md` and `outputs/leakage_checklist.json`.

## Champion results on held-out months 6-7

| Dataset | Selected model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | Logistic Regression | 0.7462 | 0.9478 | 0.1374 | 0.5142 | 0.2168 | 0.8912 | 0.1695 | 0.0460 |
| Variant I | Logistic Regression | 0.7392 | 0.9495 | 0.1209 | 0.4145 | 0.1873 | 0.8680 | 0.1404 | 0.0429 |
| Variant II | Logistic Regression | 0.7390 | 0.9458 | 0.1234 | 0.4687 | 0.1954 | 0.8724 | 0.1598 | 0.0474 |
| Variant III | Logistic Regression | 0.7755 | 0.9418 | 0.0919 | 0.4845 | 0.1545 | 0.8609 | 0.1132 | 0.0532 |
| Variant IV | Logistic Regression | 0.7641 | 0.9397 | 0.0789 | 0.4208 | 0.1329 | 0.8421 | 0.1087 | 0.0545 |
| Variant V | Logistic Regression | 0.7609 | 0.9434 | 0.0890 | 0.4499 | 0.1486 | 0.8600 | 0.0980 | 0.0511 |

## Commands

```bash
python -m ai4cyber.profiling
python -m ai4cyber.training
python -m ai4cyber.clustering
python -m ai4cyber.eda
pytest -q
ruff check .
```

## Evidence files

- `docs/DATASET_PROVENANCE.md`
- `docs/LEAKAGE_CHECKLIST.md`
- `outputs/profiling/profile_manifest.json`
- `outputs/profiling/dataset_comparison.csv`
- `artifacts/model_card.json`
- `outputs/leakage_checklist.json`
- `outputs/<dataset>/metrics.json`
- `outputs/<dataset>/model_comparison.csv`
- `data/processed/*_model_data.csv.gz` (local, Git-ignored)
