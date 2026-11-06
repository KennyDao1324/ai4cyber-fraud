# AI4Cyber ML Training Run Log

This log is generated from the model card and dataset profile manifest.

## Run identity

| Item | Value |
| --- | --- |
| Log generated | 2026-11-07T00:09:11+07:00 |
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

| Dataset | Rows | Columns | Fraud rate | Effective missing codes | SHA-256 |
| --- | ---: | ---: | ---: | ---: | --- |
| Base | 1,000,000 | 32 | 1.1029% | 1,715,706 | `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` |
| Variant I | 1,000,000 | 32 | 1.1029% | 1,705,862 | `48c637a255d1fa4515da4286ccb99251412a6d905937ec7c108838ddfc1674bd` |
| Variant II | 1,000,000 | 32 | 1.1029% | 1,756,572 | `60bcd971cf28779abb183c3286990cff9a87275bd8b83d9f6054a396c95cc736` |
| Variant III | 1,000,000 | 34 | 1.1030% | 1,767,032 | `64693e549cff7ef802cf043fc7c937839e5929966e33a0852953b3f4c4321339` |
| Variant IV | 1,000,000 | 32 | 1.1030% | 1,766,770 | `c6cddebbad34fa262a278deeb7985e7b669a64308d26e480d8eb85bfd08dac75` |
| Variant V | 1,000,000 | 34 | 1.1030% | 1,765,611 | `470899bbef97c32a100528d63e863e1b1036df9501df6ef8140e7ec13b7365b4` |

Per-column dtypes, raw null counts, sentinel counts, monthly counts, and differences from Base are stored in `outputs/profiling/`.

## Data preparation and leakage controls

- Every negative `intended_balcon_amount` value is converted to missing before preprocessing.
- Exact `-1` sentinels are converted for the other documented fields.
- Median numeric imputation and most-frequent categorical imputation are fitted on training rows only.
- Categorical fields use one-hot encoding with unknown-category handling.
- Class imbalance is handled through estimator class weights; no pre-split resampling is used.
- Target, month, protected diagnostics, `device_fraud_count`, `x1`, and `x2` are excluded from predictors.
- Sampling is stratified by month and label with fixed seed 42.
- Thresholds and model winners use validation data only.
- The age fairness diagnostic compares customers under 50 with customers aged 50 or over.
- Full checklist: `docs/LEAKAGE_CHECKLIST.md` and `outputs/leakage_checklist.json`.

## Champion results on held-out months 6-7

| Dataset | Selected model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | Logistic Regression | 0.7494 | 0.9489 | 0.1395 | 0.5114 | 0.2192 | 0.8916 | 0.1711 | 0.0449 |
| Variant I | Logistic Regression | 0.7405 | 0.9500 | 0.1220 | 0.4131 | 0.1884 | 0.8678 | 0.1394 | 0.0423 |
| Variant II | Logistic Regression | 0.7397 | 0.9459 | 0.1232 | 0.4658 | 0.1948 | 0.8722 | 0.1610 | 0.0472 |
| Variant III | Logistic Regression | 0.7755 | 0.9418 | 0.0931 | 0.4918 | 0.1565 | 0.8606 | 0.1130 | 0.0532 |
| Variant IV | Logistic Regression | 0.7645 | 0.9401 | 0.0797 | 0.4226 | 0.1341 | 0.8421 | 0.1088 | 0.0542 |
| Variant V | Logistic Regression | 0.7640 | 0.9443 | 0.0896 | 0.4444 | 0.1491 | 0.8600 | 0.0982 | 0.0502 |

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
