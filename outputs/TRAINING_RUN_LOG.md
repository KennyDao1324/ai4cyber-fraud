# AI4Cyber ML Training Run Log

This file records the reproducible evidence from the verified real-data run. It is a factual run log,
not report prose. The team should interpret these results in its own words and follow the unit's
academic-integrity requirements.

## Run identity

| Item | Value |
|---|---|
| Project | Bank Account Opening Fraud Screening |
| Architecture | ML-only; no chatbot or generative model |
| Training completed | 25 October 2026, 00:14 ICT (UTC+7) |
| Final verification | 27 October 2026 ICT (UTC+7) |
| Production dataset | BAF Base |
| Selection rule | Highest validation PR-AUC; test data not used for model selection |
| Production model | Logistic regression |
| Random seed | 42 |

## Environment

| Component | Version |
|---|---:|
| Python | 3.12.6 |
| pandas | 2.3.3 |
| NumPy | 2.5.4 |
| scikit-learn | 1.9.1 |
| FastAPI | 0.162.0 |
| Matplotlib | 3.11.2 |

## Source-data verification

All three CSVs passed archive integrity, schema, row-count, label, month-coverage, and missing-value
checks. Each has 1,000,000 rows, 32 columns, months 0-7, 11,029 fraud rows, a 1.1029% fraud rate,
and zero missing cells.

| Dataset | SHA-256 |
|---|---|
| Base.csv | `7bf10a37ce07e72e14c1b09e5efee3d27261baff4facc7da767b0474dcf9b809` |
| Variant I.csv | `48c637a255d1fa4515da4286ccb99251412a6d905937ec7c108838ddfc1674bd` |
| Variant II.csv | `60bcd971cf28779abb183c3286990cff9a87275bd8b83d9f6054a396c95cc736` |

The Base hash matches the published source hash used by the project for provenance verification.

## Data preparation and evaluation design

- Months 0-5: training.
- Month 6: validation, model selection, and threshold freezing.
- Month 7: one-time held-out testing.
- Laptop-safe reproducible sample per dataset: 250,000 training rows, 50,000 validation rows, and
  50,000 test rows.
- Sampling was stratified by month and fraud label with seed 42.
- `customer_age`, `income`, and `employment_status` were excluded from predictive features and retained
  only for diagnostic analysis.
- `device_fraud_count` was excluded because it is constant.
- Numeric fields: median imputation and standardisation.
- Categorical fields: most-frequent imputation and one-hot encoding with unknown-category handling.
- Class imbalance: balanced class weights.
- Review threshold: selected on validation data to maximise recall while targeting at most 5% FPR.
- Primary model-selection metric: validation PR-AUC, because the positive class is rare.

## Algorithms evaluated

Four supervised classifiers were trained independently on each dataset:

1. Logistic regression
2. SGD logistic classifier
3. Random forest
4. Extra trees

MiniBatch K-Means was evaluated separately as the second ML method. It does not produce the fraud
decision and its clusters are not treated as fraud labels.

## Champion results on held-out month 7

| Dataset | Selected model | Threshold | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base | Logistic regression | 0.7901 | 0.9558 | 0.1474 | 0.4179 | 0.2179 | 0.8682 | 0.1495 | 0.0362 |
| Variant I | Logistic regression | 0.7728 | 0.9512 | 0.1339 | 0.4220 | 0.2033 | 0.8654 | 0.1412 | 0.0408 |
| Variant II | Logistic regression | 0.7885 | 0.9499 | 0.1365 | 0.4505 | 0.2095 | 0.8690 | 0.1621 | 0.0426 |

## All classifier results on held-out month 7

### Base

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.7901 | 0.1474 | 0.4179 | 0.2179 | 0.8682 | 0.1495 | 0.0362 |
| SGD logistic | 1.0000 | 0.0084 | 0.0176 | 0.0114 | 0.4579 | 0.0134 | 0.0312 |
| Random forest | 0.3052 | 0.1452 | 0.3284 | 0.2013 | 0.8550 | 0.1159 | 0.0289 |
| Extra trees | 0.5667 | 0.1308 | 0.3555 | 0.1912 | 0.8407 | 0.1181 | 0.0353 |

### Variant I

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.7728 | 0.1339 | 0.4220 | 0.2033 | 0.8654 | 0.1412 | 0.0408 |
| SGD logistic | 0.9996 | 0.0115 | 0.0502 | 0.0187 | 0.4854 | 0.0142 | 0.0647 |
| Random forest | 0.3007 | 0.1436 | 0.3555 | 0.2045 | 0.8581 | 0.1208 | 0.0317 |
| Extra trees | 0.5618 | 0.1306 | 0.3826 | 0.1947 | 0.8517 | 0.1304 | 0.0381 |

### Variant II

| Model | Threshold | Precision | Recall | F1 | ROC-AUC | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.7885 | 0.1365 | 0.4505 | 0.2095 | 0.8690 | 0.1621 | 0.0426 |
| SGD logistic | 0.9929 | 0.0191 | 0.2836 | 0.0359 | 0.5296 | 0.0186 | 0.2174 |
| Random forest | 0.3357 | 0.1327 | 0.3758 | 0.1962 | 0.8556 | 0.1262 | 0.0367 |
| Extra trees | 0.5988 | 0.1228 | 0.3989 | 0.1877 | 0.8464 | 0.1406 | 0.0426 |

## Base champion confusion matrix

|  | Predicted legitimate | Predicted fraud/review |
|---|---:|---:|
| Actual legitimate | 47,481 | 1,782 |
| Actual fraud | 429 | 308 |

## Base fairness diagnostic

Age was excluded from model inputs and used only to compare outcomes after prediction.

| Group | Rows | Precision | Recall | PR-AUC | FPR |
|---|---:|---:|---:|---:|---:|
| Age 50 or under | 48,474 | 0.1379 | 0.3982 | 0.1350 | 0.0343 |
| Age over 50 | 1,526 | 0.2421 | 0.5823 | 0.2874 | 0.0995 |

The observed FPR ratio (over-50 / 50-or-under) was 2.9050. This is a diagnostic warning about unequal
outcomes in the held-out sample, not evidence of fairness or production suitability.

## Model-wide feature importance

The leading Base logistic-regression feature groups by aggregated absolute coefficient magnitude were:

| Rank | Feature | Normalised importance |
|---:|---|---:|
| 1 | housing_status | 0.2848 |
| 2 | device_os | 0.1891 |
| 3 | payment_type | 0.1406 |
| 4 | source | 0.0407 |
| 5 | has_other_cards | 0.0399 |
| 6 | phone_home_valid | 0.0364 |
| 7 | prev_address_months_count | 0.0347 |
| 8 | email_is_free | 0.0336 |
| 9 | keep_alive_session | 0.0326 |
| 10 | name_email_similarity | 0.0264 |

These are model-wide associations, not causal explanations for fraud.

## Clustering results

MiniBatch K-Means was run on a 30,000-row Base training sample after preprocessing and dimensionality
reduction. Candidate `k` values were 2-6.

| k | Silhouette score |
|---:|---:|
| 2 | 0.1124 |
| 3 | 0.0968 |
| 4 | 0.0731 |
| 5 | 0.0996 |
| 6 | 0.0786 |

`k=2` was selected. Cluster 0 contained 20,235 rows with a 0.6474% fraud rate. Cluster 1 contained
9,765 rows with a 1.8433% fraud rate. These clusters are exploratory segments only.

## Commands used

```bash
python -m ai4cyber.training
python -m ai4cyber.clustering
python -m ai4cyber.eda
pytest -q
python -m uvicorn ai4cyber.api:app --host 127.0.0.1 --port 8010
```

## Verification status

- Ruff static checks: passed.
- Pytest: 5 passed.
- Production model loads successfully.
- `GET /`, `GET /health`, `GET /model`, and `POST /predict` are present.
- No chatbot or generative-model endpoint is present.

## Evidence files

- `artifacts/model_card.json`: consolidated model-selection record.
- `artifacts/production_model.joblib`: selected Base classifier and preprocessing pipeline.
- `outputs/base/metrics.json`: detailed Base metrics, fairness diagnostics, and feature importance.
- `outputs/variant_i/metrics.json`: detailed Variant I metrics.
- `outputs/variant_ii/metrics.json`: detailed Variant II metrics.
- `outputs/base/clustering_metrics.json`: clustering candidates and selected result.
- `outputs/*/model_comparison.csv`: report-ready metric tables.
- `outputs/*/*.png` and `outputs/base/eda/*.png`: report-ready figures.
- `data/processed/*_model_data.csv.gz`: exact sampled rows used by the experiments.

