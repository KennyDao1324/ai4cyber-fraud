# AI4Cyber: Bank Account Opening Fraud Screening

This repository contains the machine-learning implementation for detecting potentially fraudulent bank
account applications. It trains and compares four classifiers on all six datasets in the Bank Account
Fraud (BAF) dataset, evaluates the models using a time-based split, performs exploratory clustering, and
serves the selected model through a FastAPI application.

The system is ML-only. It does not contain a chatbot or an LLM. A model score is a screening signal, not
a verified probability of fraud, and the prototype does not approve, reject, or block real applications.

> **Academic integrity:** the supplied assignment brief says generative AI must not be used to generate
> assessment material. Confirm permitted use with the tutor, disclose assistance where required, and
> make sure every team member can explain and reproduce the submitted work.

## Quick start

```bash
git clone https://github.com/KennyDao1324/ai4cyber-fraud.git
cd ai4cyber-fraud
conda env create -f environment.yml
conda activate ai4cyber-fraud
pytest -q
python -m uvicorn ai4cyber.api:app --host 127.0.0.1 --port 8010
```

Open <http://127.0.0.1:8010/docs> to use the API. The small selected production model is stored in the
repository, so inference and the API work after the dependencies are installed. The raw datasets are only
needed when retraining.

## Project map

| Location | Purpose |
| --- | --- |
| `src/ai4cyber/data.py` | Dataset validation, temporal splits, and preprocessing |
| `src/ai4cyber/training.py` | Classifier training, evaluation, selection, and artifact export |
| `src/ai4cyber/clustering.py` | MiniBatch K-Means experiment |
| `src/ai4cyber/eda.py` | Exploratory data analysis charts |
| `src/ai4cyber/metrics.py` | Classification metrics and threshold selection |
| `src/ai4cyber/inference.py` | Command-line prediction and local feature effects |
| `src/ai4cyber/api.py` | FastAPI endpoints |
| `src/ai4cyber/decision.py` | Score-to-review-label rules |
| `configs/training.json` | Real experiment settings and sample limits |
| `configs/demo_training.json` | Small smoke-test settings |
| `examples/sample_application.json` | Example prediction input |
| `tests/` | Unit, API, and end-to-end tests |
| `notebooks/baf_profiling.ipynb` | Reproducible Base and Variants I-V profiling notebook |
| `docs/DATASET_PROVENANCE.md` | Source, version, licence, citation, and variant rationale |
| `docs/canvas_dataset_archive.json` | Hash and integrity evidence for the separately supplied course archive |
| `docs/LEAKAGE_CHECKLIST.md` | Auditable data-leakage controls |
| `outputs/TRAINING_RUN_LOG.md` | Consolidated experiment log for the team |
| `outputs/profiling/` | Per-dataset dtypes, missing codes, hashes, and Base comparison |
| `outputs/base/` | Base dataset metrics, CSV files, and charts |
| `outputs/variant_i/` | Variant I metrics, CSV files, and charts |
| `outputs/variant_ii/` | Variant II metrics, CSV files, and charts |
| `outputs/variant_iii/` | Variant III metrics, CSV files, and charts |
| `outputs/variant_iv/` | Variant IV metrics, CSV files, and charts |
| `outputs/variant_v/` | Variant V metrics, CSV files, and charts |
| `artifacts/model_card.json` | Model configuration, selection, and evaluation record |
| `artifacts/production_model.joblib` | Model loaded by inference and the API |
| `data/raw/` | Downloaded BAF CSV files; local only |
| `data/processed/` | Experiment rows exported during training; local only |

## Methods

The project keeps Base and Variants I-V separate and trains these classifiers on each one:

1. Logistic regression
2. SGD logistic classification
3. Random forest
4. Extra trees

MiniBatch K-Means is used as a second, unsupervised method. The number of clusters is selected using the
silhouette score.

The temporal evaluation uses:

- Months 0-4 for training
- Month 5 for validation, model selection, and threshold selection
- Months 6-7 as the held-out test set

This makes months 0-5 the complete development period and reserves months 6-7 for final testing.

The threshold is selected on validation data with a target false-positive rate of 5%. Test results are not
used for model selection. The main metrics are precision, recall, F1, ROC-AUC, PR-AUC, false-positive
rate, and the confusion matrix. PR-AUC and recall are especially important because fraud is rare.

`customer_age`, `income`, and `employment_status` are used only for diagnostics and are excluded from
model inputs. Age-group results are reported as a fairness diagnostic. The synthetic `x1` and `x2`
fields in Variants III and V are also excluded to avoid artificial shortcut learning.

## Current real-data results

Logistic regression was selected for all six datasets using validation PR-AUC.

| Dataset | Test PR-AUC | Recall | FPR | Threshold |
| --- | ---: | ---: | ---: | ---: |
| Base | 0.1695 | 0.5142 | 0.0460 | 0.7462 |
| Variant I | 0.1404 | 0.4145 | 0.0429 | 0.7392 |
| Variant II | 0.1598 | 0.4687 | 0.0474 | 0.7390 |
| Variant III | 0.1132 | 0.4845 | 0.0532 | 0.7755 |
| Variant IV | 0.1087 | 0.4208 | 0.0545 | 0.7641 |
| Variant V | 0.0980 | 0.4499 | 0.0511 | 0.7609 |

Detailed results are available in each `outputs/<dataset>/model_comparison.csv` file. The JSON files
contain complete metrics, confusion matrices, and fairness diagnostics. The PNG files contain the report
figures. `outputs/demo/` is for software testing only and must not be used as assignment evidence.

## Environment setup

### Option A: Conda

```bash
conda env create -f environment.yml
conda activate ai4cyber-fraud
```

To update an existing environment:

```bash
conda env update -f environment.yml --prune
conda activate ai4cyber-fraud
```

### Option B: Python virtual environment

Python 3.11, 3.12, or 3.13 is required.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,download]"
```

In PyCharm, open the repository folder and select the environment's Python interpreter under
**Settings > Project > Python Interpreter**. If imports are not detected, mark `src` as a Sources Root.

## Download the real datasets

The data comes from the Feedzai BAF project:

- <https://github.com/feedzai/bank-account-fraud>
- <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>

Configure Kaggle credentials, then run:

```bash
python scripts/download_baf.py
```

The downloader verifies Kaggle version 2 and the CC BY-NC-SA 4.0 dataset licence before accepting the
files. Full provenance, version history, citation, and licence separation are recorded in
`docs/DATASET_PROVENANCE.md`.

The following files must exist before real training:

```text
data/raw/Base.csv
data/raw/Variant I.csv
data/raw/Variant II.csv
data/raw/Variant III.csv
data/raw/Variant IV.csv
data/raw/Variant V.csv
```

The raw suite is approximately 1.36 GB uncompressed, and each CSV exceeds GitHub's normal 100 MB
per-file limit. Do not redistribute raw or processed data without following CC BY-NC-SA 4.0, and never
commit Kaggle credentials.

## Run a smoke test

The smoke test creates small BAF-shaped data to verify the software. It is not assignment evidence.

```bash
python scripts/make_demo_data.py
python -m ai4cyber.training \
  --data-dir data/demo \
  --config configs/demo_training.json \
  --artifacts artifacts/demo \
  --outputs outputs/demo \
  --processed data/demo_processed
python -m ai4cyber.clustering \
  --dataset "data/demo/Base.csv" \
  --config configs/demo_training.json \
  --artifacts artifacts/demo/base \
  --outputs outputs/demo/base
```

## Train the real experiments

```bash
python -m ai4cyber.profiling
python -m ai4cyber.eda
python -m ai4cyber.training
python -m ai4cyber.clustering
python scripts/build_training_log.py
```

The laptop-safe configuration samples up to 250,000 training rows and 50,000 validation and test rows
using fixed seeds and class/month stratification. These limits are recorded in `configs/training.json`.
Set the three `max_*_rows` values to `null` only on a machine that can process all six million rows.

Training creates:

- `artifacts/production_model.joblib`: selected Base model used by the API
- `artifacts/model_card.json`: experiment configuration and selected-model results
- `artifacts/<dataset>/`: every trained classifier and the dataset champion
- `outputs/<dataset>/`: metrics, comparisons, and report figures
- `data/processed/*_model_data.csv.gz`: exact rows used, with a `_split` column

The full classifier collection and processed data are reproducible local files and are not committed.

## Make a prediction

```bash
python -m ai4cyber.inference --input examples/sample_application.json
```

The output contains the model score, frozen threshold, deterministic review label, and five local
one-field-ablation associations. These associations are not causal explanations.

## Run the API

```bash
python -m uvicorn ai4cyber.api:app --host 127.0.0.1 --port 8010
```

Available endpoints:

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Service information |
| `GET` | `/health` | Health and model-load check |
| `GET` | `/model` | Model metadata |
| `POST` | `/predict` | Score one application |

Interactive documentation is at <http://127.0.0.1:8010/docs>. Stop the server with `Control+C`.

## Tests and style checks

```bash
pytest -q
ruff check .
```

The end-to-end test generates temporary data, trains the pipeline, exports processed rows, reloads the
production artifact, and performs a prediction.

## Files stored on GitHub

GitHub contains the source code, tests, configurations, documentation, real experiment metrics and plots,
the model card, and the small selected production model. The following remain local and are regenerated
as needed:

- `.venv/`, `.idea/`, caches, and operating-system files
- `data/raw/`, `data/processed/`, and generated demo data
- Full random-forest and extra-trees artifacts
- Demo outputs and demo model artifacts

This keeps the repository small and avoids uploading licensed data or machine-specific files.

## Team Git workflow

Before starting work:

```bash
git switch main
git pull
```

Create a branch for the task:

```bash
git switch -c feature/short-description
```

After making and testing changes:

```bash
git status
git add .
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Open a pull request on GitHub and ask another team member to review it. Do not commit tokens, raw data,
virtual environments, or large generated artifacts.

## Troubleshooting

- `No module named uvicorn`: activate the correct environment and run
  `python -m pip install -e ".[dev,download]"`.
- `No module named ai4cyber`: install the project in editable mode with the same command, then run commands
  from the repository root.
- Missing `data/raw/*.csv`: configure Kaggle and run `python scripts/download_baf.py`.
- Missing `artifacts/production_model.joblib`: pull the latest `main` branch or run the real training
  command. For demo artifacts, set `AI4CYBER_MODEL_PATH=artifacts/demo/production_model.joblib`.
- Port 8010 already in use: select another port, for example `--port 8011`.
- GitHub rejects a large file: verify it is excluded by `.gitignore`; do not force-add raw datasets.

