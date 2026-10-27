# AI4Cyber: Bank Account Opening Fraud Screening

This repository is the code-focused Assignment 2 implementation for the project plan. It trains and
evaluates fraud-screening models on three BAF datasets, runs exploratory clustering as a second ML
method, exports the exact processed rows used, and exposes the selected model through a small API.
The architecture is ML-only: predictions, review labels, and explanations come from the trained model
and deterministic application logic.

> Academic-integrity warning: the supplied assignment brief says generative AI must **not** be used to
> generate assessment material. Obtain written tutor approval before submitting any generated code,
> disclose assistance as required, and make sure every team member can explain and reproduce the work.

## What is implemented

- Three separate datasets: `Base.csv`, `Variant I.csv`, and `Variant II.csv`.
- Four supervised classifiers per dataset: logistic regression, SGD logistic classification, random
  forest, and extra trees.
- A second ML method: MiniBatch K-Means, with `k` selected by silhouette score.
- Leak-resistant temporal evaluation: months 0-5 train, month 6 validation/threshold selection, and
  month 7 one-time test.
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix, and recall under a 5% FPR policy.
- Fairness diagnostics by age group. `customer_age`, `income`, and `employment_status` are retained
  only for diagnostics and excluded from model inputs.
- Dataset profiling, EDA charts, precision-recall curves, model comparisons, cluster plots, model
  artifacts, a model card, and compressed CSVs containing exactly the experiment rows.
- `GET /`, `POST /predict`, `GET /health`, and `GET /model` endpoints.

The model score is a ranking signal, not a verified fraud probability. The prototype never approves,
rejects, or blocks a real application.

## 1. Configure the project with Conda

From a terminal opened at this folder:

```bash
conda env create -f environment.yml
conda activate ai4cyber-fraud
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate ai4cyber-fraud
```

In PyCharm, open this folder, then choose **Settings > Project > Python Interpreter > Add
Interpreter > Conda Environment > Existing** and select the `ai4cyber-fraud` interpreter. Mark
`src` as a Sources Root if PyCharm does not detect the editable package automatically.

## 2. Obtain the datasets

The upstream source is the Feedzai BAF project and its Kaggle distribution:

- <https://github.com/feedzai/bank-account-fraud>
- <https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022>

Create a Kaggle API token using Kaggle's current instructions, configure the token locally, and run:

```bash
python scripts/download_baf.py
```

The required files are placed in `data/raw/`. Raw data is git-ignored. The upstream materials have
shown differing Creative Commons notices, so confirm the controlling license with the dataset owner or
tutor before redistributing raw or derived rows. Do not commit Kaggle credentials.

## 3. Run a quick smoke test before the full datasets

The following data is generated and BAF-shaped, but it is only for code testing and must not be used in
the assignment results:

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

## 4. Train the real experiments

```bash
python -m ai4cyber.eda
python -m ai4cyber.training
python -m ai4cyber.clustering
```

The laptop-safe default uses up to 250,000 training rows and 50,000 rows for each later split, sampled
with fixed seeds and class/month stratification. This limit is transparent in `configs/training.json`.
Set the three `max_*_rows` values to `null` only if the machine can train all three million-row variants.

Training produces:

- `outputs/TRAINING_RUN_LOG.md`: consolidated factual evidence for the team's report.
- `artifacts/production_model.joblib`: the best Base model selected by validation PR-AUC.
- `artifacts/model_card.json`: configuration, dataset profiles, selected models, and held-out results.
- `artifacts/<dataset>/`: all four model artifacts plus that dataset's champion.
- `outputs/<dataset>/`: JSON/CSV metrics and plots.
- `data/processed/*_model_data.csv.gz`: exactly the rows/columns used by the experiments, including
  `_split` so the evaluation can be reproduced.

The test month is reported only after model/threshold selection. It is never used to choose a model or
tune the 5% FPR threshold.

## 5. Predict one application

After training:

```bash
python -m ai4cyber.inference --input examples/sample_application.json
```

The response contains the score, frozen threshold, review label, and five local ablation associations.
Those associations are not causal explanations.

## 6. Run the ML API

```bash
python -m uvicorn ai4cyber.api:app --host 127.0.0.1 --port 8010
```

Open <http://127.0.0.1:8010/docs> for interactive endpoint documentation. `POST /predict` loads the
selected Base classifier and returns its score, frozen review threshold, deterministic review label, and
local feature-ablation associations. No chatbot, prompt, or generative model is used.

## 7. Run tests

```bash
pytest -q
```

The end-to-end test creates temporary fake data, trains the pipeline, exports processed rows, reloads the
production artifact, and predicts one case. No real dataset or model artifact is bundled in the source.

## Design rationale to use in the team's own explanation

PR-AUC, recall, and FPR matter more than accuracy because fraud is rare. The fixed temporal split tests
whether a model trained on earlier applications generalizes to later months. Four classifiers provide a
meaningful algorithm comparison, while clustering reveals applicant-pattern structure without pretending
that clusters are fraud labels. Each BAF variant stays separate because its bias scenario is part of the
experiment; merging variants would blur that evidence. The API remains ML-only so every result is
reproducible from the saved model and deterministic code.

