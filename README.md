# CPSBayesRisk


The method is a cyber-physical risk-aware intrusion detection model for power-system traces. It augments telemetry with relay, phasor, and Snort-derived CPS summary features, trains a calibrated Random Forest classifier, and converts calibrated class probabilities into Bayesian-style cyber-physical risk scores.

## Repository Contents

- `src/cps_bayes_risk.py`: implementation of the proposed CPSBayesRisk model.
- `src/data_loading.py`: loader for the local binary and multiclass power-system CSV files.
- `run_reproduce_method.py`: minimal runner for training and evaluating only the proposed method.
- `requirements.txt`: Python dependencies.

## Data

The runner expects the local power-system dataset folder that contains:

```text
Binary\TrainingDataBinary.csv
Multi\TrainingDataMulti.csv
```

For the original Paper 2 experiments, the data root was:

```text
C:\Users\MMASSAOUDI\Downloads\IECON 2 papers\A_Bayesian_Attack_Tree_Based_Approach_to_Assess_Cyber-Physical_Security_of_Power_System (1)
```

The external test CSV files in the local source package did not contain ground-truth markers, so the reproduction runner uses a stratified held-out split of the labeled training CSVs.

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## Reproduce The Method

Run the proposed method on both binary and multiclass tasks:

```bash
python run_reproduce_method.py --data-root "C:\Users\MMASSAOUDI\Downloads\IECON 2 papers\A_Bayesian_Attack_Tree_Based_Approach_to_Assess_Cyber-Physical_Security_of_Power_System (1)" --task all --seed 42 --output method_results.json
```

Run one task:

```bash
python run_reproduce_method.py --data-root "C:\Users\MMASSAOUDI\Downloads\IECON 2 papers\A_Bayesian_Attack_Tree_Based_Approach_to_Assess_Cyber-Physical_Security_of_Power_System (1)" --task multiclass --seed 42 --output method_results.json
```

The output is a JSON file containing only the proposed-method detection, calibration, and risk metrics. No benchmark comparison, paper table, or figure is generated.

## Method Summary

The model follows these steps:

1. Load the power-system CSV and use `marker` as the class label.
2. Add CPS-aware features from relay groups, voltage/current magnitudes, phasor/angle columns, and Snort alert columns.
3. Impute missing numeric values and standardize numeric features using training data only.
4. Train a class-balanced Random Forest.
5. Calibrate the classifier with isotonic calibration using cross-validation on the training split.
6. Convert calibrated class probabilities into risk scores using class criticality weights.

## Leakage Controls

The held-out test split is not used to fit feature scaling, imputation, calibration, or the classifier. Risk scores are computed from predicted probabilities only.

