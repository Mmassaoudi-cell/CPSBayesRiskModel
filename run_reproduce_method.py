from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from src.cps_bayes_risk import CPSBayesRiskConfig, CPSBayesRiskModel
from src.data_loading import load_power_dataset, paper2_dataset_specs, stratified_holdout


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> float:
    confidence = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins, endpoint=False), np.linspace(0.1, 1, bins)):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray) -> float:
    encoded = label_binarize(y_true, classes=classes)
    if encoded.shape[1] == 1:
        return float(brier_score_loss(y_true, proba[:, 1] if proba.shape[1] > 1 else proba.ravel()))
    return float(np.mean(np.sum((encoded - proba) ** 2, axis=1)))


def evaluate_method(model: CPSBayesRiskModel, X_test, y_test, task: str) -> dict[str, float]:
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)
    risk = model.predict_risk(X_test)
    y_arr = np.asarray(y_test)
    classes = np.asarray(sorted(np.unique(y_arr)))
    row = {
        "accuracy": float(accuracy_score(y_arr, pred)),
        "precision_macro": float(precision_score(y_arr, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_arr, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_arr, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_arr, pred, average="weighted", zero_division=0)),
        "brier": multiclass_brier(y_arr, proba, classes),
        "ece": expected_calibration_error(y_arr, proba),
        "mean_risk": float(risk.mean()),
        "risk_std": float(risk.std()),
    }
    if task == "binary":
        row["auc"] = float(roc_auc_score(y_arr, proba[:, 1]))
        row["false_positive_rate"] = float(((pred == 1) & (y_arr == 0)).sum() / max((y_arr == 0).sum(), 1))
    else:
        row["auc"] = float(roc_auc_score(y_arr, proba, multi_class="ovr", labels=classes))
        row["false_positive_rate"] = float(((pred != 0) & (y_arr == 0)).sum() / max((y_arr == 0).sum(), 1))
        attack_truth = (y_arr != 0).astype(int)
        row["risk_auc"] = float(roc_auc_score(attack_truth, risk))
        top_k = max(1, int(0.10 * len(risk)))
        top_idx = np.argsort(-risk)[:top_k]
        row["top10_attack_capture"] = float(attack_truth[top_idx].mean())
        row["risk_gap"] = float(risk[attack_truth == 1].mean() - risk[attack_truth == 0].mean())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the CPSBayesRisk method only.")
    parser.add_argument("--data-root", required=True, help="Path to the local power-system dataset folder.")
    parser.add_argument("--task", choices=["all", "binary", "multiclass"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="method_results.json", help="JSON file for proposed-method metrics.")
    args = parser.parse_args()

    selected = [s for s in paper2_dataset_specs(args.data_root) if args.task in ("all", s.task)]
    if not selected:
        raise ValueError(f"No task selected for {args.task}")

    results = {}
    for spec in selected:
        print(f"Loading {spec.task} power-system dataset")
        X, y = load_power_dataset(spec)
        X_train, X_test, y_train, y_test = stratified_holdout(X, y, seed=args.seed, test_size=0.25)
        model = CPSBayesRiskModel(CPSBayesRiskConfig(random_state=args.seed))
        print(f"Training CPSBayesRiskModel for {spec.task}")
        model.fit(X_train, y_train)
        results[spec.task] = evaluate_method(model, X_test, y_test, spec.task)
        print(f"{spec.task}: macro_f1={results[spec.task]['macro_f1']:.4f}; ece={results[spec.task]['ece']:.4f}")

    output = Path(args.output)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved metrics to {output}")


if __name__ == "__main__":
    main()
