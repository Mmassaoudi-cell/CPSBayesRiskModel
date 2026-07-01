from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class CPSBayesRiskConfig:
    random_state: int = 42
    n_estimators: int = 180
    max_depth: int = 14
    min_samples_leaf: int = 2
    calibration_method: str = "isotonic"
    calibration_cv: int = 3
    class_risk_weights: dict[int, float] = field(default_factory=lambda: {0: 0.0, 1: 0.90, 2: 0.75})
    default_attack_weight: float = 0.65


def add_cps_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add relay, phasor, and Snort-derived CPS summary features."""
    out = X.copy()
    out.columns = [str(c).strip() for c in out.columns]
    out = out.replace([np.inf, -np.inf], np.nan)
    numeric = out.select_dtypes(include=[np.number]).columns

    relay_ids = sorted({c.split("-")[0] for c in numeric if "-" in c})
    for relay in relay_ids:
        relay_cols = [c for c in numeric if c.startswith(relay + "-")]
        if relay_cols:
            vals = out[relay_cols].astype(float)
            out[f"{relay}_mean"] = vals.mean(axis=1)
            out[f"{relay}_std"] = vals.std(axis=1)
            out[f"{relay}_max_abs"] = vals.abs().max(axis=1)

    magnitude_cols = [c for c in numeric if ":V" in c or ":I" in c]
    angle_cols = [c for c in numeric if ":VH" in c or ":IH" in c or ":PA" in c]
    snort_cols = [c for c in numeric if c.lower().startswith("snort")]

    if magnitude_cols:
        mag = out[magnitude_cols].astype(float)
        out["cps_magnitude_mean"] = mag.mean(axis=1)
        out["cps_magnitude_std"] = mag.std(axis=1)
        out["cps_magnitude_range"] = mag.max(axis=1) - mag.min(axis=1)
    if angle_cols:
        ang = out[angle_cols].astype(float)
        out["cps_angle_std"] = ang.std(axis=1)
        out["cps_angle_range"] = ang.max(axis=1) - ang.min(axis=1)
    if snort_cols:
        snort = out[snort_cols].astype(float)
        out["snort_sum"] = snort.sum(axis=1)
        out["snort_any"] = (snort.sum(axis=1) > 0).astype(int)
        out["snort_max"] = snort.max(axis=1)
    return out


class CPSBayesRiskModel(BaseEstimator, ClassifierMixin):
    """CPS-aware calibrated IDS and Bayesian-style risk scorer.

    The model augments power-system telemetry with relay/phasor/Snort summary
    features, fits a calibrated Random Forest, then converts calibrated class
    probabilities into cyber-physical risk scores using class criticality
    weights. The fitted model exposes standard `predict` and `predict_proba`
    methods plus `predict_risk`.
    """

    def __init__(self, config: CPSBayesRiskConfig | None = None):
        self.config = config or CPSBayesRiskConfig()

    def fit(self, X: pd.DataFrame, y: Iterable):
        X_cps = add_cps_features(X)
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(np.asarray(list(y)))
        self.classes_ = self.label_encoder_.classes_

        numeric = list(X_cps.select_dtypes(include=[np.number]).columns)
        self.feature_columns_ = list(X_cps.columns)
        self.preprocessor_ = ColumnTransformer(
            [("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric)],
            remainder="drop",
        )
        base = RandomForestClassifier(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            class_weight="balanced_subsample",
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        calibrated = CalibratedClassifierCV(base, method=self.config.calibration_method, cv=self.config.calibration_cv)
        self.pipeline_ = Pipeline([("preprocess", self.preprocessor_), ("model", calibrated)])
        self.pipeline_.fit(X_cps, y_enc)
        return self

    def predict(self, X: pd.DataFrame):
        X_cps = add_cps_features(X)
        pred = self.pipeline_.predict(X_cps)
        return self.label_encoder_.inverse_transform(pred)

    def predict_proba(self, X: pd.DataFrame):
        X_cps = add_cps_features(X)
        return self.pipeline_.predict_proba(X_cps)

    def predict_risk(self, X: pd.DataFrame, prior_scale: float = 1.0) -> np.ndarray:
        proba = self.predict_proba(X)
        risk = np.zeros(proba.shape[0], dtype=float)
        for j, encoded_class in enumerate(self.pipeline_.named_steps["model"].classes_):
            original_class = self.label_encoder_.inverse_transform([encoded_class])[0]
            try:
                cls_key = int(original_class)
            except Exception:
                cls_key = encoded_class
            weight = self.config.class_risk_weights.get(cls_key, self.config.default_attack_weight)
            risk += proba[:, j] * weight * prior_scale
        return np.clip(risk, 0.0, 1.0)

    def attack_probability(self, X: pd.DataFrame, normal_label: int | str = 0) -> np.ndarray:
        proba = self.predict_proba(X)
        model_classes = self.pipeline_.named_steps["model"].classes_
        original_classes = self.label_encoder_.inverse_transform(model_classes)
        normal_mask = original_classes == normal_label
        if not normal_mask.any():
            return np.ones(proba.shape[0], dtype=float)
        normal_probability = proba[:, normal_mask].sum(axis=1)
        return 1.0 - normal_probability
