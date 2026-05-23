"""Baseline model training.

Starts with a multinomial logistic regression over three classes:
home win / draw / away win. Real models (XGBoost, LightGBM, calibrated
ensembles) will live in sibling modules once the data pipeline is solid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from soccerpredict.utils.logging import get_logger

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from sklearn.pipeline import Pipeline

log = get_logger(__name__)

OUTCOME_CLASSES: tuple[str, ...] = ("home_win", "draw", "away_win")


@dataclass(slots=True)
class TrainResult:
    """Container for a trained model plus headline metrics."""

    model: "Pipeline"
    feature_names: list[str]
    accuracy: float
    log_loss: float


def train_baseline(
    X: "pd.DataFrame",
    y: "pd.Series | np.ndarray",
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    """Train a logistic-regression baseline and return the fitted model.

    Parameters
    ----------
    X
        Feature matrix (numeric columns only; encode categoricals upstream).
    y
        Target with values in :data:`OUTCOME_CLASSES`.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, log_loss
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipe: Pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    multi_class="multinomial",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)
    pred = pipe.predict(X_test)

    acc = float(accuracy_score(y_test, pred))
    ll = float(log_loss(y_test, proba, labels=list(pipe.classes_)))

    log.info("Baseline trained — accuracy={:.4f}, log_loss={:.4f}", acc, ll)

    return TrainResult(
        model=pipe,
        feature_names=list(X.columns),
        accuracy=acc,
        log_loss=ll,
    )
