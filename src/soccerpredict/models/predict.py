"""Prediction helpers for a trained model."""

from __future__ import annotations

from typing import TYPE_CHECKING

from soccerpredict.utils.logging import get_logger

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

log = get_logger(__name__)


def predict_match_outcome(model: "Pipeline", X: "pd.DataFrame") -> "pd.DataFrame":
    """Return a DataFrame with one probability column per class.

    Columns are named after the class labels stored on the fitted model
    (e.g. ``home_win``, ``draw``, ``away_win``) and rows align 1:1 with
    the input ``X``.
    """
    import pandas as pd

    proba = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if not classes:
        classes = [f"class_{i}" for i in range(proba.shape[1])]

    out = pd.DataFrame(proba, columns=classes, index=X.index)
    out["pred"] = out[classes].idxmax(axis=1)
    log.info("Predicted {} matches", len(out))
    return out
