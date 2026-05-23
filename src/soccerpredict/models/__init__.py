"""Modeling layer: train, predict, evaluate."""

from soccerpredict.models.predict import predict_match_outcome
from soccerpredict.models.train import train_baseline

__all__ = ["train_baseline", "predict_match_outcome"]
