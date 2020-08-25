"""
Build callback objects to run during model training.

Classes
-------
Callback
    Callback ABC.

BatchMetricScorer
    Calculate and log metrics after batch sampling and forward computation.
EarlyStopping
    Stop training when a monitored evaluation function has stopped improving.
EpochMetricScorer
    Calculate and log metrics at the end of every epoch.
ProgressBar
    Print progress bar every epoch of model training.
"""

__all__ = (
    'BatchMetricScorer',
    'Callback',
    'EarlyStopping',
    'EpochMetricScorer',
    'ProgressBar',
)


from .base import Callback
from .callbacks import (
    BatchMetricScorer,
    EarlyStopping,
    EpochMetricScorer,
    ProgressBar,
)
