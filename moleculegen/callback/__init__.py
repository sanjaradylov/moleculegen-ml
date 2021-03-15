"""
Build callback objects to run during model training.

Classes:
    Callback:
        Callback ABC.
    CallbackList:
        Perform callbacks sequentially and log output messages.

    BatchMetricScorer:
        Calculate and log metrics after batch sampling and forward computation.
    EarlyStopping:
        Stop training when a monitored evaluation function has stopped improving.
    EpochMetricScorer:
        Calculate and log metrics at the end of every epoch.
    Generator:
        Generate, save, and optionally evaluate new compounds.
    PhysChemDescriptorPlotter:
        Project physicochemical descriptors of training and validation data into 2D using
        sklearn-compatible transformers (e.g. `TSNE` or `PCA`) and plot every chosen
        epoch.
    ProgressBar:
        Print progress bar every epoch of model training.
"""

__all__ = (
    # Callback API
    'Callback',
    'CallbackList',

    # Callbacks
    'BatchMetricScorer',
    'EarlyStopping',
    'EpochMetricScorer',
    'Generator',
    'PhysChemDescriptorPlotter',
    'ProgressBar',
)


from .base import (
    Callback,
    CallbackList,
)
from .callbacks import (
    BatchMetricScorer,
    EarlyStopping,
    EpochMetricScorer,
    Generator,
    PhysChemDescriptorPlotter,
    ProgressBar,
)
