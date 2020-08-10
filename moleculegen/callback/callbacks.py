"""
The collection of callbacks.

Classes
-------
BatchMetricScorer
    Calculate and log metrics after batch sampling and forward computation.
EpochMetricScorer
    Calculate and log metrics at the end of every epoch.
ProgressBar
    Print progress bar every epoch of model training.
"""

__all__ = (
    'BatchMetricScorer',
    'EpochMetricScorer',
    'ProgressBar',
)


import collections
import datetime
import math
import statistics
import sys
import time
from typing import Deque, List, Optional, Sequence

import mxnet as mx

from .base import Callback
from ..base import Token
from ..data.vocabulary import SMILESVocabulary
from ..estimation.model import SMILESEncoderDecoder
from ..evaluation.metric import Metric
from ..generation.greedy_search import GreedySearch


class BatchMetricScorer(Callback):
    """Calculate and log metrics after batch sampling and forward computation. Average
    the results at the end of every epoch.
    Use for "predictions-outputs" based metrics, e.g. Perplexity.

    Parameters
    ----------
    metrics : sequence of mxnet.metric.EvalMetric
        The metrics to calculate during mini-batch sampling and average at the end
        of an epoch.
    """

    def __init__(self, metrics: Sequence[mx.metric.EvalMetric]):
        self.__metrics = metrics

    def on_epoch_begin(self, **fit_kwargs):
        """Initialize/Reset internal states of the metrics.
        """
        for metric in self.__metrics:
            metric.reset()

    def on_batch_end(self, **fit_kwargs):
        """Get labels and predictions to score metrics.

        Parameters
        ----------
        Expected named arguments:
            - predictions
            - outputs
        """
        predictions = fit_kwargs.get('predictions')
        outputs = fit_kwargs.get('outputs')

        for metric in self.__metrics:
            metric.update(labels=outputs, preds=predictions)

    def on_epoch_end(self, **fit_kwargs):
        """Calculate and log average scores.
        """
        sys.stdout.write('\nCalculating metrics...\t')
        for metric in self.__metrics:
            name, result = metric.get()
            sys.stdout.write(f'{name}: {result:.3f}  ')


class EpochMetricScorer(Callback):
    """Calculate and log metrics at the end of every epoch.
    Use for distribution-based metrics, e.g. RAC.

    Parameters
    ----------
    metrics : sequence of Metric
        The metrics to calculate at the end of an epoch on a set of generated
        compounds (e.g. RAC).
    predictor : GreedySearch
        A SMILES string predictor.
    vocabulary : SMILESVocabulary
        The vocabulary to encode-decode tokens.
    prefix : str, default: Token.BOS
        The prefix of a SMILES string to generate
    max_length : int, default: 80
        The maximum number of tokens to generate.
    n_predictions : int, default 10000
        The number of SMILES strings to generate.
    train_dataset : sequence of str, default None
        A dataset to compare the generated compounds with.
    """

    def __init__(
            self,
            metrics: Sequence[Metric],
            predictor: GreedySearch,
            vocabulary: SMILESVocabulary,
            prefix: str = Token.BOS,
            max_length: int = 80,
            n_predictions: int = 1000,
            train_dataset: Optional[Sequence[str]] = None,
    ):
        self.__metrics = metrics
        self.__predictor = predictor
        self.__predictor_kwargs = {
            'vocabulary': vocabulary,
            'prefix': prefix,
            'max_length': max_length,
        }
        self.__n_predictions = n_predictions
        self.__train_dataset = train_dataset

    def on_epoch_begin(self, **fit_kwargs):
        """Initialize/Reset internal states of the metrics.
        """
        for metric in self.__metrics:
            metric.reset()

    def on_epoch_end(self, **fit_kwargs):
        """Score and log every metric.

        Parameters
        ----------
        Expected named arguments:
            - model
        """
        sys.stdout.write('\nCalculating metrics...\t')

        model: SMILESEncoderDecoder = fit_kwargs.get('model')

        for metric in self.__metrics:

            for _ in range(self.__n_predictions):
                states = model.begin_state(batch_size=1)
                smiles = self.__predictor(
                    model=model,
                    states=states,
                    **self.__predictor_kwargs,
                )
                metric.update(predictions=[smiles], labels=self.__train_dataset)

            name, result = metric.get()
            sys.stdout.write(f'{name}: {result:.3f}  ')


class ProgressBar(Callback):
    """Print progress bar every epoch of model training.

    Default format:
    Epoch {epoch} [✓✗✗✗] Batch {batch_no}/{n_batches}, Loss {loss}, {time} sec/batch

    Parameters
    ----------
    length : int, default 30
        The length of a progress bar (without logs).
    left_border : str, default '['
        The left border of a progress bar.
    right_border : str, default ']'
        The right border of a progress bar.
    done_char : str, default '✓'
        The character inside a progress bar that indicates the jobs done.
    wait_char : str, default '✗'
        The character inside a progress bar that indicates the jobs waiting.
    """

    def __init__(
            self,
            length: int = 30,
            left_border: str = '[',
            right_border: str = ']',
            done_char: str = '\u2713',
            wait_char: str = '\u2717',
    ):
        # Constants.
        self.__length = length
        self.__left_border = left_border
        self.__right_border = right_border
        self.__done_char = done_char
        self.__wait_char = wait_char

        # Variables.
        self._bar: Optional[Deque[str]] = None  # Progress bar.

        self._n_batches: Optional[int] = None  # Number of batches.
        self._batches_per_char: Optional[int] = None  # Number of batches as one symbol.
        self._batch_digits: Optional[int] = None  # Number of digits.
        self._batch_start_time: Optional[float] = None  # Batch countdown.
        self._batch_time: Optional[float] = None  # Batch execution time.

        self._epoch: Optional[int] = None  # Current epoch.
        self._epoch_start_time: Optional[float] = None  # Epoch countdown.
        self._epoch_time: Optional[float] = None  # Epoch execution time.

        self._loss_list: Optional[List[float]] = None  # List of losses.

    @property
    def format(self) -> str:
        """Return the format of a progress bar, in which parameters `epoch`, `bar`,
        `batch_no`, `loss`, and `batch_time` must be passed in order to print the bar.
        """
        return (
            'Epoch {epoch} '
            + self.__left_border + '{bar}' + self.__right_border + ' '
            + 'Batch {batch_no}/' + str(self._n_batches) + ', '
            + 'Loss {loss:.3f}, '
            + '{batch_time:.3f} sec/batch'
        )

    def _init_bar(self, length: int):
        """Initialize a progress bar `self._bar`.
        """
        self._bar: Deque[str] = collections.deque([self.__wait_char] * length)

    def on_epoch_begin(self, **fit_kwargs):
        """Initialize a progress bar and a loss list. Start countdown. Retrieve the
        number of epochs and batches.

        Parameters
        ----------
        Expected named arguments:
            - batch_sampler
            - n_epochs
            - epoch
        """
        self._epoch_start_time = time.time()

        epoch_digits = len(str(fit_kwargs.get('n_epochs')))
        self._epoch = f'{fit_kwargs.get("epoch"):>{epoch_digits}}'

        self._loss_list = []

        self._n_batches = len(fit_kwargs.get('batch_sampler'))
        self._batches_per_char = self._n_batches // self.__length or 1
        self._batch_digits = len(str(self._n_batches))

        if self._n_batches < self.__length:
            bar_length = self._n_batches
        else:
            bar_length = self.__length
        self._init_bar(bar_length)

    def on_batch_begin(self, **fit_kwargs):
        """Start countdown.
        """
        self._batch_start_time = time.time()

    def on_batch_end(self, **fit_kwargs):
        """Retrieve loss. Print the bar. End countdown.

        Parameters
        ----------
        Expected named arguments:
            - loss
            - batch_no
        """
        loss: float = fit_kwargs.get('loss').mean().item()
        self._loss_list.append(loss)

        self._batch_time = time.time() - self._batch_start_time

        batch_no: int = fit_kwargs.get('batch_no')
        if (
                batch_no % self._batches_per_char == 0
                or batch_no == self._n_batches
        ):
            self._bar.appendleft(self.__done_char)
            self._bar.pop()
        batch_no_str = f'{batch_no:>{self._batch_digits}}'

        sys.stdout.write(
            '\r'
            + self.format.format(
                epoch=self._epoch,
                bar=''.join(self._bar),
                batch_no=batch_no_str,
                loss=loss,
                batch_time=self._batch_time,
            )
        )
        sys.stdout.flush()

    def on_epoch_end(self, **fit_kwargs):
        """End countdown. Write execution time and mean loss.
        """
        self._epoch_time = math.ceil(time.time() - self._epoch_start_time)

        sys.stdout.write('\n')
        sys.stdout.write(f'Time {datetime.timedelta(seconds=self._epoch_time)}, ')
        sys.stdout.write(
            f'Mean loss: {statistics.mean(self._loss_list):.3f} '
            f'(+/-{statistics.stdev(self._loss_list):.3f})'
        )
        sys.stdout.write('\n')
