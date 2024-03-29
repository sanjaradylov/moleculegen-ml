"""
The collection of callbacks.

Classes:
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
    'BatchMetricScorer',
    'EarlyStopping',
    'EpochMetricScorer',
    'Generator',
    'PhysChemDescriptorPlotter',
    'ProgressBar',
)


import collections
import statistics
import sys
import tempfile
import time
from typing import Deque, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
import mxnet as mx

from .base import Callback
from ..description.base import get_descriptors_df
from ..description.physicochemical import PHYSCHEM_DESCRIPTOR_MAP
from ..evaluation.metric import CompositeMetric, Metric
from ..generation.search import BaseSearch


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
        for metric in self.__metrics:
            name, result = metric.get()
            sys.stdout.write(f'{name}: {result:.3f}  ')
        sys.stdout.write('\n')


class EpochMetricScorer(Callback):
    """Calculate and log metrics at the end of every epoch.
    Use for distribution-based metrics, e.g. RAC.

    Parameters
    ----------
    metrics : moleculegen.evaluation.CompositeMetric
            or sequence of moleculegen.evaluation.Metric
        The metrics to calculate at the end of an epoch on a set of generated
        compounds (e.g. moleculegen.evaluation.RAC).
    predictor : moleculegen.generation.BaseSearch
        A SMILES string predictor.
    n_predictions : int, default=1000
        The number of SMILES strings to generate.
    train_dataset : sequence of str, default=None
        A dataset to compare the generated compounds with.
    """

    def __init__(
            self,
            metrics: Union[Sequence[Metric], CompositeMetric],
            predictor: BaseSearch,
            n_predictions: int = 1000,
            train_dataset: Optional[Sequence[str]] = None,
    ):
        if not isinstance(metrics, CompositeMetric):
            self._metrics = CompositeMetric(*metrics)
        else:
            self._metrics = metrics

        self._predictor = predictor
        self._n_predictions = n_predictions
        self._train_dataset = train_dataset

        # `KLDivergence` `NearestNeighborSimilarity`, and `InternalDiversity`
        # do not support evaluating one single prediction;
        # if it's present in metrics, we will first generate `self._n_predictions`
        # strings and evaluate, otherwise, we generate and evaluate one string at a
        # time to save space.
        # ??? Maybe create a separate `Metric` abstract base subclass for the metrics
        #     that cannot be accumulated?
        from ..evaluation import (
            KLDivergence, NearestNeighborSimilarity, InternalDiversity)

        no_acc_metrics = (KLDivergence, NearestNeighborSimilarity, InternalDiversity)

        for metric in self._metrics:
            if isinstance(metric, no_acc_metrics):
                self._update = self._update_multiple
                break
        else:
            self._update = self._update_one

    def _update_one(self):
        for _ in range(self._n_predictions):
            smiles = self._predictor()
            self._metrics.update(predictions=[smiles], labels=self._train_dataset)

    def _update_multiple(self):
        predictions = tuple(self._predictor() for _ in range(self._n_predictions))
        self._metrics.update(predictions=predictions, labels=self._train_dataset)

    def on_epoch_begin(self, **fit_kwargs):
        """Initialize/Reset internal states of the metrics.
        """
        self._metrics.reset()

    def on_epoch_end(self, **fit_kwargs):
        """Score and log every metric.

        Parameters
        ----------
        Expected named arguments:
            - model
        """
        self._update()
        results = ', '.join(
            f'{name}: {value:.3f}'
            for name, value in self._metrics.get()
        )
        sys.stdout.write(f'{results}\n')


class EarlyStopping(Callback):
    """Stop training when a monitored evaluation function (loss) has stopped improving.

    Parameters
    ----------
    min_delta : float, default=0.0
        An absolute change higher than `min_delta` will count as improvement.
    patience : int, default=0
        If no improvement takes place after `patience` epochs, stop training.
    restore_best_weights : bool, default=False
        Whether to restore the parameters from the epoch with the best `monitor` value.
    """

    def __init__(
            self,
            min_delta: float = 0.0,
            patience: int = 0,
            restore_best_weights: bool = False,
    ):
        self.__min_delta = min_delta
        self.__patience = patience
        self.__restore_best_weights = restore_best_weights

        if self.__restore_best_weights:
            self._path_to_weights = tempfile.gettempdir()

        self._best_epoch: int = 0
        self._best_loss: float = float('inf')
        self._patience_count: int = 0
        self._current_loss: Optional[float] = None
        self._n_instances: Optional[int] = None

    def _get_path_to_weights(self, epoch: int) -> str:
        """Return the path to the parameters at epoch #`epoch`.
        """
        return f'{self._path_to_weights}/epoch_{epoch}.params'

    def on_epoch_begin(self, **fit_kwargs):
        """Initialize the variables to store loss values.
        """
        self._current_loss = 0.0
        self._n_instances = 0

    def on_batch_end(self, **fit_kwargs):
        """Save the loss value on the current batch.

        Parameters
        ----------
        Expected named arguments:
            - loss
        """
        self._current_loss += fit_kwargs.get('loss').mean().item()
        self._n_instances += 1

    def on_epoch_end(self, **fit_kwargs):
        """Calculate the average loss and compare it with the previous results to make
        a decision on early stopping.

        Parameters
        ----------
        Expected named arguments:
            - epoch
            - model

        Raises
        ------
        KeyboardInterrupt
            If no improvement occurs during `self.__patience` epochs.
            See `on_keyboard_interrupt` implementation.
        """
        epoch = fit_kwargs.get('epoch')

        mean_loss = self._current_loss / self._n_instances

        if mean_loss + self.__min_delta > self._best_loss:
            self._patience_count += 1
        else:
            self._patience_count = 0

            if self.__restore_best_weights:
                model = fit_kwargs.get('model')
                filename = self._get_path_to_weights(epoch)
                with open(filename, mode='wb') as fh:
                    model.save_parameters(fh.name)

        if mean_loss < self._best_loss:
            self._best_loss = mean_loss
            self._best_epoch = epoch

        if self._patience_count == self.__patience:
            raise KeyboardInterrupt

    def on_keyboard_interrupt(self, **fit_kwargs):
        """Restore the best weights if the corresponding formal parameter is specified.

        Parameters
        ----------
        Expected named arguments:
            - epoch
            - model
        """
        epoch = fit_kwargs.get('epoch')

        if (
                self.__restore_best_weights
                and self._best_epoch != 0
                and self._best_epoch != epoch
        ):
            model = fit_kwargs.get('model')
            model.load_parameters(self._get_path_to_weights(self._best_epoch))


class ProgressBar(Callback):
    """Print progress bar every epoch of model training.

    Default format:
    Epoch {epoch} [✓✗✗✗] Batch {batch_no}/{n_batches}, Loss {loss}, {time} sec/batch

    Parameters
    ----------
    length : int, default=30
        The length of a progress bar (without logs).
    left_border : str, default='['
        The left border of a progress bar.
    right_border : str, default=']'
        The right border of a progress bar.
    done_char : str, default='✓'
        The character inside a progress bar that indicates the jobs done.
    wait_char : str, default='✗'
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
            + 'Loss {loss_mean:.3f} (+/-{loss_std:.3f}), '
            + '{batch_time:.3f} sec/batch'
        )

    def _init_bar(self, length: int):
        """Initialize a progress bar `self._bar`.
        """
        self._bar: Deque[str] = collections.deque([self.__wait_char] * length)

    def on_train_begin(self, **fit_kwargs):
        """Compute and save the number of batches.

        Parameters
        ----------
        Expected named arguments:
            - batch_sampler
        """
        self._n_batches = len(fit_kwargs.get('batch_sampler'))

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
        epoch_digits = len(str(fit_kwargs.get('n_epochs')))
        self._epoch = f'{fit_kwargs.get("epoch"):>{epoch_digits}}'

        self._loss_list = []

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
        loss: mx.np.ndarray = fit_kwargs.get('loss')
        loss_mean: float = loss.mean().item()
        loss_std: float = loss.std().item()
        self._loss_list.append(loss_mean)

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
                loss_mean=loss_mean,
                loss_std=loss_std,
                batch_time=self._batch_time,
            )
        )
        sys.stdout.flush()

    def on_epoch_end(self, **fit_kwargs):
        """End countdown. Write execution time and mean loss.
        """
        loss_mean = statistics.mean(self._loss_list)
        loss_std = statistics.stdev(self._loss_list)

        batch_no = f'{self._n_batches:>{self._batch_digits}}'

        sys.stdout.write(
            '\r'
            + self.format.format(
                epoch=self._epoch,
                bar=''.join(self._bar),
                batch_no=batch_no,
                loss_mean=loss_mean,
                loss_std=loss_std,
                batch_time=self._batch_time,
            )
            + '\n'
        )

    def on_keyboard_interrupt(self, **fit_kwargs):
        sys.stdout.write('\n')


class Generator(Callback):
    """Generate, save, and optionally evaluate new compounds.

    Parameters
    ----------
    filename : str
        The prefix of a file `{filename}_epoch_{epoch}.smi` to save predictions.
    predictor : moleculegen.generation.BaseSearch
        A SMILES string predictor.
    n_predictions : int, default=1_000
        The number of compounds to generate.
    epoch : int, default=None
        Generate every `epoch` epoch.
        If None, generate only after full training.
    metric : moleculegen.evaluation.Metric or moleculegen.evaluation.CompositeMetric,
             default=None
    on_interrupt : bool, default=False
        Generate on keyboard interrupt
        (e.g. manual keyboard interrupt or early stopping).
    kwargs : dict, str -> any, default=None
        Additional key-word arguments for `metric`.
    """

    def __init__(
            self,
            filename: str,
            predictor: BaseSearch,
            n_predictions: int = 1_000,
            metric: Union[Metric, CompositeMetric, None] = None,
            epoch: Optional[int] = None,
            on_interrupt: bool = False,
            **kwargs,
    ):
        self._predictor = predictor
        self._n_predictions = n_predictions
        self._filename = filename
        self._metric = metric
        self._epoch = epoch
        self._on_interrupt = on_interrupt
        self._kwargs = kwargs

    def _generate_and_save(self, epoch: Optional[int] = None):
        """Generate and save predictions. If `metric` is specified, evaluate the
        predictions.

        Parameters
        ----------
        epoch : int, default=None
        """
        # If a metric is specified, generate and save SMILES strings `n_predictions`
        # times and evaluate them;
        if self._metric is not None:
            def call():
                predictions = tuple(self._predictor() for _ in range(self._n_predictions))
                self._metric.update(
                    predictions=predictions, labels=self._kwargs.get('train_dataset'))
                for smiles in predictions:
                    fh.write(f'{smiles}\n')

                if isinstance(self._metric, CompositeMetric):
                    results = ', '.join(
                        f'{name}: {value:.3f}'
                        for name, value in self._metric.get()
                    )
                else:
                    name, value = self._metric.get()
                    results = f'{name}: {value:.3f}'
                sys.stdout.write(f'{results}\n')
        # otherwise, generate and save strings `n_predictions` times.
        else:
            def call():
                for _ in range(self._n_predictions):
                    fh.write(f'{self._predictor()}\n')

        if epoch is not None:
            filename = f'{self._filename}_epoch_{epoch}.smi'
        else:
            filename = f'{self._filename}.smi'

        # Run the main function.
        with open(filename, 'w') as fh:
            sys.stdout.write(f'Saving generated compounds to {filename}.\n')
            call()

    def on_epoch_begin(self, **fit_kwargs):
        """If a metric is specified, reset its internal state.

        Parameters
        ----------
        Expected named arguments:
            - n_epochs
        """
        if self._metric is not None:
            self._metric.reset()

        n_epochs = fit_kwargs.get('n_epochs')
        if self._epoch is None:
            self._epoch = n_epochs

    def on_epoch_end(self, **fit_kwargs):
        """Launch generation process every specified epoch.

        Parameters
        ----------
        Expected named arguments:
            - epoch
        """
        epoch = fit_kwargs.get('epoch')

        if epoch % self._epoch == 0:
            self._generate_and_save(epoch)

    def on_keyboard_interrupt(self, **fit_kwargs):
        """Optionally launch generation process on (keyboard) interrupt.

        Parameters
        ----------
        Expected named arguments:
            - epoch
        """
        epoch = fit_kwargs.get('epoch')

        if self._on_interrupt:
            self._generate_and_save(epoch)


class PhysChemDescriptorPlotter(Callback):
    """Project physicochemical descriptors of training and validation data into 2D using
    sklearn-compatible transformers (e.g. `TSNE` or `PCA`) and plot every chosen epoch.

    Parameters
    ----------
    transformer
        A dimensionality reduction method supporting `fit_transform` and/or `transform`
        methods.
    train_data : list of str
        A training SMILES data.
    image_file_prefix : str
        The prefix of saved images (e.g. if prefix is 'descriptors', then files will have
        names 'descriptors_epoch_1.png', ..., 'descriptors_epoch_N.png').
    epoch : int, default=None
        If None, plot only after the full training process.
        If int, plot every `epoch`th epoch.
    predictor : moleculegen.generation.BaseSearch, default=None
        If not None, generate a validation SMILES data of size equalling to the training
        data from this predictor.
        Pass either `predictor` or `valid_data_file_prefix`.
    valid_data_file_prefix : str, default=None
        If not None, load the validation data from files with names
        '{valid_data_file_prefix}_epoch_{epoch}.smi'.
        Pass either `predictor` or `valid_data_file_prefix`.
    """

    def __init__(
            self,
            transformer,
            train_data: List[str],
            image_file_prefix: str,
            epoch: Optional[int] = None,
            predictor: Optional[BaseSearch] = None,
            valid_data_file_prefix: Optional[str] = None,
    ):
        if predictor is None and valid_data_file_prefix is None:
            raise ValueError('pass either `predictor` or `valid_data_file_prefix`')

        self._transformer = transformer
        self._predictor = predictor
        self._valid_data_file_prefix = valid_data_file_prefix
        self._image_file_prefix = image_file_prefix
        self._epoch = epoch

        self._train_data = train_data

    def on_train_begin(self, **fit_kwargs):
        """Generate physicochemical descriptors for training data and apply manifold
        transformation.
        """
        self._train_data = get_descriptors_df(self._train_data, PHYSCHEM_DESCRIPTOR_MAP)
        self._train_data = self._transformer.fit_transform(self._train_data)

    def on_epoch_begin(self, **fit_kwargs):
        n_epochs = fit_kwargs.get('n_epochs')

        if self._epoch is None:
            self._epoch = n_epochs

    def on_epoch_end(self, **fit_kwargs):
        """Load or generate a validation SMILES data set, generate descriptors,
        transform the descriptors into 2D, and save the scatter plot of train vs. valid
        descriptors projection.

        Parameters
        ----------
        Expected named arguments:
            - epoch
        """

        def get_valid_data_desc():
            if hasattr(self._transformer, 'transform'):
                return self._transformer.transform(valid_data_desc)
            return self._transformer.fit_transform(valid_data_desc)

        def plot():
            plt.figure(figsize=(10, 10))
            plt.title(f'Phys.Chem Descriptors ({self._transformer.__class__.__name__})')
            plt.scatter(
                self._train_data[:, 0], self._train_data[:, 1],
                label='Training', c='m', edgecolors='k', alpha=0.5,
            )
            plt.scatter(
                valid_data_t[:, 0], valid_data_t[:, 1],
                label='Generated', c='g', edgecolors='k', alpha=0.85,
            )
            plt.legend()
            plt.savefig(f'{self._image_file_prefix}_epoch_{epoch}.png')

        epoch = fit_kwargs.get('epoch')
        if epoch % self._epoch == 0:

            if self._valid_data_file_prefix is not None:
                with open(f'{self._valid_data_file_prefix}_epoch_{epoch}.smi') as fh:
                    valid_data_desc = get_descriptors_df(
                        compounds=fh.readlines(),
                        function_map=PHYSCHEM_DESCRIPTOR_MAP,
                        invalid='skip',
                    )
            else:
                valid_data_desc = (
                    self._predictor() for _ in range(len(self._train_data))
                )
                valid_data_desc = get_descriptors_df(
                    compounds=valid_data_desc,
                    function_map=PHYSCHEM_DESCRIPTOR_MAP,
                    invalid='skip',
                )

            valid_data_t = get_valid_data_desc()
            plot()
