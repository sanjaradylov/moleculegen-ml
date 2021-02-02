"""
Additional metrics for model evaluation.

Classes
-------
CompositeMetric
    Calculate a series of metrics on the same predictions and labels.
Metric
    A metric ABC for the comparison of training/validation and generated compounds.
Novelty
    Calculate the rate of compounds not presented in the training set.
Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.
RAC
    Calculate the rate of unique acceptable compounds.
Uniqueness
    Calculate the rate of unique generated compounds.
Validity
    Calculate the rate of valid (theoretically realistic) molecules.
"""

__all__ = (
    'CompositeMetric',
    'KLDivergence',
    'Metric',
    'Novelty',
    'Perplexity',
    'RAC',
    'Uniqueness',
    'Validity',
)


import abc
import itertools
from numbers import Real
from typing import Any, Iterator, List, Optional, Sequence, Tuple

import mxnet as mx
import numpy as np
import scipy.stats as stats
from rdkit.Chem import MolFromSmiles

from ..description.base import check_compounds_valid
from ..description.common import get_descriptor_df_from_mol
from ..description.fingerprints import InternalTanimoto
from ..description.physicochemical import PHYSCHEM_DESCRIPTOR_MAP


class Metric(metaclass=abc.ABCMeta):
    """A metric ABC for the comparison of training/validation and generated compounds.

    Parameters
    ----------
    name : str, default None
        The name of a metric. Default is the name of the class.
    empty_value : any, default nan
        The instance indicating that no metric evaluation was performed.
    """

    def __init__(
            self,
            name: Optional[str] = None,
            empty_value: Any = float('nan'),
    ):
        self.__name = name or self.__class__.__name__
        self.__empty_value = empty_value

        self._result: Real = 0.0  # The accumulated result.
        self._n_samples: int = 0  # The number of samples evaluated.

    @property
    def name(self) -> str:
        return self.__name

    @property
    def empty_value(self) -> Any:
        return self.__empty_value

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}(name={self.__name!r}) at {hex(id(self))}>'

    def __eq__(self, other: 'Metric') -> bool:
        """Return True if the names and internal states of metrics are identical.
        """
        return (
            self.__name == other.__name
            and self._result == other._result
            and self._n_samples == other._n_samples
        )

    @abc.abstractmethod
    def _calculate(self, *, predictions, labels, **kwargs) -> Tuple[Real, int]:
        """Return the calculated result and the number of evaluated samples."""

    def get(self) -> Tuple[str, Real]:
        """Return the name and the accumulated average score.

        Returns
        -------
        tuple
            name : str
            value : any or numbers.Real
        """
        if self._n_samples == 0:
            value = self.__empty_value
        else:
            value = self._result / self._n_samples

        return self.__name, value

    def update(self, *, predictions, labels=None, **kwargs):
        """Update the internal evaluation result.
        """
        result, n_samples = self._calculate(
            predictions=predictions, labels=labels, **kwargs)

        self._result += result
        self._n_samples += n_samples

    def reset(self):
        """Reset the internal evaluation state.
        """
        self._result = 0.0
        self._n_samples = 0


class CompositeMetric:
    """Calculate a series of metrics on the same predictions and labels.
    Works with distribution-based metrics (i.e. `predictions` are a set of generated
    compounds and `labels` are a training/validation set).

    Parameters
    ----------
    metrics : tuple of Metric, default ()
        The metrics to evaluate.
        Instantiate a sequence of metrics during initialization
        (i.e. metrics = CompositeMetric(metric1, metric2, ..., metricN),
        or use `CompositeMetric.add`
        (i.e. metrics = CompositeMetric(); metrics.add(metric1, ..., metricN)).
    """

    def __init__(self, *metrics):
        self._metrics: List[Metric] = []
        if metrics:
            self.add(*metrics)

    def __getitem__(self, index: int) -> Metric:
        """Return the metric with index `index`.

        Parameters
        ----------
        index : int

        Returns
        -------
        metric : Metric

        Raises
        ------
        IndexError
            If the metric is not in the list.
        """
        try:
            return self._metrics[index]
        except IndexError as err:
            err.args = ('metrics index out of range',)
            raise

    def __setitem__(self, index: int, metric: Metric):
        """Insert `metric` in position `index`.

        Parameters
        ----------
        index : int
        metric : Metric

        Raises
        ------
        TypeError
            If `index` is out of range.
            If `metric` is not of type `Metric`.
        """
        if not isinstance(metric, Metric):
            raise TypeError(f'metric must be of type Metric, not {type(metric)}')

        try:
            self._metrics[index] = metric
        except IndexError as err:
            err.args = ('metrics assignment index out of range',)
            raise err

    def __delitem__(self, index: int):
        """Delete the metric in position `key`.

        Parameters
        ----------
        index : int

        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        try:
            del self._metrics[index]
        except IndexError as err:
            err.args = ('metrics assignment index out of range',)
            raise err

    def __len__(self) -> int:
        """Return the number of metrics in the sequence.

        Returns
        -------
        int
        """
        return len(self._metrics)

    def __contains__(self, metric: Metric) -> bool:
        """Check if `metric` is in the sequence.

        Parameters
        ----------
        metric : Metric

        Returns
        -------
        bool
        """
        return metric in self._metrics

    def __iter__(self) -> Iterator[Metric]:
        """Iterate over the metrics in the sequence.

        Yields
        ------
        metric : Metric
        """
        return iter(self._metrics)

    def insert(self, index: int, metric: Metric):
        """Insert `metric` in position `index`.

        Parameters
        ----------
        index : int
        metric : Metric

        Raises
        ------
        TypeError
            If `metric` is not of type `Metric`.
        """
        if index < len(self):
            self.__setitem__(index, metric)
        else:
            if not isinstance(metric, Metric):
                raise TypeError(f'metric must be of type Metric, not {type(metric)}')

            self._metrics.append(metric)

    def pop(self, index: int = -1) -> Metric:
        """Delete the metric in position `key`.

        Parameters
        ----------
        index : int, default len(self)-1

        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        try:
            return self._metrics.pop(index)
        except IndexError as err:
            err.args = ('pop index out of range',)
            raise

    def add(self, metric, *metrics):
        """Append metric(s) to the sequence.

        Parameters
        ----------
        metric : Metric
        metrics : tuple of Metric, default ()

        Raises
        ------
        TypeError
            If `metric` is not of type `Metric`.
        """
        metrics_chain = list(itertools.chain((metric,), metrics))

        for metric in metrics_chain:
            if not isinstance(metric, Metric):
                raise TypeError(f'metric must be of type Metric, not {type(metric)}')

        self._metrics.extend(metrics_chain)

    def get(self) -> List[Tuple[str, Real]]:
        """Retrieve the names and calculated values for every metric in the sequence.

        Returns
        -------
        list of (str, numbers.Real)
        """
        return [metric.get() for metric in self._metrics]

    def update(
            self,
            *,
            predictions: Sequence[str],
            labels: Optional[Sequence[str]] = None,
            **kwargs
    ):
        """Update the internal evaluation results of every metric in the sequence..

        Parameters
        ----------
        predictions : sequence of str
            The generated SMILES strings.
        labels : sequence of str, default None
            The labels (usually, training set) to compare the predictions with.
            Metrics like `Novelty` require this parameter, while for `RAC` this is
            optional.
        """
        for metric in self._metrics:
            metric.update(predictions=predictions, labels=labels, **kwargs)

    def reset(self):
        """Reset the internal evaluation results of every metric in the sequence.
        """
        for metric in self._metrics:
            metric.reset()


class Novelty(Metric):
    """Calculate the rate of compounds not presented in the training set.
    """

    def _calculate(
            self,
            *,
            predictions: Sequence[str],
            labels: Sequence[str],
            **kwargs,
    ) -> Tuple[Real, int]:
        """Return the number of compounds from `predictions` not presented in `labels`
        and the total number of compounds.

        Parameters
        ----------
        predictions : sequence of str
            The generated SMILES strings.
        labels : sequence of str
            The training set to compare with.

        Returns
        -------
        tuple
            result : int
            n_samples: int
        """
        result: Real = 0
        n_samples: int = 0
        labels = frozenset(labels)

        for n_samples, smiles in enumerate(predictions, start=1):
            result += int(smiles not in labels)

        return result, n_samples


class Uniqueness(Metric):
    """Calculate the rate of unique generated compounds.
    """

    def _calculate(
            self,
            *,
            predictions: Sequence[str],
            ignore: Optional[Sequence[str]] = None,
            **kwargs,
    ) -> Tuple[Real, int]:
        """Return the number of unique compounds and the total number of compounds.

        Parameters
        ----------
        predictions : sequence of str
            The generated SMILES strings.
        ignore

        Returns
        -------
        tuple
            result : int
            n_samples: int
        """
        return len(set(predictions)), len(predictions)


class Validity(Metric):
    """Calculate the rate of valid (theoretically realistic) molecules.
    """

    def _calculate(
            self,
            *,
            predictions: Sequence[str],
            ignore: Optional[Sequence[str]] = None,
            **kwargs,
    ) -> Tuple[Real, int]:
        """Return the number of valid compounds and the total number of compounds.

        Parameters
        ----------
        predictions : sequence of str
            The generated SMILES strings.
        ignore

        Returns
        -------
        tuple
            result : int
            n_samples: int
        """
        result: Real = 0
        n_samples: int = 0

        for n_samples, smiles in enumerate(predictions, start=1):
            molecule = MolFromSmiles(smiles)
            if molecule is not None:
                result += 1

        return result, n_samples


class RAC(Metric):
    """Calculate the rate of unique acceptable compounds.

    Parameters
    ----------
    name : str, default None
        The name of a metric. Default is the name of the class.
    empty_value : any, default nan
        The instance indicating that no metric evaluation was performed.
    count_unique : bool, default False
        Whether to count similar compounds.

    Attributes
    ----------
    name : str
    empty_value : any
    """

    def __init__(
            self,
            name: Optional[str] = None,
            empty_value: Any = float('nan'),
            count_unique: bool = False,
    ):
        super().__init__(name=name, empty_value=empty_value)

        self.__count_unique = count_unique

    def _calculate(
            self,
            *,
            predictions: Sequence[str],
            labels: Optional[Sequence[str]] = None,
            **kwargs,
    ) -> Tuple[Real, int]:
        """Return the number of acceptable compounds and the total number of presented
        compounds. If `count_unique` is True, filter out similar compounds. If `labels`
        is not None, filter out the compounds from `predictions` presented in `labels`.

        Parameters
        ----------
        predictions : sequence of str
            The generated SMILES strings.
        labels : sequence of str, default None
            If not None, do not count the compounds from this container.

        Returns
        -------
        tuple
            result : int
            n_samples: int
        """
        result: int = 0
        n_samples = len(predictions)

        if self.__count_unique:
            predictions = frozenset(predictions)

        if labels:
            label_set = frozenset(labels)

            for smiles in predictions:
                molecule = MolFromSmiles(smiles)
                if molecule is not None and smiles not in label_set:
                    result += 1
        else:
            for smiles in predictions:
                molecule = MolFromSmiles(smiles)
                if molecule is not None:
                    result += 1

        return result, n_samples


class KLDivergence(Metric):
    r"""Calculate the Kullback-Leibler divergence of physicochemical descriptors.

    References
    ----------
    .. [1] Brown et al. Guacamol: Benchmarking Models for de Novo Molecular Design.
       J. Chem. Inf. Model. 2019, 59, 1096−1108
    """

    @staticmethod
    def calculate_for_continuous(data_train: np.array, data_valid: np.array) -> Real:
        kde_desc_train = stats.gaussian_kde(data_train)
        kde_desc_valid = stats.gaussian_kde(data_valid)

        interval = np.linspace(
            start=min(data_train.min(), data_valid.min()),
            stop=max(data_train.max(), data_valid.max()),
            num=1000,
        )

        return stats.entropy(
            kde_desc_train.evaluate(interval) + 1e-10,
            kde_desc_valid.evaluate(interval) + 1e-10,
        )

    @staticmethod
    def calculate_for_discrete(data_train: np.array, data_valid: np.array) -> Real:
        hist_train, bins = np.histogram(data_train, density=True)
        hist_valid, _ = np.histogram(data_valid, bins=bins, density=True)

        return stats.entropy(hist_train + 1e-10, hist_valid + 1e-10)

    def _calculate(
            self,
            *,
            predictions: Sequence[str],
            labels: Sequence[str],
            **kwargs,
    ) -> Tuple[Real, int]:
        molecules_valid = check_compounds_valid(predictions, invalid='skip')
        if len(molecules_valid) < 2:
            return self.empty_value, 1

        descriptors_valid = get_descriptor_df_from_mol(molecules_valid,
                                                       PHYSCHEM_DESCRIPTOR_MAP)

        molecules_train = check_compounds_valid(labels, invalid='raise')
        descriptors_train = get_descriptor_df_from_mol(molecules_train,
                                                       PHYSCHEM_DESCRIPTOR_MAP)

        discrete_cols = set(
            c for c in PHYSCHEM_DESCRIPTOR_MAP.keys()
            if c.startswith('#')
        )
        continuous_cols = PHYSCHEM_DESCRIPTOR_MAP.keys() - discrete_cols

        kl_divs = []

        for column in continuous_cols:
            continuous_data_train = descriptors_train[column].values.astype(np.float16)
            continuous_data_valid = descriptors_valid[column].values.astype(np.float16)

            try:
                kl_div = self.calculate_for_continuous(
                    continuous_data_train, continuous_data_valid)
                kl_divs.append(kl_div)
            except np.linalg.LinAlgError:
                return self.empty_value, 1

        it = InternalTanimoto(dtype=np.float16)
        sim_train = it.fit_transform(molecules_train)
        sim_valid = it.fit_transform(molecules_valid)
        np.fill_diagonal(sim_train, 0.)
        np.fill_diagonal(sim_valid, 0.)
        sim_train = sim_train.max(axis=1)
        sim_valid = sim_valid.max(axis=1)

        try:
            kl_div = self.calculate_for_continuous(sim_train, sim_valid)
            kl_divs.append(kl_div)
        except np.linalg.LinAlgError:
            return self.empty_value, 1

        for column in discrete_cols:
            discrete_data_train = descriptors_train[column].values.astype(np.int32)
            discrete_data_valid = descriptors_valid[column].values.astype(np.int32)

            kl_div = self.calculate_for_discrete(
                discrete_data_train, discrete_data_valid)
            kl_divs.append(kl_div)

        valid_ratio = len(molecules_valid) / len(predictions)
        result = valid_ratio * np.mean([np.exp(-kl_div) for kl_div in kl_divs])
        return result, 1


class Perplexity(mx.metric.Perplexity):
    """Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray. See the documentation for more information.
    """

    def __init__(
            self,
            from_probabilities: bool = False,
            ignore_label: Optional[int] = None,
            axis=-1,
            name='Perplexity',
            output_names=None,
            label_names=None,
    ):
        super().__init__(
            ignore_label=ignore_label,
            axis=axis,
            name=name,
            output_names=output_names,
            label_names=label_names,
        )

        self._from_probabilities = from_probabilities

    def update(
            self,
            labels: mx.np.ndarray,
            preds: mx.np.ndarray,
    ):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : mxnet.np.ndarray, shape = (time steps,) or (batch size, time steps)
            The labels of the data.

        preds : mxnet.np.ndarray, shape = (time steps, vocabulary size) or
                (batch size, time steps, vocabulary size)
            Predicted values.

        See the documentation for more information.
        """
        # If predictions and labels are given as mini-batches of shape
        # (batch size, time steps, vocabulary dim)
        if preds.ndim == 3 and labels.ndim == 2:
            preds = preds.reshape(-1, preds.shape[-1])
            labels = labels.reshape(-1)

        if not self._from_probabilities:
            # noinspection PyUnresolvedReferences
            preds = mx.npx.softmax(preds)

        score = 0.0
        n_instances = 0

        # noinspection PyUnresolvedReferences
        pick_preds = mx.npx.pick(preds, labels, axis=self.axis)

        if self.ignore_label is not None:
            ignore_mask = (labels == self.ignore_label).astype(pick_preds.dtype)
            pick_preds[:] = pick_preds * (1 - ignore_mask) + ignore_mask
            n_instances -= ignore_mask.sum().astype(mx.np.int32)

        # noinspection PyUnresolvedReferences
        score -= mx.np.sum(mx.np.log(mx.np.maximum(1e-7, pick_preds))).item()
        n_instances += pick_preds.shape[0]

        self.sum_metric += score
        self.global_sum_metric += score
        self.num_inst += n_instances
        self.global_num_inst += n_instances
