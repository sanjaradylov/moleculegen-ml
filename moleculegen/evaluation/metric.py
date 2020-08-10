"""
Additional metrics for model evaluation.

Classes
-------
Metric
    A metric ABC for the comparison of training/validation and generated compounds.
Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.
RAC
    Calculate the rate of unique acceptable compounds.
"""

__all__ = (
    'Metric',
    'Perplexity',
    'RAC',
)


import abc
from numbers import Real
from typing import Any, List, Optional, Tuple

import mxnet as mx
from rdkit.Chem import MolFromSmiles


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

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}(name={self.__name!r}) at {hex(id(self))}>'

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
            predictions: List[str],
            labels: Optional[List[str]] = None,
            **kwargs,
    ) -> Tuple[Real, int]:
        """Return the number of acceptable compounds and the total number of presented
        compounds. If `count_unique` is True, filter out similar compounds. If `labels`
        is not None, filter out the compounds from `predictions` presented in `labels`.

        Parameters
        ----------
        predictions : list of str
            The generated SMILES strings.
        labels : list of str, default None
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
