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
from typing import Any, List, Optional, Sequence, Tuple, Union

from rdkit.Chem import MolFromSmiles
from mxnet import metric, np


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


class Perplexity(metric.Perplexity):
    """Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray. See the documentation for more information.
    """

    def update(
            self,
            labels: Union[np.ndarray, Sequence[np.ndarray]],
            predictions: Union[np.ndarray, Sequence[np.ndarray]],
    ):
        """Updates the internal evaluation result.

        Parameters
        ----------
        labels : sequence of mxnet.np.ndarray
            The labels of the data.

        predictions : sequence of mxnet.np.ndarray
            Predicted values.

        See the documentation for more information.
        """
        if len(labels) != len(predictions):
            raise ValueError(
                f"Labels vs. Predictions size mismatch: "
                f"{len(labels)} vs. {len(predictions)}"
            )

        loss = 0.0
        num = 0

        for label, prediction in zip(labels, predictions):
            probability = prediction[label.astype(np.int32).item()]
            if self.ignore_label is not None:
                ignore = (label == self.ignore_label).astype(prediction.dtype)
                num -= ignore.astype(np.int32).item()
                probability = probability * (1 - ignore) + ignore
            # noinspection PyUnresolvedReferences
            loss -= np.sum(np.log(np.maximum(1e-10, probability))).item()
            num += 1

        self.sum_metric += loss
        self.global_sum_metric += loss
        self.num_inst += num
        self.global_num_inst += num
