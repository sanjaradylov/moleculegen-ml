"""
Additional metrics for model evaluation.

Classes
-------
Perplexity
    Re-implementation of mxnet.metrics.Perplexity, which supports Numpy
    ndarray.
"""

__all__ = (
    'Perplexity',
)


from typing import Sequence, Union

from mxnet import metric, np


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
