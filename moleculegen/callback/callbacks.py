"""
The collection of callbacks.

Classes
-------
ProgressBar
    Print progress bar every epoch of model training.
"""

import collections
import datetime
import math
import statistics
import sys
import time
from typing import Deque, List, Optional

from .base import Callback


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
            + '{batch_time:.2f} sec/batch'
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
        self._loss_list = []

        self._n_batches = len(fit_kwargs.get('batch_sampler'))
        self._batches_per_char = self._n_batches // self.__length or 1
        self._batch_digits = len(str(self._n_batches))

        if self._n_batches < self.__length:
            bar_length = self._n_batches
        else:
            bar_length = self.__length
        self._init_bar(bar_length)

        epoch_digits = len(str(fit_kwargs.get('n_epochs')))
        self._epoch = f'{fit_kwargs.get("epoch"):>{epoch_digits}}'
        self._epoch_start_time = time.time()

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
