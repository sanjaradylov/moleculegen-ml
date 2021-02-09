"""
Build new callbacks inheriting from Callback ABC.

Classes
-------
Callback
    An ABC to build callbacks.
CallbackList
    Perform callbacks sequentially and log output messages.
"""

__all__ = (
    'Callback',
    'CallbackList',
)


import abc
import contextlib
import datetime
import functools
import io
import itertools
import sys
import time
from typing import Any, Callable, List, Optional, TextIO


class Callback(metaclass=abc.ABCMeta):
    """An ABC to build new callbacks.
    """

    def on_train_begin(self, **fit_kwargs):
        """Call at the beginning of training process."""

    def on_train_end(self, **fit_kwargs):
        """Call at the end of training process."""

    def on_batch_begin(self, **fit_kwargs):
        """Call at the beginning of batch sampling."""

    def on_batch_end(self, **fit_kwargs):
        """Call at the end of batch sampling, after trainer step."""

    def on_epoch_begin(self, **fit_kwargs):
        """Call at the beginning of an epoch."""

    def on_epoch_end(self, **fit_kwargs):
        """Call at the end of an epoch."""

    def on_keyboard_interrupt(self, **fit_kwargs):
        """Call on keyboard interruption."""


class CallbackList:
    """A list of callbacks. Perform callbacks sequentially and log output messages.

    Instantiate a callback list and append one or more callbacks:
    `callbacks = CallbackList(); callbacks.add(callback1, ..., callbackN)`

    ??? The strategy is to redirect every stdout message of callbacks into a log file,
    and optionally print back to stdout. Can we implement more convenient strategy, for
    example, using logging API?

    Parameters
    ----------
    log_handler : file-like, default None
        A log file handler.
        If None, simply call every callback, and if they print/save any messages, they
        will not be redirected.
    verbose : bool, default False
        If `log_handler` is not None, whether to print logs to stdout.
    """

    def __init__(
            self,
            log_handler: Optional[TextIO] = None,
            verbose: bool = False,
    ):
        self._log_handler = log_handler
        self._verbose = verbose

        self._callbacks: List[Callback] = []
        self._begin_time: Optional[float] = None

    def _callbacks_call(self, method_name: str, **fit_kwargs: Any):
        """Call `on_*` methods on every callback in the list. Redirect the output
        messages of calls into the logger.

        Parameters
        ----------
        method_name : str
            The name of a callback call (e.g. 'on_batch_end').
        fit_kwargs : dict, str -> any
            The key-word arguments for callback calls (see `Callback` ABC docs).
        """

        def redirect_stdout(callback_function: Callable) -> Callable:
            """A decorator to return the output messages of the callbacks into the
            logger.

            Parameters
            ----------
            callback_function : callable

            Returns
            -------
            wrapper : callable
            """

            @functools.wraps(callback_function)
            def wrapper(*args: Any, **kwargs: Any) -> str:
                handler_redirect = io.StringIO()

                with contextlib.redirect_stdout(handler_redirect):
                    callback_function(*args, **kwargs)

                return handler_redirect.getvalue()

            return wrapper

        def current_time() -> str:
            return time.strftime('%a, %d %b %Y %H:%M:%S')

        # Redirect outputs into the file handler, but also print in stdout.
        if self._log_handler is not None and self._verbose:
            def call():
                for callback in self._callbacks:
                    caller: Callable = redirect_stdout(getattr(callback, method_name))

                    # Log messages in both the log handler and stdout.
                    for handler in (self._log_handler, sys.stdout):
                        if method_name in ('on_epoch_begin', 'on_epoch_end'):
                            handler.write(
                                f'{current_time()}'
                                f' :: {method_name}'
                                f' :: {callback.__class__.__name__}'
                                f'\n'
                            )

                        log: str = caller(**fit_kwargs)
                        if log:
                            # For `ProgressBar` and similar callbacks.
                            if log.startswith('\r'):
                                handler.write(f'\r\t{log[1:]}')
                                handler.flush()
                            else:
                                handler.write(f'\t{log}')

        # Redirect outputs into the file handler.
        elif self._log_handler is not None and not self._verbose:
            def call():
                for callback in self._callbacks:
                    caller: Callable = redirect_stdout(getattr(callback, method_name))

                    if method_name in ('on_epoch_begin', 'on_epoch_end'):
                        self._log_handler.write(
                            f'{current_time()}'
                            f' :: {method_name}'
                            f' :: {callback.__class__.__name__}'
                            f'\n'
                        )

                    log: str = caller(**fit_kwargs)
                    if log:
                        # For `ProgressBar` and similar callbacks.
                        if log.startswith('\r'):
                            self._log_handler.write(f'\r\t{log[1:]}')
                            self._log_handler.flush()
                        else:
                            self._log_handler.write(f'\t{log}')

        # Simply call every callback.
        else:
            def call():
                for callback in self._callbacks:
                    caller: Callable = getattr(callback, method_name)
                    caller(**fit_kwargs)

        call()

    def on_train_begin(self, **fit_kwargs):
        """Call at the beginning of training process.
        """
        self._launch_timer()
        self._callbacks_call('on_train_begin', **fit_kwargs)

    def on_train_end(self, **fit_kwargs):
        """Call at the end of training process.
        """
        self._callbacks_call('on_train_end', **fit_kwargs)
        self._log_train_time()

    def on_batch_begin(self, **fit_kwargs):
        """Call at the beginning of batch sampling.
        """
        self._callbacks_call('on_batch_begin', **fit_kwargs)

    def on_batch_end(self, **fit_kwargs):
        """Call at the end of batch sampling, after trainer step.
        """
        self._callbacks_call('on_batch_end', **fit_kwargs)

    def on_epoch_begin(self, **fit_kwargs):
        """Call at the beginning of an epoch.
        """
        self._callbacks_call('on_epoch_begin', **fit_kwargs)

    def on_epoch_end(self, **fit_kwargs):
        """Call at the end of an epoch.
        """
        self._callbacks_call('on_epoch_end', **fit_kwargs)

    def on_keyboard_interrupt(self, **fit_kwargs):
        """Call on keyboard interruption.
        """
        self._callbacks_call('on_keyboard_interrupt', **fit_kwargs)
        self._log_train_time()

    def add(self, callback: Callback, *callbacks: Callback):
        """Append callback(s) to the list.

        Parameters
        ----------
        callback : Callback
        callbacks : tuple of Callback, default ()

        Raises
        ------
        TypeError
            If `callback` is not of type `Callback`.
        """
        callbacks_chain = list(itertools.chain((callback,), callbacks))

        for callback in callbacks_chain:
            if not isinstance(callback, Callback):
                raise TypeError(
                    f'callback must be of type Callback, not {type(callback)}'
                )

        self._callbacks.extend(callbacks_chain)

    def pop(self) -> Callback:
        """Pop the last callback.

        Returns
        -------
        callback : Callback

        Raises
        ------
        IndexError
        """
        if not self._callbacks:
            raise IndexError('pop from empty callback list')

        return self._callbacks.pop()

    def __getitem__(self, index: int) -> Callback:
        """Return the callback with index `index`.

        Parameters
        ----------
        index : int

        Returns
        -------
        callback : Callback

        Raises
        ------
        IndexError
            If `index` is out of range.
        """
        try:
            return self._callbacks[index]
        except IndexError as err:
            err.args = ('callbacks index out of range',)
            raise

    def __len__(self) -> int:
        """Return the number of callbacks in the list.

        Returns
        -------
        int
        """
        return len(self._callbacks)

    def _launch_timer(self):
        if self._log_handler is not None or self._verbose:
            self._begin_time = time.time()

    def _log_train_time(self):
        if self._log_handler is not None or self._verbose:
            end_time = time.time() - self._begin_time

            stdout = sys.stdout if self._verbose else None
            handlers = [h for h in (self._log_handler, stdout) if h is not None]
            for handler in handlers:
                handler.write(f'Time: {datetime.timedelta(seconds=end_time)}')
