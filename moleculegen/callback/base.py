"""
Build new callbacks inheriting from Callback ABC.

Classes
-------
Callback
    An ABC to build callbacks.
"""

import abc


class Callback(metaclass=abc.ABCMeta):
    """An ABC to build new callbacks.
    """

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
