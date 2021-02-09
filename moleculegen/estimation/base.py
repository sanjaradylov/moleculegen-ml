"""
The base class for generative language models.

Classes
-------
SMILESEncoderDecoderABC
    An ABC for a generative recurrent neural network to encode-decode SMILES strings.
"""

__all__ = (
    'SMILESEncoderDecoderABC',
)


import abc
from typing import Callable, List, Optional, Sequence, TextIO, Tuple

import mxnet as mx
from mxnet import autograd, gluon

from ..callback.base import Callback, CallbackList
from ..data.sampler import BatchSampler
from ..evaluation.loss import get_mask_for_loss


class SMILESEncoderDecoderABC(gluon.HybridBlock, metaclass=abc.ABCMeta):
    """An ABC for a generative recurrent neural network to encode-decode SMILES strings.

    When subclassing, implement abstract properties `embedding`, `encoder`, and
    `decoder`.

    Parameters
    ----------
    ctx : mxnet.context.Context, default mxnet.context.cpu()
        CPU or GPU.
    prefix : str, default None
    params : mxnet.gluon.ParameterDict, default None
    """

    def __init__(
            self,
            *,
            ctx: mx.context.Context = mx.context.gpu(),
            prefix: Optional[str] = None,
            params: Optional[gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.__ctx = ctx

    @property
    @abc.abstractmethod
    def embedding(self):
        """Return the embedding layer."""

    @property
    @abc.abstractmethod
    def encoder(self):
        """Return the RNN encoder."""

    @property
    @abc.abstractmethod
    def decoder(self):
        """Return the Feed-Forward NN decoder."""

    @property
    def ctx(self) -> mx.context.Context:
        """Return the model's context.
        """
        return self.__ctx

    @ctx.setter
    def ctx(self, ctx: mx.context.Context):
        """Set the model's context and reset the parameters' context.
        """
        if ctx != self.__ctx:
            self.collect_params().reset_ctx(ctx)
            self.__ctx = ctx

    def begin_state(
            self,
            batch_size: int = 0,
            func: Callable[..., mx.nd.NDArray] = mx.nd.zeros,
            **func_kwargs,
    ) -> List[mx.np.ndarray]:
        """Return initial hidden states of a model.

        Parameters
        ----------
        batch_size : int, default 0
            Batch size.
        func : callable, any -> mxnet.nd.NDArray, default mxnet.nd.zeros
            The state initializer function.
        **func_kwargs
            Additional arguments for RNN layer's `begin_state` method
            excluding an mxnet context (we explicitly pass the model's ctx).

        Returns
        -------
        states : list of mxnet.np.ndarray
            The list of initial hidden states.
        """
        return self.encoder.begin_state(
            batch_size=batch_size, func=func, ctx=self.__ctx, **func_kwargs)

    # noinspection PyMethodOverriding
    def hybrid_forward(
            self,
            module,
            inputs: mx.np.ndarray,
            states: List[mx.np.ndarray],
    ) -> Tuple[mx.np.ndarray, List[mx.np.ndarray]]:
        """Run forward computation.

        Parameters
        ----------
        module : mxnet.ndarray or mxnet.symbol
            We ignore the model.
        inputs : mxnet.np.ndarray,
                shape = (batch size, time steps)
            Input samples.
        states : list of mxnet.np.ndarray,
                shape = (rnn layers, batch size, rnn units)
            Hidden states.

        Returns
        -------
        outputs : mxnet.np.ndarray,
                shape = (batch size, time steps, vocabulary dimension)
            The decoded outputs.
        states : list of mxnet.np.ndarray,
                shape = (rnn layers, batch size, rnn units)
            The updated hidden states.
        """
        inputs = self.embedding(inputs.T)
        outputs, states = self.encoder(inputs, states)
        outputs = self.decoder(outputs).swapaxes(0, 1)

        return outputs, states

    def fit(
            self,
            batch_sampler: BatchSampler,
            optimizer: mx.optimizer.Optimizer,
            loss_fn: gluon.loss.Loss,
            n_epochs: int = 1,
            callbacks: Optional[Sequence[Callback]] = None,
            log_filename: Optional[str] = None,
            log_verbose: Optional[bool] = False,
    ):
        """Train the model for `n_epochs` number of epochs. Optionally call `callbacks`
        to monitor training progress.

        Parameters
        ----------
        batch_sampler : BatchSampler
            The dataloader to sample mini-batches.
        optimizer : mxnet.optimizer.Optimizer
            An mxnet optimizer.
        loss_fn : mxnet.gluon.loss.Loss
            A gluon loss functor.
        n_epochs : int, default 1
            The number of epochs to train.
        callbacks : sequence of Callback, default None
            The callbacks to run during training.
        log_filename : str, default None
            The path to a log file.
            If None, callbacks' default options can still print or save some logs. See
            docs of the callbacks used.
        log_verbose : bool, default False
            If `log_filename` is specified, whether to additionally print logs to stdout.
        """
        # The named arguments required by callbacks on epoch/batch calls.
        init_params = {
            # Constant parameters.
            'batch_sampler': batch_sampler,
            'optimizer': optimizer,
            'loss_fn': loss_fn,
            'n_epochs': n_epochs,
            # Changeable parameters.
            'model': self,
            'epoch': 0,
        }
        # The named arguments required by callbacks on batch calls.
        train_params = dict.fromkeys([
            'batch_no', 'batch', 'loss', 'predictions', 'outputs'
        ])
        callbacks = callbacks or []
        callback_list: Optional[CallbackList] = None
        log_handler: Optional[TextIO] = None

        try:
            if log_filename is not None:
                log_handler = open(log_filename, 'w')

            callback_list = CallbackList(log_handler, verbose=log_verbose)
            callback_list.add(*callbacks)

            self.hybridize()

            trainer = gluon.Trainer(self.collect_params(), optimizer)
            state_func: Callable = batch_sampler.init_state_func()

            callback_list.on_train_begin(**init_params)

            for epoch in range(1, n_epochs + 1):

                init_params['epoch'] = epoch
                callback_list.on_epoch_begin(**init_params)

                states: Optional[List[mx.np.ndarray]] = None

                for batch_no, batch in enumerate(batch_sampler, start=1):

                    batch = batch.as_in_ctx(self.__ctx)
                    states = batch_sampler.init_states(
                        model=self,
                        mini_batch=batch,
                        states=states,
                        init_state_func=state_func,
                    )

                    train_params['batch_no'] = batch_no
                    train_params['batch'] = batch
                    callback_list.on_batch_begin(**init_params, **train_params)

                    with autograd.record():
                        predictions, states = self.forward(batch.inputs, states)
                        weights = get_mask_for_loss(batch.shape, batch.valid_lengths)
                        loss = loss_fn(predictions, batch.outputs, weights)

                    loss.backward()
                    trainer.step(batch.valid_lengths.sum())

                    train_params['loss'] = loss
                    train_params['predictions'] = predictions
                    train_params['outputs'] = batch.outputs
                    callback_list.on_batch_end(**init_params, **train_params)

                init_params['model'] = self
                callback_list.on_epoch_end(**init_params, **train_params)

            callback_list.on_train_end(**init_params)

        except KeyboardInterrupt:
            init_params['model'] = self
            # If keyboard interrupt was not triggered before the callback list
            # initialization.
            if callback_list is not None:
                callback_list.on_keyboard_interrupt(**init_params)

        finally:
            # If `log_handler` is a file-like and was opened.
            if log_handler is not None:
                log_handler.close()
