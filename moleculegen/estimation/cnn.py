"""
Generative convolutional neural network models.
"""

__all__ = (
    'SMILESCNN',
    'SMILESConv1D',
    'SMILESForwardPad',
)

from typing import Callable, List, Optional, Union

import mxnet as mx

from .._types import ActivationT, ContextT, InitializerT
from ..description.common import OneHotEncoder
from ._gluon_common import mlp
from .base import SMILESLM


class SMILESCNN(SMILESLM):
    """Convolutional neural network for language modeling.

    Parameters
    ----------
    vocab_size : int
        The vocabulary dimension, which will indicate the number of output
        neurons of a decoder.
    use_one_hot : bool, default=False
        Whether to use one-hot-encoding or an embedding layer.
    embedding_dim : int, default=32
        The output dimension of an embedding layer.
    embedding_dropout : float, default=0.4
        The dropout rate of an embedding layer.
    embedding_init : {'uniform', 'normal', 'orthogonal', 'xavier'}
            or mxnet.init.Initializer or None,
            default=mxnet.init.Xavier()
        The parameter initializer of an embedding layer.
    embedding_prefix : str, default='embedding_'
        The prefix of an embedding block.
    cnn_n_layers : int, default=4
        The number of convolution blocks.
    cnn_channels : int or list of int, default=64
        The number of convolution channels.
    cnn_kernel_size : int, default=3
    cnn_activation : {'sigmoid', 'tanh', 'relu', 'softrelu', 'softsign'}
            or mxnet.gluon.nn.Activation
            or callable, mxnet.nd.NDArray -> mxnet.nd.NDArray
            or None,
            default='relu'
    cnn_use_bias : bool, default=False
        Whether to use bias parameter of convolution block.
    cnn_residual : bool, default=False
        Whether to include residual connection.
    cnn_dropout : float, default=0.5
        Apply dropout after all convolutions.
    cnn_init : {'uniform', 'normal', 'orthogonal', 'xavier'}
            or mxnet.init.Initializer or None,
            default=mxnet.init.Xavier()
    cnn_prefix : str, default='decoder_'
    dense_n_layers : int, default=2
        The number of dense layers.
    dense_n_units : int, default=128
        The number of neurons in each dense layer.
    dense_activation : {'sigmoid', 'tanh', 'relu', 'softrelu', 'softsign'}
            or mxnet.gluon.nn.Activation
            or callable, mxnet.nd.NDArray -> mxnet.nd.NDArray
            or None,
            default='relu'
        The activation function in a dense layer.
    dense_dropout : float, default=0.5
        The dropout rate of a dense layer.
    dense_init : {'uniform', 'normal', 'orthogonal', 'xavier'}
            or mxnet.init.Initializer or None,
            default=mxnet.init.Xavier()
        The parameter initializer of a dense layer.
    dense_prefix : str, default='decoder_'
        The prefix of a decoder block.
    initialize : bool, default=True
        Whether to initialize model parameters.
        When one decides to load parameters from a file, deferred
        initialization is needless.
    dtype : str or numpy dtype or None, default='float32'
        Data type.
    ctx : mxnet.context.Context or list of mxnet.context.Context,
            default=mxnet.context.cpu()
        CPU or GPU.

    Other Parameters
    ----------------
    prefix : str, default=None
    params : mxnet.gluon.ParameterDict, default=None

    Attributes
    ----------
    ctx : mxnet.context.Context or list of mxnet.context.Context
        The model's context.
    embedding : moleculegen.description.OneHotEncoder or mxnet.gluon.nn.Embedding
            or mxnet.gluon.nn.HybridSequential
        An embedding block.
    encoder : SMILESConv1D
        An CNN encoder block.
    decoder : mxnet.gluon.nn.Dense or mxnet.gluon.nn.HybridSequential
        A Feed-Forward NN decoder block.
    """

    def __init__(
            self,
            vocab_size: int,
            *,

            use_one_hot: bool = False,
            embedding_dim: int = 32,
            embedding_dropout: float = 0.25,
            embedding_init: InitializerT = mx.init.Xavier(),
            embedding_prefix: str = 'embedding_',

            cnn_n_layers: int = 4,
            cnn_channels: Union[int, List[int]] = 64,
            cnn_kernel_size: int = 3,
            cnn_activation: Union[ActivationT, List[ActivationT]] = 'relu',
            cnn_use_bias: bool = False,
            cnn_residual: bool = False,
            cnn_dropout: float = 0.5,
            cnn_init: InitializerT = mx.init.Xavier(),
            cnn_prefix: str = 'encoder_',

            dense_n_layers: int = 2,
            dense_n_units: Union[int, List[int]] = 128,
            dense_activation: Union[ActivationT, List[ActivationT]] = 'relu',
            dense_dropout: Union[float, List[float]] = 0.25,
            dense_init: InitializerT = mx.init.Xavier(),
            dense_prefix: str = 'decoder_',

            initialize: bool = True,
            dtype: Optional[str] = 'float32',
            ctx: Optional[ContextT] = None,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(ctx=ctx, prefix=prefix, params=params)

        with self.name_scope():
            if use_one_hot:
                self._embedding = OneHotEncoder(vocab_size)
            else:
                embedding_block = mx.gluon.nn.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_dim,
                    dtype=dtype,
                    prefix=embedding_prefix,
                )

                if embedding_dropout != 0.:
                    seq_prefix = f'{embedding_prefix.rstrip("_")}seq_'
                    self._embedding = mx.gluon.nn.Sequential(prefix=seq_prefix)
                    self._embedding.add(embedding_block)
                    self._embedding.add(mx.gluon.nn.Dropout(embedding_dropout, axes=1))
                else:
                    self._embedding = embedding_block

                if initialize:
                    self._embedding.initialize(init=embedding_init, ctx=ctx)

            self._encoder = SMILESConv1D(
                n_blocks=cnn_n_layers,
                channels=cnn_channels,
                kernel_size=cnn_kernel_size,
                activation=cnn_activation,
                residual=cnn_residual,
                use_bias=cnn_use_bias,
                prefix=cnn_prefix,
            )
            self._encoder.cast(dtype)
            if initialize:
                self._encoder.initialize(init=cnn_init, ctx=ctx)

            self._dropout = (
                mx.gluon.nn.Dropout(cnn_dropout, axes=2) if cnn_dropout != 0.
                else lambda x: x
            )

            self._decoder = mlp(
                n_layers=dense_n_layers,
                n_units=dense_n_units,
                activation=dense_activation,
                dropout=dense_dropout,
                output_dim=vocab_size,
                dtype=dtype,
                prefix=dense_prefix,
            )
            if initialize:
                self._decoder.initialize(init=dense_init, ctx=ctx)

    @property
    def embedding(self) -> Union[
            OneHotEncoder, mx.gluon.nn.Embedding, mx.gluon.nn.Sequential]:
        return self._embedding

    @property
    def encoder(self) -> 'SMILESConv1D':
        return self._encoder

    @property
    def decoder(self) -> Union[mx.gluon.nn.Dense, mx.gluon.nn.Sequential]:
        return self._decoder

    # noinspection PyMethodOverriding
    def forward(self, x) -> mx.np.ndarray:
        """Run forward computation.

        Parameters
        ----------
        x : mxnet.np.ndarray, shape = (batch size, time steps)

        Returns
        -------
        mxnet.np.ndarray, shape = (batch size, time steps, vocabulary dimension)
        """
        # b=batch size, t=time steps, v=vocab dim, e=embed dim, c=channels
        x = self._embedding(x)  # input=(b, t), output=(b, t, e)
        x = self._encoder(x.swapaxes(1, 2))  # input=(b, e, t), output=(b, c, t)
        x = self._dropout(x)  # output=(b, c, t)
        x = self._decoder(x.swapaxes(1, 2))  # input=(b, t, c), output=(b, t, v)
        return x


class SMILESForwardPad(mx.gluon.Block):
    """Pad the beginning of SMILES sequence.

    Used in conjunction with `SMILESConv1D` to get the same number of channels after
    convolution.

    Parameters
    ----------
    pad_size : int
    """

    def __init__(
            self,
            pad_size: int,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.pad_size = pad_size

    # noinspection PyMethodOverriding
    def forward(self, x):
        x_pad = mx.np.zeros(shape=(x.shape[0], x.shape[1], self.pad_size), ctx=x.ctx)
        return mx.np.concatenate((x_pad, x), axis=2)


class SMILESConv1D(mx.gluon.Block):
    """Apply multiple 1D convolutions sequentially. Prepend padding of size
    `kernel_size-1` to get outputs with the same number of channels as in inputs.
    """

    def __init__(
            self,
            n_blocks: int = 2,
            channels: Union[int, List[int]] = 128,
            kernel_size: int = 4,
            activation: Union[ActivationT, List[ActivationT]] = 'relu',
            use_bias: bool = True,
            residual: bool = False,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        if isinstance(channels, int):
            channels = [channels] * n_blocks
        elif len(channels) != n_blocks:
            raise ValueError(f'`channels` must contain exactly `n_blocks` elements.')
        if not isinstance(activation, list):
            activation = [activation] * n_blocks
        elif len(activation) != n_blocks:
            raise ValueError(f'`activation` must contain exactly `n_blocks` elements.')

        self._pad_size = kernel_size - 1

        self._blocks = mx.gluon.nn.Sequential()
        for i in range(n_blocks):
            self._blocks.add(SMILESForwardPad(self._pad_size))
            self._blocks.add(mx.gluon.nn.Conv1D(
                channels=channels[i],
                kernel_size=kernel_size,
                activation=None if i == n_blocks-1 and residual else activation[i],
                use_bias=use_bias,
            ))

        final_activation = activation[-1]
        if residual:
            if (
                    isinstance(final_activation, mx.gluon.nn.Activation)
                    or isinstance(final_activation, Callable)
            ):
                self._final_activation = final_activation
            else:
                self._final_activation = mx.gluon.nn.Activation(final_activation)

            self.forward = self._residual_forward
        else:
            self._final_activation = lambda x: x

            self.forward = self._no_residual_forward

    def _no_residual_forward(self, x):
        return self._blocks(x)

    def _residual_forward(self, x):
        return self._final_activation(self._blocks(x) + x)
