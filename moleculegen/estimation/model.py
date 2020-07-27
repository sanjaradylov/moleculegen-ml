"""
Generative language models.

Classes
-------
SMILESRNNModel
    A generative recurrent neural network to encode-decode SMILES strings.
"""

__all__ = (
    'SMILESRNNModel',
)


from typing import Callable, List, Optional, Tuple, Union

import mxnet as mx
from mxnet import gluon

from . import _gluon_common
from ..description.common import OneHotEncoder


class SMILESRNNModel(gluon.Block):
    """A generative recurrent neural network to encode-decode SMILES strings.

    Parameters
    ----------
    vocab_size : int
        The vocabulary dimension, which will indicate the number of output
        neurons of a decoder.
    use_one_hot : bool, default False
        Whether to use one-hot-encoding or an embedding layer.
    embedding_dim : int, default 4
        The output dimension of an embedding layer.
    embedding_init : str or mxnet.init.Initializer,
            default mxnet.init.Orthogonal()
        The parameter initializer of an embedding layer.
    rnn : {'vanilla', 'lstm', 'gru'}, default 'lstm'
        A recurrent layer.
    n_rnn_layers : int, default 1
        The number of layers of a (deep) recurrent layer.
    n_rnn_units : int, default 64
        The number of neurons in an RNN.
    rnn_dropout : float, default 0.0
        The dropout rate of a recurrent layer.
    rnn_init : str or mxnet.init.Initializer,
            default mxnet.init.Orthogonal()
        The parameter initializer of a recurrent layer.
    n_dense_layers : int, default 1
        The number of dense layers.
    n_dense_units : int, default 128
        The number of neurons in each dense layer.
    dense_activation : str, default 'relu'
        The activation function in a dense layer.
    dense_dropout : float, default 0.0
        The dropout rate of a dense layer.
    dense_init : str or mxnet.init.Initializer,
            default mxnet.init.Xavier()
        The parameter initializer of a dense layer.
    dtype : str, default 'float32'
        Data type.
    ctx : mxnet.context.Context, default mxnet.context.cpu()
        CPU or GPU.

    prefix : str, default None
    params : mxnet.gluon.ParameterDict, default None
    """

    def __init__(
            self,
            vocab_size: int,
            use_one_hot: bool = False,
            embedding_dim: int = 4,
            embedding_init: Optional[
                Union[str, mx.init.Initializer]] = mx.init.Uniform(),
            rnn: str = 'lstm',
            n_rnn_layers: int = 1,
            n_rnn_units: int = 64,
            rnn_dropout: float = 0.,
            rnn_init: Optional[
                Union[str, mx.init.Initializer]] = mx.init.Orthogonal(),
            n_dense_layers: int = 1,
            n_dense_units: int = 128,
            dense_activation: str = 'relu',
            dense_dropout: float = 0.,
            dense_init: Optional[
                Union[str, mx.init.Initializer]] = mx.init.Xavier(),
            dtype: Optional[str] = 'float32',
            ctx: mx.context.Context = mx.context.cpu(),
            *,
            prefix: Optional[str] = None,
            params: Optional[gluon.ParameterDict] = None,
    ):
        # Validate the formal parameters that are not explicitly sent into and
        # validated in mxnet.gluon objects.
        if not isinstance(use_one_hot, bool):
            raise TypeError(
                '`use_one_hot` must be either True for OneHotEncoder layer '
                'or False for Embedding layer.'
            )

        if rnn not in _gluon_common.RNN_MAP:
            raise ValueError(
                f'The recurrent layer must be one of '
                f'{list(_gluon_common.RNN_MAP.keys())}.'
            )

        if n_dense_layers < 1:
            raise ValueError(
                'The number of dense layers must be positive non-zero.'
            )

        # Initialize mxnet.gluon.Block parameters.
        super().__init__(prefix=prefix, params=params)

        # Define (and initialize) an embedding layer.
        if use_one_hot:
            self.embedding = OneHotEncoder(vocab_size)
        else:
            self.embedding = gluon.nn.Embedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                dtype=dtype,
            )
            self.embedding.initialize(init=embedding_init, ctx=ctx)

        # Select and initialize a recurrent block.
        self.rnn = _gluon_common.RNN_MAP[rnn](
            hidden_size=n_rnn_units,
            num_layers=n_rnn_layers,
            dropout=rnn_dropout,
            dtype=dtype,
        )
        self.rnn.initialize(init=rnn_init, ctx=ctx)

        # Define and initialize a dense layer(s).
        if n_dense_layers > 1:

            if dense_dropout > 1e-3:
                mlp_func = _gluon_common.dropout_mlp
            else:
                mlp_func = _gluon_common.mlp

            self.dense = mlp_func(
                n_layers=n_dense_layers,
                n_units=n_dense_units,
                activation=dense_activation,
                output_dim=vocab_size,
                dtype=dtype,
                dropout=dense_dropout,
            )

        else:
            self.dense = gluon.nn.Dense(
                units=vocab_size,
                dtype=dtype,
                flatten=False,
            )

        self.dense.initialize(init=dense_init, ctx=ctx)

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
            including an mxnet context.

        Returns
        -------
        states : list of mxnet.np.ndarray
            The list of initial hidden states.
        """
        return self.rnn.begin_state(
            batch_size=batch_size, func=func, **func_kwargs)

    def forward(
            self,
            inputs: mx.np.ndarray,
            states: List[mx.np.ndarray],
    ) -> Tuple[mx.np.ndarray, List[mx.np.ndarray]]:
        """Run forward computation.

        Parameters
        ----------
        inputs : mxnet.np.ndarray,
                shape = (batch size, time steps)
            Input samples.
        states : list of mxnet.np.ndarray,
                shape = (rnn layers, batch size, rnn units)
            Hidden states.

        Returns
        -------
        outputs : mxnet.np.ndarray,
                shape = (time steps, batch size, vocabulary dimension)
            The decoded outputs.
        states : list of mxnet.np.ndarray,
                shape = (rnn layers, batch size, rnn units)
            The updated hidden states.
        """
        inputs = self.embedding(inputs.T)
        outputs, states = self.rnn(inputs, states)
        outputs = self.dense(outputs).swapaxes(0, 1)

        return outputs, states
