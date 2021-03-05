"""
Generative RNN models.

Classes:
    SMILESRNN: A generative recurrent neural network to encode-decode SMILES strings.
"""

__all__ = (
    'SMILESRNN',
)

import json
from typing import Callable, List, Optional, Union

import mxnet as mx

from .._types import ActivationT, ContextT, InitializerT
from ..description.common import OneHotEncoder
from . import _gluon_common
from .base import SMILESLM


class SMILESRNN(SMILESLM):
    """A generative recurrent neural network to encode-decode SMILES strings.

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
            default=mxnet.init.Uniform()
        The parameter initializer of an embedding layer.
    embedding_prefix : str, default='embedding_'
        The prefix of an embedding block.
    rnn : {'vanilla', 'lstm', 'gru'}, default='lstm'
        A recurrent layer.
    rnn_n_layers : int, default=2
        The number of layers of a (deep) recurrent layer.
    rnn_n_units : int, default=256
        The number of neurons in an RNN.
    rnn_dropout : float, default=0.6
        The dropout rate of a recurrent layer.
    rnn_init : {'uniform', 'normal', 'orthogonal', 'xavier'}
            or mxnet.init.Initializer or None,
            default=mxnet.init.Orthogonal()
        The parameter initializer of a recurrent layer.
    rnn_prefix : str, default='encoder_'
        The prefix of an encoder block.
    rnn_state_init : callable, any -> mxnet.nd.NDArray, default=mxnet.nd.zeros
        The RNN hidden state initializer.
    rnn_reinit_state : bool, default=False
        Whether to reinitialize the hidden state on `begin_state` call.
    rnn_detach_state : bool, default=True
        Whether to detach the hidden state from the computational graph on `begin_state`
        call.
    dense_n_layers : int, default=1
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
    tie_weights : bool, default=False
        Whether to share the embedding block parameters w/ a decoder block.
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
    ctx : mxnet.context.Context
        The model's context.
    embedding : OneHotEncoder or mxnet.gluon.nn.Embedding
        An embedding block.
    encoder : mxnet.gluon.rnn.RNN or mxnet.gluon.rnn.LSTM
            or mxnet.gluon.rnn.GRU
        An RNN encoder block.
    decoder : mxnet.gluon.nn.Dense or mxnet.gluon.nn.Sequential
        A Feed-Forward NN decoder block.
    state : list of mxnet.np.ndarray, shape = (rnn_n_layers, batch size, n_rnn_units)
        The hidden state of `encoder`.
    state_initializer : callable, any -> mxnet.nd.NDArray
        The hidden state initializer.
    reinit_state : bool
        Whether to reinitialize the hidden state on `begin_state` call.
    detach_state : bool
        Whether to detach the hidden state from the computational graph on `begin_state`
        call.
    """

    def __init__(
            self,
            vocab_size: int,
            *,

            use_one_hot: bool = False,
            embedding_dim: int = 32,
            embedding_dropout: float = 0.4,
            embedding_init: InitializerT = mx.init.Uniform(),
            embedding_prefix: str = 'embedding_',

            rnn: str = 'lstm',
            rnn_n_layers: int = 2,
            rnn_n_units: int = 256,
            rnn_dropout: float = 0.6,
            rnn_init: InitializerT = mx.init.Orthogonal(),
            rnn_prefix: str = 'encoder_',
            rnn_state_init: Callable[..., mx.nd.NDArray] = mx.nd.zeros,
            rnn_reinit_state: bool = False,
            rnn_detach_state: bool = True,

            dense_n_layers: int = 1,
            dense_n_units: Union[int, List[int]] = 128,
            dense_activation: Union[ActivationT, List[ActivationT]] = 'relu',
            dense_dropout: Union[float, List[float]] = 0.5,
            dense_init: InitializerT = mx.init.Xavier(),
            dense_prefix: str = 'decoder_',

            tie_weights: bool = False,
            initialize: bool = True,
            dtype: Optional[str] = 'float32',
            ctx: Optional[ContextT] = None,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        # Validate the formal parameters that are not explicitly sent into and
        # validated in mxnet.gluon objects.
        if not isinstance(use_one_hot, bool):
            raise TypeError(
                '`use_one_hot` must be either True for OneHotEncoder layer '
                'or False for Embedding layer.'
            )

        if not isinstance(initialize, bool):
            raise TypeError(
                '`initialize` must be either True for deferred '
                'initialization or False for no initialization.'
            )

        if rnn not in _gluon_common.RNN_MAP:
            raise ValueError(
                f'The recurrent layer must be one of '
                f'{list(_gluon_common.RNN_MAP.keys())}.'
            )

        if dense_n_layers < 1:
            raise ValueError(
                'The number of dense layers must be positive non-zero.'
            )

        if (
                tie_weights
                and (
                    embedding_dim != rnn_n_units
                    or
                    dense_n_layers > 1
                    and embedding_dim != dense_n_units
                )
        ):
            raise ValueError(
                f'When sharing weights, the number of hidden units must be equal to '
                f'the embedding dimension.'
            )

        super().__init__(ctx=ctx, prefix=prefix, params=params)

        with self.name_scope():
            # Define (and initialize) an embedding block.
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
                    self._embedding = mx.gluon.nn.HybridSequential(prefix=seq_prefix)
                    self._embedding.add(embedding_block)
                    self._embedding.add(mx.gluon.nn.Dropout(embedding_dropout))
                    shared_params = self._embedding[0].params if tie_weights else None
                else:
                    self._embedding = embedding_block
                    shared_params = self._embedding.params if tie_weights else None

                if initialize:
                    self._embedding.initialize(init=embedding_init, ctx=ctx)

            # Select (and initialize) a recurrent block.
            self._encoder = _gluon_common.RNN_MAP[rnn](
                hidden_size=rnn_n_units,
                num_layers=rnn_n_layers,
                dropout=rnn_dropout,
                dtype=dtype,
                prefix=rnn_prefix,
            )
            if initialize:
                self._encoder.initialize(init=rnn_init, ctx=ctx)

            # Define (and initialize) a dense (sequential) block.
            self._decoder = _gluon_common.mlp(
                n_layers=dense_n_layers,
                n_units=dense_n_units,
                activation=dense_activation,
                output_dim=vocab_size,
                dtype=dtype,
                dropout=dense_dropout,
                prefix=dense_prefix,
                params=shared_params,
            )
            if initialize:
                self._decoder.initialize(init=dense_init, ctx=ctx)

        self._state: Optional[List[mx.np.ndarray]] = None
        self.state_initializer: Callable[..., mx.nd.NDArray] = rnn_state_init
        self.reinit_state: bool = rnn_reinit_state
        self.detach_state: bool = rnn_detach_state

    @property
    def embedding(self) -> Union[
            OneHotEncoder, mx.gluon.nn.Embedding, mx.gluon.nn.HybridSequential]:
        return self._embedding

    @property
    def encoder(self) -> mx.gluon.rnn.RNN:
        return self._encoder

    @property
    def decoder(self) -> Union[mx.gluon.nn.Dense, mx.gluon.nn.HybridSequential]:
        return self._decoder

    @property
    def state(self) -> List[mx.np.ndarray]:
        return self._state

    def empty_state(self):
        self._state = None

    def state_info(self, batch_size):
        return self.encoder.state_info(batch_size)

    def begin_state(self, batch_size: int = 0, **func_kwargs):
        """Reinitialize the hidden state or keep the previous one. Optionally, detach it
        from the computational graph.

        Parameters
        ----------
        batch_size : int, default 0
            Batch size.
        func_kwargs
            Additional arguments for RNN layer's `begin_state` method
            excluding an mxnet context (we explicitly pass the model's ctx).
        """
        if self._state is None or self.reinit_state:
            self._state = self.encoder.begin_state(
                batch_size=batch_size, func=self.state_initializer, ctx=self.ctx,
                **func_kwargs)
        if self.detach_state:
            self._state = [s.detach() for s in self._state]

    # noinspection PyMethodOverriding
    def forward(self, x: mx.np.ndarray) -> mx.np.ndarray:
        """Run forward computation.

        Parameters
        ----------
        x : mxnet.np.ndarray, shape = (batch size, time steps)

        Returns
        -------
        mxnet.np.ndarray, shape = (batch size, time steps, vocabulary dimension)
        """
        # b=batch size, t=time steps, v=vocab dim, e=embed dim, h=hidden units
        x = self.embedding(x.T)  # input=(t, b), output=(t, b, e)
        x, self._state = self.encoder(x, self._state)  # output=(t, b, h)
        x = self.decoder(x)  # output=(t, b, v)
        return x.swapaxes(0, 1)  # output=(b, t, v)

    @classmethod
    def from_config(cls, config_file: str) -> 'SMILESRNN':
        """Instantiate a model loading its formal parameters from the JSON file named
        `config_file`.

        Returns
        -------
        model : SMILESRNN
        """
        with open(config_file) as fh:
            raw_data = json.load(fh)

        return cls(
            vocab_size=raw_data['vocab_size'],
            initialize=raw_data['initialize'],
            tie_weights=raw_data['tie_weights'],
            dtype=raw_data['dtype'],
            ctx=_gluon_common.get_ctx(raw_data['ctx'].lower()),
            prefix=raw_data['prefix'],

            use_one_hot=raw_data['embedding']['use_one_hot'],
            embedding_dim=raw_data['embedding']['dim'],
            embedding_dropout=raw_data['embedding']['dropout'],
            embedding_init=_gluon_common.INIT_MAP[raw_data['embedding']['init'].lower()],
            embedding_prefix=raw_data['embedding']['prefix'],

            rnn=raw_data['encoder']['rnn'],
            rnn_n_layers=raw_data['encoder']['n_layers'],
            rnn_n_units=raw_data['encoder']['n_units'],
            rnn_dropout=raw_data['encoder']['dropout'],
            rnn_init=_gluon_common.INIT_MAP[raw_data['encoder']['init'].lower()],
            rnn_prefix=raw_data['encoder']['prefix'],
            rnn_reinit_state=raw_data['encoder']['rnn_reinit_state'],
            rnn_detach_state=raw_data['encoder']['rnn_detach_state'],

            dense_n_layers=raw_data['decoder']['n_layers'],
            dense_n_units=raw_data['decoder']['n_units'],
            dense_activation=raw_data['decoder']['activation'],
            dense_dropout=raw_data['decoder']['dropout'],
            dense_init=_gluon_common.INIT_MAP[raw_data['decoder']['init'].lower()],
            dense_prefix=raw_data['decoder']['prefix'],
        )
