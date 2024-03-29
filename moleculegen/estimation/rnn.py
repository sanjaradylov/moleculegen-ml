"""
Generative RNN models.

Classes:
    SMILESRNN: A generative recurrent neural network to encode-decode SMILES strings.
"""

__all__ = (
    'SMILESRNN',
)

import json
from typing import Callable, List, Literal, Optional, Tuple, Union

import mxnet as mx

from .._types import ActivationT, ContextT, InitializerT, StateInitializerT
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
    embedding_dropout_axes : int or tuple of int, default=1
        Whether to drop out entire feature column (1), entire token column (2), or
        token-feature entries selectively (0).
    embedding_init : {'uniform', 'normal', 'orthogonal_uniform', 'orthogonal_normal',
            'xavier_uniform', 'xavier_normal'},
            or mxnet.init.Initializer or None,
            default='xavier_uniform'
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
    rnn_i2h_init : {'uniform', 'normal', 'orthogonal_uniform', 'orthogonal_normal',
            'xavier_uniform', 'xavier_normal'},
            or mxnet.init.Initializer or None,
            default='xavier_uniform'
        The input-to-hidden parameter initializer of a recurrent layer.
    rnn_h2h_init : {'uniform', 'normal', 'orthogonal_uniform', 'orthogonal_normal',
            'xavier_uniform', 'xavier_normal'},
            or mxnet.init.Initializer or None,
            default='orthogonal_normal'
        The hidden-to-hidden parameter initializer of a recurrent layer.
    rnn_prefix : str, default='encoder_'
        The prefix of an encoder block.
    rnn_state_init : callable, any -> mxnet.nd.NDArray,
            or {'zeros', 'ones', 'uniform', 'normal'},
            default='zeros'
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
    dense_init : {'uniform', 'normal', 'orthogonal_uniform', 'orthogonal_normal',
            'xavier_uniform', 'xavier_normal'},
            or mxnet.init.Initializer or None,
            default='xavier_uniform'
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
    embedding : moleculegen.description.OneHotEncoder or mxnet.gluon.nn.Embedding
            or mxnet.gluon.nn.HybridSequential
        An embedding block. (read-only)
    encoder : mxnet.gluon.rnn.RNN or mxnet.gluon.rnn.LSTM or mxnet.gluon.rnn.GRU
        An RNN encoder block. (read-only)
    decoder : mxnet.gluon.nn.Dense or mxnet.gluon.nn.HybridSequential
        A Feed-Forward NN decoder block. (read-only)
    state : list of mxnet.np.ndarray, shape = (rnn_n_layers, batch size, n_rnn_units)
        The hidden state of `encoder`. (read-only)

    ctx : mxnet.context.Context or list of mxnet.context.Context
        The model's context. (writable)
    state_initializer : callable, any -> mxnet.nd.NDArray
        The hidden state initializer. (writable)
    reinit_state : bool
        Whether to reinitialize the hidden state on `begin_state` call. (writable)
    detach_state : bool
        Whether to detach the hidden state from the computational graph on `begin_state`
        call. (writable)
    """

    def __init__(
            self,
            vocab_size: int,
            *,

            use_one_hot: bool = False,
            embedding_dim: int = 32,
            embedding_dropout: float = 0.4,
            embedding_dropout_axes: Union[int, Tuple[int]] = 1,
            embedding_init: InitializerT = 'xavier_uniform',
            embedding_prefix: str = 'embedding_',

            rnn: Literal['lstm', 'gru', 'vanilla'] = 'lstm',
            rnn_n_layers: int = 2,
            rnn_n_units: int = 256,
            rnn_dropout: float = 0.6,
            rnn_i2h_init: InitializerT = 'xavier_uniform',
            rnn_h2h_init: InitializerT = 'orthogonal_normal',
            rnn_prefix: str = 'encoder_',
            rnn_state_init: StateInitializerT = 'zeros',
            rnn_reinit_state: bool = False,
            rnn_detach_state: bool = True,

            dense_n_layers: int = 1,
            dense_n_units: Union[int, List[int]] = 128,
            dense_activation: Union[ActivationT, List[ActivationT]] = 'relu',
            dense_dropout: Union[float, List[float]] = 0.5,
            dense_init: InitializerT = 'xavier_uniform',
            dense_prefix: str = 'decoder_',

            tie_weights: bool = False,
            initialize: bool = True,
            dtype: Optional[str] = 'float32',
            ctx: Optional[ContextT] = None,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        if (
                tie_weights
                and (
                    embedding_dim != rnn_n_units
                    and dense_n_layers == 1
                    or dense_n_layers > 1
                    and embedding_dim != dense_n_units
                )
        ):
            raise ValueError(
                f'When sharing weights, the number of hidden units must be equal to '
                f'the embedding dimension.'
            )

        super().__init__(ctx=ctx, prefix=prefix, params=params)

        with self.name_scope():
            if use_one_hot:
                self._embedding = OneHotEncoder(vocab_size)
                shared_params = None
            else:
                embedding_block = mx.gluon.nn.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_dim,
                    dtype=dtype,
                    prefix=embedding_prefix,
                    weight_initializer=_gluon_common.get_init(embedding_init),
                )

                if embedding_dropout != 0.:
                    seq_prefix = f'{embedding_prefix.rstrip("_")}seq_'
                    self._embedding = mx.gluon.nn.HybridSequential(prefix=seq_prefix)
                    self._embedding.add(embedding_block)
                    self._embedding.add(mx.gluon.nn.Dropout(
                        embedding_dropout, axes=embedding_dropout_axes))
                    shared_params = self._embedding[0].params if tie_weights else None
                else:
                    self._embedding = embedding_block
                    shared_params = self._embedding.params if tie_weights else None

            self._encoder = _gluon_common.RNN_MAP[rnn](
                hidden_size=rnn_n_units,
                num_layers=rnn_n_layers,
                dropout=rnn_dropout,
                i2h_weight_initializer=_gluon_common.get_init(rnn_i2h_init),
                h2h_weight_initializer=_gluon_common.get_init(rnn_h2h_init),
                dtype=dtype,
                prefix=rnn_prefix,
            )

            self._decoder = _gluon_common.mlp(
                n_layers=dense_n_layers,
                n_units=dense_n_units,
                activation=dense_activation,
                output_dim=vocab_size,
                dtype=dtype,
                dropout=dense_dropout,
                init=_gluon_common.get_init(dense_init),
                prefix=dense_prefix,
                params=shared_params,
            )

            if initialize:
                self.initialize(ctx=self.ctx)

        self._state: Optional[List[mx.np.ndarray]] = None
        self.state_initializer: Callable[..., mx.nd.NDArray] = \
            _gluon_common.get_state_init(rnn_state_init)
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
        batch_size : int, default=0
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

    def forward(self, batch, *args, **kwargs) -> mx.np.ndarray:
        """Run forward computation.

        Parameters
        ----------
        batch : moleculegen.data.Batch,
                batch.inputs.shape = (batch size, time steps)

        Returns
        -------
        mxnet.np.ndarray, shape = (batch size, time steps, vocabulary dimension)
        """
        # b=batch size, t=time steps, v=vocab dim, e=embed dim, h=hidden units
        x = self.embedding(batch.inputs.T)  # input=(t, b), output=(t, b, e)
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
            embedding_dropout_axes=raw_data['embedding']['dropout_axes'],
            embedding_init=_gluon_common.get_init(raw_data['embedding']['init'].lower()),
            embedding_prefix=raw_data['embedding']['prefix'],

            rnn=raw_data['encoder']['rnn'],
            rnn_n_layers=raw_data['encoder']['n_layers'],
            rnn_n_units=raw_data['encoder']['n_units'],
            rnn_dropout=raw_data['encoder']['dropout'],
            rnn_i2h_init=_gluon_common.get_init(raw_data['encoder']['i2h_init'].lower()),
            rnn_h2h_init=_gluon_common.get_init(raw_data['encoder']['h2h_init'].lower()),
            rnn_prefix=raw_data['encoder']['prefix'],
            rnn_state_init=_gluon_common.get_state_init(
                raw_data['encoder']['state_init'].lower()),
            rnn_reinit_state=raw_data['encoder']['reinit_state'],
            rnn_detach_state=raw_data['encoder']['detach_state'],

            dense_n_layers=raw_data['decoder']['n_layers'],
            dense_n_units=raw_data['decoder']['n_units'],
            dense_activation=raw_data['decoder']['activation'],
            dense_dropout=raw_data['decoder']['dropout'],
            dense_init=_gluon_common.get_init(raw_data['decoder']['init'].lower()),
            dense_prefix=raw_data['decoder']['prefix'],
        )
