__all__ = (
    'masked_softmax',

    'DotProductAttention',
    'FixedPositionalEncoding',
    'LearnedPositionalEmbedding',
    'MultiHeadSelfAttention',
    'PositionWiseFFN',
    'TransformerDecoderBlock',

    'SMILESTransformer',
)

from typing import Optional, Tuple, Union

import mxnet as mx

from .._types import ActivationT, ContextT, InitializerT
from .base import SMILESLM


LARGE_NEGATIVE = -1e7


class SMILESTransformer(SMILESLM):
    def __init__(
            self,
            vocab_size: int,
            *,

            embedding_dim: int = 64,
            embedding_dropout: float = 0.4,
            embedding_init: InitializerT = mx.init.Xavier(),

            att_n_layers: int = 4,
            att_n_heads: int = 4,
            att_n_units: int = 64,
            att_dropout: float = 0.25,
            att_an_dropout: float = 0.1,
            att_ffn_n_units: int = 128,
            att_ffn_activation: ActivationT = 'relu',
            att_init: InitializerT = mx.init.Xavier(),

            decoder_dropout: float = 0.1,
            decoder_init: InitializerT = mx.init.Xavier(),

            initialize: bool = True,
            dtype: Optional[str] = 'float32',
            ctx: Optional[ContextT] = None,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(ctx=ctx, prefix=prefix, params=params)

        with self.name_scope():
            self._embedding = LearnedPositionalEmbedding(
                input_dim=vocab_size,
                output_dim=embedding_dim,
                n_steps=64,
                dropout=embedding_dropout,
                axes=1,
                dtype=dtype,
            )
            if initialize:
                self._embedding.initialize(init=embedding_init, ctx=ctx)

            self._encoder = mx.gluon.nn.Sequential()
            for _ in range(att_n_layers):
                self._encoder.add(TransformerDecoderBlock(
                    att_n_heads=att_n_heads,
                    att_n_qkv_units=att_n_units,
                    att_n_output_units=embedding_dim,
                    att_dropout=att_dropout,
                    an_dropout=att_an_dropout,
                    ffn_n_units=att_ffn_n_units,
                    ffn_n_outputs=embedding_dim,
                    ffn_activation=att_ffn_activation,
                ))
            self._encoder.cast(dtype)
            if initialize:
                self._encoder.initialize(init=att_init, ctx=ctx)

            decoder = mx.gluon.nn.Dense(vocab_size, flatten=False, dtype=dtype)
            if decoder_dropout != 0.:
                self._decoder = mx.gluon.nn.Sequential()
                self._decoder.add(mx.gluon.nn.Dropout(decoder_dropout))
                self._decoder.add(decoder)
            else:
                self._decoder = decoder
            if initialize:
                self._decoder.initialize(init=decoder_init, ctx=ctx)

    @property
    def embedding(self) -> 'LearnedPositionalEmbedding':
        return self._embedding

    @property
    def encoder(self) -> mx.gluon.nn.Sequential:
        return self._encoder

    @property
    def decoder(self) -> Union[mx.gluon.nn.Dense, mx.gluon.nn.Sequential]:
        return self._decoder

    # noinspection PyMethodOverriding
    def forward(self, batch) -> mx.np.ndarray:
        x = self._embedding(batch.inputs)
        for attention_cell in self._encoder:
            x = attention_cell(x, batch.valid_lengths)
        return self._decoder(x)


class TransformerDecoderBlock(mx.gluon.Block):
    def __init__(
            self,
            att_n_heads: int,
            att_n_qkv_units: int,
            att_n_output_units: int,
            att_dropout: float,
            an_dropout: float,
            ffn_n_units: int,
            ffn_n_outputs: int,
            ffn_activation: ActivationT = 'relu',

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.attention = MultiHeadSelfAttention(
            n_heads=att_n_heads,
            n_qkv_units=att_n_qkv_units,
            n_output_units=att_n_output_units,
            dropout=att_dropout,
        )
        self.add_norm_1 = AddNorm(an_dropout)
        self.ffn = PositionWiseFFN(
            n_units=ffn_n_units,
            n_outputs=ffn_n_outputs,
            activation=ffn_activation,
        )
        self.add_norm_2 = AddNorm(an_dropout)

    def forward(
            self,
            x: mx.np.ndarray,
            valid_lengths: Optional[mx.np.ndarray] = None,
    ) -> mx.np.ndarray:
        y = self.attention(x, valid_lengths)  # shape=(b, #q, e)
        y = self.add_norm_1(x, y)             # shape=(b, #q, e)
        z = self.ffn(y)                       # shape=(b, #q, e)
        return self.add_norm_2(y, z)          # shape=(b, #q, e)


class MultiHeadSelfAttention(mx.gluon.Block):
    """Multi-head dot-product attention pooling.

    Parameters
    ----------
    n_heads : int
        The number of attention heads.
    n_qkv_units : int
        The linear projection dimension.
    n_output_units : int
    dropout : float
        The dropout rate of dot-product attention weights.

    Other Parameters
    ----------------
    prefix : str, default=None
    params : mxnet.gluon.ParameterDict, default=None

    Notes
    -----
    To encourage parallel computation, specify `n_units` to be `n_heads * embedding_dim`.
    """

    def __init__(
            self,
            n_heads: int,
            n_qkv_units: int,
            n_output_units: int,
            dropout: float = 0.,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.attention = DotProductAttention(dropout)
        self._n_heads = n_heads

        n_units = n_heads * n_qkv_units
        self.query_t = mx.gluon.nn.Dense(n_units, use_bias=False, flatten=False)
        self.key_t = mx.gluon.nn.Dense(n_units, use_bias=False, flatten=False)
        self.value_t = mx.gluon.nn.Dense(n_units, use_bias=False, flatten=False)

        self.output_t = mx.gluon.nn.Dense(n_output_units, use_bias=True, flatten=False)

    @staticmethod
    def _transpose_qkv(
            x: mx.np.ndarray,
            num_heads: int,
    ) -> mx.np.ndarray:
        x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(-1, x.shape[2], x.shape[3])

    @staticmethod
    def _transpose_output(
            x: mx.np.ndarray,
            num_heads: int,
    ) -> mx.np.ndarray:
        x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(x.shape[0], x.shape[1], -1)

    def forward(
            self,
            x: mx.np.ndarray,
            valid_lengths: Optional[mx.np.ndarray] = None,
    ) -> mx.np.ndarray:
        """Run forward computation.

        Parameters
        ----------
        x : mxnet.np.ndarray, shape=(b, #q, e)
        valid_lengths : mxnet.np.ndarray, shape=(b,) or (b, #q), default=None

        Returns
        -------
        mxnet.np.ndarray, shape=(b, #q, `self._n_units`)

        Notes
        -----
        b=batch size, e=embedding dim, q=queries
        """
        # shape=(b * self_n_heads, #q, e / self._n_heads)
        queries = self._transpose_qkv(self.query_t(x), self._n_heads)
        keys = self._transpose_qkv(self.key_t(x), self._n_heads)
        values = self._transpose_qkv(self.value_t(x), self._n_heads)

        if valid_lengths is not None:
            # shape=(b, self._n_heads)
            valid_lengths = valid_lengths.repeat(self._n_heads, axis=0)

        # shape=(b * self._n_heads, #q, e / self._n_heads)
        output = self.attention(queries, keys, values, valid_lengths)
        # shape=(b, #q, e)
        output_concat = self._transpose_output(output, self._n_heads)
        # shape=(b, #q, self._n_units)
        return self.output_t(output_concat)


class DotProductAttention(mx.gluon.Block):
    """Scaled dot-product attention pooling. Optionally, apply dropout on attention
    weights.

    .. math::

        w = softmax(\frac{QK^T}{\sqrt{d}})V

    Parameters
    ----------
    dropout : float, default=0.
    prefix : str, default=None
    """

    def __init__(self, dropout: float = 0., prefix: Optional[str] = None):
        super().__init__(prefix=prefix, params=None)

        self.dropout = mx.gluon.nn.Dropout(dropout) if dropout > 0. else lambda x: x

    def forward(
            self,
            queries: mx.np.ndarray,
            keys: mx.np.ndarray,
            values: mx.np.ndarray,
            valid_lengths: Optional[mx.np.ndarray] = None,
    ) -> mx.np.ndarray:
        """Run forward computation.

        Parameters
        ----------
        queries : mxnet.np.ndarray, shape=(b, #q, e)
        keys : mxnet.np.ndarray, shape=(b, #k-v, e)
        values : mxnet.np.ndarray, shape=(b, #k-v, v)
        valid_lengths : mxnet.np.ndarray, shape=(b,) or (b, #q), default=None

        Returns
        -------
        mxnet.np.ndarray, shape=(b, #q, v)

        Notes
        -----
        b=batch size, e=embedding dim, q=queries, k-v=key-value pairs, v=value dim
        """
        # noinspection PyUnresolvedReferences
        batch_dot = mx.npx.batch_dot

        scale = queries.shape[-1] ** 0.5
        scores = batch_dot(queries, keys, transpose_b=True)/scale  # shape=(b, #q, #k-v)

        if valid_lengths is not None:
            valid_lengths = (
                valid_lengths
                .reshape(queries.shape[0], 1)
                .repeat(queries.shape[1], axis=1)
            )
            bq_valid_lengths = mx.np.tile(
                mx.np.arange(
                    1, queries.shape[1] + 1,
                    ctx=queries.ctx,
                    dtype=valid_lengths.dtype,
                ),
                (queries.shape[0], 1),
            )
            # shape=(b, #q)
            bq_valid_lengths = mx.np.minimum(bq_valid_lengths, valid_lengths)
        else:
            bq_valid_lengths = None

        weights = masked_softmax(scores, bq_valid_lengths)  # shape=(b, #q, #k-v)
        weights = self.dropout(weights)                     # shape=(b, #q, #k-v)
        return batch_dot(weights, values)                   # shape=(b, #q, v)


class AddNorm(mx.gluon.nn.Block):
    """Residual connection followed by layer normalization.

    Parameters
    ----------
    dropout : float, default=0.
        The dropout rate of the transformed inputs.

    Other Parameters
    ----------------
    prefix : str, default=None
    params : mxnet.gluon.ParameterDict, default=None
    """

    def __init__(
            self,
            dropout: float = 0.,

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.dropout = mx.gluon.nn.Dropout(dropout) if dropout > 0. else lambda x: x
        self.layer_norm = mx.gluon.nn.LayerNorm()

    def forward(self, x, x_t):
        return self.layer_norm(x + self.dropout(x_t))


class FixedPositionalEncoding(mx.gluon.nn.Block):
    """Encode positional sequence information.

    Parameters
    ----------
    embedding_dim : int
    max_len : int, default=100
    dtype : str or numpy.dtype, default='float32'

    Other Parameters
    ----------------
    prefix : str, default=None
    """

    def __init__(
            self,
            embedding_dim: int,
            max_len: int = 100,
            dtype: str = 'float32',

            prefix: Optional[str] = None,
    ):
        super().__init__(prefix=prefix, params=None)

        self.matrix = mx.np.zeros((1, max_len, embedding_dim), dtype=dtype)

        powers = mx.np.power(
            10000,
            mx.np.arange(0, embedding_dim, 2, dtype=dtype) / embedding_dim,
        )
        x = mx.np.arange(max_len, dtype=dtype).reshape(-1, 1) / powers
        self.matrix[:, :, 0::2] = mx.np.sin(x)
        if embedding_dim % 2 == 0:
            self.matrix[:, :, 1::2] = mx.np.cos(x)
        else:
            self.matrix[:, :, 1::-1] = mx.np.cos(x[:, :-1])

    def forward(self, x: mx.np.ndarray) -> mx.np.ndarray:
        return x * x.shape[-1]**.5 + self.matrix[:, :x.shape[1], :].as_in_ctx(x.ctx)


class LearnedPositionalEmbedding(mx.gluon.nn.Block):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            n_steps: int,
            dropout: float = 0.,
            axes: Union[int, Tuple[int]] = (),
            dtype: str = 'float32',

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.token_embedding = mx.gluon.nn.Embedding(input_dim, output_dim, dtype)
        self.position_embedding = mx.gluon.nn.Embedding(n_steps, output_dim, dtype)
        self.dropout = mx.gluon.nn.Dropout(dropout, axes)
        self._dtype = dtype

    def forward(self, x: mx.np.ndarray) -> mx.np.ndarray:
        positions = (
            mx.np.arange(0, x.shape[1], dtype='int32', ctx=x.ctx)
            .reshape(x.shape[1], 1)
            .repeat(x.shape[0], 1)
        )
        x = self.token_embedding(x)
        x = x * x.shape[-1]**0.5 + self.position_embedding(positions).swapaxes(0, 1)
        return self.dropout(x)


class PositionWiseFFN(mx.gluon.nn.Block):
    """Transform all sequence positions using a two-layer perceptron.

    Parameters
    ----------
    n_units : int
        The number of hidden layer neurons.
    n_outputs : int
        The number of output layer neurons.
    activation : {'sigmoid', 'tanh', 'relu', 'softrelu', 'softsign'}
            or mxnet.gluon.nn.Activation
            or callable, mxnet.nd.NDArray -> mxnet.nd.NDArray
            or None,
            default='relu'
        The activation function of the hidden layer.

    Other Parameters
    ----------------
    prefix : str, default=None
    params : mxnet.gluon.ParameterDict, default=None
    """

    def __init__(
            self,
            n_units: int,
            n_outputs: int,
            activation: ActivationT = 'relu',

            prefix: Optional[str] = None,
            params: Optional[mx.gluon.ParameterDict] = None,
    ):
        super().__init__(prefix=prefix, params=params)

        self.net = mx.gluon.nn.Sequential()
        self.net.add(mx.gluon.nn.Dense(n_units, flatten=False, activation=activation))
        self.net.add(mx.gluon.nn.Dense(n_outputs, flatten=False))

    def forward(self, x: mx.np.ndarray) -> mx.np.ndarray:
        return self.net(x)


def masked_softmax(
        x: mx.np.ndarray,
        valid_lengths: Optional[mx.np.ndarray] = None,
) -> mx.np.ndarray:
    """For every entry `x[i]`, perform softmax only on first `valid_lengths[i]` elements.

    Parameters
    ----------
    x : mxnet.np.ndarray, shape=(batch size, embedding dim, #queries)
    valid_lengths : mxnet.np.ndarray, default=None,
            shape=(batch size,) or shape=(batch size, #queries)

    Returns
    -------
    mxnet.np.ndarray, shape=(batch size, embedding dim, #queries)
    """
    # noinspection PyUnresolvedReferences
    softmax = mx.npx.softmax
    if valid_lengths is None:
        return softmax(x)
    else:
        shape = x.shape
        if valid_lengths.ndim == 1:
            valid_lens = valid_lengths.repeat(shape[1])
        else:
            valid_lens = valid_lengths.reshape(-1)
        # noinspection PyUnresolvedReferences
        x = mx.npx.sequence_mask(
            x.reshape(-1, shape[-1]), valid_lens, True, value=LARGE_NEGATIVE, axis=1)
        return softmax(x).reshape(shape)
