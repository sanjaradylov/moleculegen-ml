"""
Generative language models.

Classes
-------
SMILESEncoderDecoder
    A generative recurrent neural network to encode-decode SMILES strings.
SMILESEncoderDecoderFineTuner
    The fine-tuner of SMILESEncoderDecoder model.
"""

__all__ = (
    'SMILESEncoderDecoder',
    'SMILESEncoderDecoderFineTuner',
)

import json
from typing import Optional, Union

import mxnet as mx
from mxnet import gluon

from . import _gluon_common
from .base import SMILESEncoderDecoderABC
from ..description.common import OneHotEncoder


class SMILESEncoderDecoder(SMILESEncoderDecoderABC):
    """A generative recurrent neural network to encode-decode SMILES strings.

    Parameters
    ----------
    vocab_size : int
        The vocabulary dimension, which will indicate the number of output
        neurons of a decoder.
    initialize : bool, default True
        Whether to initialize model parameters.
        When one decides to load parameters from a file, deferred
        initialization is needless.
    use_one_hot : bool, default False
        Whether to use one-hot-encoding or an embedding layer.
    embedding_dim : int, default 4
        The output dimension of an embedding layer.
    embedding_init : str or mxnet.init.Initializer,
            default mxnet.init.Orthogonal()
        The parameter initializer of an embedding layer.
    embedding_prefix : str, default 'embedding_'
        The prefix of an embedding block.
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
    rnn_prefix : str, default 'encoder_'
        The prefix of an encoder block.
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
    dense_prefix : str, default 'decoder_'
        The prefix of a decoder block.
    tie_weights : bool, default False
        Whether to share the embedding block parameters w/ a decoder block.
    dtype : str, default 'float32'
        Data type.
    ctx : mxnet.context.Context, default mxnet.context.cpu()
        CPU or GPU.

    prefix : str, default None
    params : mxnet.gluon.ParameterDict, default None

    Attributes
    ----------
    ctx : mxnet.context.Context
        The model's context.
    embedding : OneHotEncoder or mxnet.gluon.nn.Embedding
        An embedding layer.
    encoder : mxnet.gluon.rnn.RNN or mxnet.gluon.rnn.LSTM
            or mxnet.gluon.rnn.GRU
        An RNN encoder.
    decoder : mxnet.gluon.nn.Dense or mxnet.gluon.nn.Sequential
        A Feed-Forward NN decoder.
    """

    def __init__(
            self,
            vocab_size: int,
            initialize: bool = True,
            use_one_hot: bool = False,
            embedding_dim: int = 4,
            embedding_dropout: float = 0.,
            embedding_init: Optional[
                Union[str, mx.init.Initializer]] = mx.init.Uniform(),
            embedding_prefix: str = 'embedding_',
            rnn: str = 'lstm',
            n_rnn_layers: int = 1,
            n_rnn_units: int = 64,
            rnn_dropout: float = 0.,
            rnn_init: Optional[Union[str, mx.init.Initializer]] = mx.init.Orthogonal(),
            rnn_prefix: str = 'encoder_',
            n_dense_layers: int = 1,
            n_dense_units: int = 128,
            dense_activation: str = 'relu',
            dense_dropout: float = 0.,
            dense_init: Optional[Union[str, mx.init.Initializer]] = mx.init.Xavier(),
            dense_prefix: str = 'decoder_',
            tie_weights: bool = False,
            dtype: Optional[str] = 'float32',
            *,
            ctx: mx.context.Context = mx.context.cpu(),
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

        if n_dense_layers < 1:
            raise ValueError(
                'The number of dense layers must be positive non-zero.'
            )

        if (
                tie_weights
                and (
                    embedding_dim != n_rnn_units
                    or
                    n_dense_layers > 1
                    and embedding_dim != n_dense_units
                )
        ):
            raise ValueError(
                f'When sharing weights, the number of hidden units must be equal to '
                f'the embedding dimension.'
            )

        # Initialize mxnet.gluon.Block parameters.
        super().__init__(ctx=ctx, prefix=prefix, params=params)

        with self.name_scope():
            # Define (and initialize) an embedding layer.
            if use_one_hot:
                self._embedding = OneHotEncoder(vocab_size)
            else:
                embedding_block = gluon.nn.Embedding(
                    input_dim=vocab_size,
                    output_dim=embedding_dim,
                    dtype=dtype,
                    prefix=embedding_prefix,
                )

                if embedding_dropout > 1e-3:
                    seq_prefix = f'{embedding_prefix.rstrip("_")}seq_'
                    self._embedding = gluon.nn.HybridSequential(prefix=seq_prefix)
                    self._embedding.add(embedding_block)
                    self._embedding.add(gluon.nn.Dropout(embedding_dropout))
                    shared_params = self._embedding[0].params if tie_weights else None
                else:
                    self._embedding = embedding_block
                    shared_params = self._embedding.params if tie_weights else None

                if initialize:
                    self._embedding.initialize(init=embedding_init, ctx=ctx)

            # Select and initialize a recurrent block.
            self._encoder = _gluon_common.RNN_MAP[rnn](
                hidden_size=n_rnn_units,
                num_layers=n_rnn_layers,
                dropout=rnn_dropout,
                dtype=dtype,
                prefix=rnn_prefix,
            )
            if initialize:
                self._encoder.initialize(init=rnn_init, ctx=ctx)

            # Define and initialize a dense layer(s).
            self._decoder = _gluon_common.mlp(
                n_layers=n_dense_layers,
                n_units=n_dense_units,
                activation=dense_activation,
                output_dim=vocab_size,
                dtype=dtype,
                dropout=dense_dropout,
                prefix=dense_prefix,
                params=shared_params,
            )
            if initialize:
                self._decoder.initialize(init=dense_init, ctx=ctx)

    @property
    def embedding(self) -> Union[OneHotEncoder, gluon.nn.Embedding]:
        """Return the embedding layer.
        """
        return self._embedding

    @property
    def encoder(self) -> Union[gluon.rnn.RNN, gluon.rnn.LSTM, gluon.rnn.GRU]:
        """Return the RNN encoder.
        """
        return self._encoder

    @property
    def decoder(self) -> Union[gluon.nn.Dense, gluon.nn.Sequential]:
        """Return the Feed-Forward NN decoder.
        """
        return self._decoder

    @classmethod
    def from_config(cls, config_file: str) -> 'SMILESEncoderDecoder':
        """Instantiate a model loading formal parameters from a JSON file `config_file`.

        config_file : str
            A JSON file to load formal parameters from.

        model : SMILESEncoderDecoder
        """
        with open(config_file) as fh:
            raw_data = json.load(fh)

        return cls(
            vocab_size=raw_data['vocab_size'],
            initialize=raw_data['initialize'],
            tie_weights=raw_data['tie_weights'],
            dtype=raw_data['dtype'],
            ctx=_gluon_common.CTX_MAP[raw_data['ctx'].lower()],
            prefix=raw_data['prefix'],

            use_one_hot=raw_data['embedding']['use_one_hot'],
            embedding_dim=raw_data['embedding']['dim'],
            embedding_dropout=raw_data['embedding']['dropout'],
            embedding_init=_gluon_common.INIT_MAP[raw_data['embedding']['init'].lower()],
            embedding_prefix=raw_data['embedding']['prefix'],

            rnn=raw_data['encoder']['rnn'],
            n_rnn_layers=raw_data['encoder']['n_layers'],
            n_rnn_units=raw_data['encoder']['n_units'],
            rnn_dropout=raw_data['encoder']['dropout'],
            rnn_init=_gluon_common.INIT_MAP[raw_data['encoder']['init'].lower()],
            rnn_prefix=raw_data['encoder']['prefix'],

            n_dense_layers=raw_data['decoder']['n_layers'],
            n_dense_units=raw_data['decoder']['n_units'],
            dense_activation=raw_data['decoder']['activation'],
            dense_dropout=raw_data['decoder']['dropout'],
            dense_init=_gluon_common.INIT_MAP[raw_data['decoder']['init'].lower()],
            dense_prefix=raw_data['decoder']['prefix'],
        )

    @classmethod
    def load_fine_tuner(
            cls,
            path: str,
            update_features: bool = True,
            decoder_init: Optional[Union[str, mx.init.Initializer]] = mx.init.Xavier(),
    ) -> 'SMILESEncoderDecoder':
        """Create a new fine-tuner model: load model configuration and parameters, and
        initialize decoder weights.

        Parameters
        ----------
        path : str
            The path to the directory of model configuration and parameters.
            path/config.json - the formal parameters of a model;
            path/weights.params - the parameters of a model.
        update_features : bool, default True
            Whether to update embedding and encoder parameters during training.
        decoder_init : str or mxnet.init.Initializer, default None
            A decoder initializer.

        Returns
        -------
        model : SMILESEncoderDecoder
        """
        model = cls.from_config(f'{path}/config.json')
        model.load_parameters(f'{path}/weights.params', ctx=model.ctx)

        if not update_features:
            model.embedding.collect_params().setattr('grad_req', 'null')
            model.encoder.collect_params().setattr('grad_req', 'null')

        model.decoder.initialize(init=decoder_init, force_reinit=True, ctx=model.ctx)

        return model


class SMILESEncoderDecoderFineTuner(SMILESEncoderDecoderABC):
    """The fine-tuner of SMILESEncoderDecoder model. Loads embedding and encoder blocks,
    and trains a new decoder block.

    Parameters
    ----------
    model : SMILESEncoderDecoder
        An encoder-decoder model to fine-tune.
    output_dim : int
        The number of output neurons.
    initialize : bool, default True
        Whether to initialize decoder's parameters.
    update_features : bool, default True
        Whether to update embedding and encoder parameters during training.
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
    dense_prefix : str, default 'decoder_'
        The prefix of a decoder block.
    dtype : str, default 'float32'
        Data type.
    ctx : mxnet.context.Context, default mxnet.context.cpu()
        CPU or GPU.

    prefix : str, default None
    params : mxnet.gluon.ParameterDict, default None

    Attributes
    ----------
    ctx : mxnet.context.Context
        The model's context.
    embedding : OneHotEncoder or mxnet.gluon.nn.Embedding
        An embedding layer.
    encoder : mxnet.gluon.rnn.RNN or mxnet.gluon.rnn.LSTM
            or mxnet.gluon.rnn.GRU
        An RNN encoder.
    decoder : mxnet.gluon.nn.Dense or mxnet.gluon.nn.Sequential
        A Feed-Forward NN decoder.
    """

    def __init__(
            self,
            model: SMILESEncoderDecoder,
            output_dim: int,
            initialize: bool = True,
            update_features: bool = True,
            n_dense_layers: int = 1,
            n_dense_units: int = 128,
            dense_activation: str = 'relu',
            dense_dropout: float = 0.,
            dense_init: Optional[Union[str, mx.init.Initializer]] = mx.init.Xavier(),
            dense_prefix: str = 'fine_tuner_decoder_',
            dtype: Optional[str] = 'float32',
            *,
            ctx: mx.context.Context = mx.context.cpu(),
            prefix: Optional[str] = None,
            params: Optional[gluon.ParameterDict] = None,
    ):
        super().__init__(ctx=ctx, prefix=prefix, params=params)

        model.ctx = self.ctx

        self._embedding = model.embedding
        self._encoder = model.encoder

        if not update_features:
            self._embedding.collect_params().setattr('grad_req', 'null')
            self._encoder.collect_params().setattr('grad_req', 'null')

        self._decoder = _gluon_common.mlp(
            n_layers=n_dense_layers,
            n_units=n_dense_units,
            activation=dense_activation,
            output_dim=output_dim,
            dtype=dtype,
            dropout=dense_dropout,
            prefix=dense_prefix,
            params=None,
        )
        if initialize:
            self._decoder.initialize(init=dense_init, ctx=self.ctx)

    @property
    def embedding(self) -> Union[OneHotEncoder, gluon.nn.Embedding]:
        """Return the embedding layer.
        """
        return self._embedding

    @property
    def encoder(self) -> Union[gluon.rnn.RNN, gluon.rnn.LSTM, gluon.rnn.GRU]:
        """Return the RNN encoder.
        """
        return self._encoder

    @property
    def decoder(self) -> Union[gluon.nn.Dense, gluon.nn.Sequential]:
        """Return the Feed-Forward NN decoder.
        """
        return self._decoder
