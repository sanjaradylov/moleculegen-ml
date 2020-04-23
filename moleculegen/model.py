"""
Language models for generation of novel molecules.

Classes
-------
SMILESRNNModel
    Recurrent neural network to encode-decode SMILES strings.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

from mxnet import context, gluon, nd, np, npx

from .base import Token
from .vocab import Vocabulary


def _distribution_partial(
        distribution: Callable,
        **distribution_args: Any,
) -> Callable:
    """Return distribution callable `distribution` with arbitrary
    (non-default) arguments `distribution_args` specified in advance. Use
    primarily for state initialization.

    Parameters
    ----------
    distribution : callable
        One of the distribution functions from mxnet.nd.random.
    distribution_args : dict, default None
        Parameters of mxnet.nd.random.uniform excluding `shape`.

    Returns
    -------
    func : callable
        Partial distribution callable.

    Raises
    ------
    ValueError
        If `shape` parameter is included in `distribution_args`.
        This parameter will be used separately in state initialization.
    """
    if 'shape' in distribution_args:
        raise ValueError('`shape` parameter should be not be specified.')

    return partial(distribution, **distribution_args)


class OneHotEncoder:
    """One-hot encoder class. It is implemented as a functor for more
    convenience, to pass it as a detached embedding layer.

    Parameters
    ----------
    depth : int
        The depth of one-hot encoding.
    """

    def __init__(self, depth: int):
        self.depth = depth

    def __call__(self, indices: np.ndarray, *args, **kwargs) -> np.ndarray:
        """Return one-hot encoded tensor.

        Parameters
        ----------
        indices : np.ndarray
            The indices (categories) to encode.
        *args, **kwargs
            Additional arguments for `nd.one_hot`.
        """
        return npx.one_hot(indices, self.depth, *args, **kwargs)


class SMILESRNNModel(gluon.Block):
    """Recurrent neural network generating SMILES strings of novel molecules.
    Follows gluon's Block API.

    Parameters
    ----------
    embedding_layer : gluon.nn.Embedding or OneHotEncoder
        Embedding layer.
    rnn_layer : gluon.rnn._RNNLayer
        Recurrent layer.
    dense_layer : gluon.nn.Dense or gluon.nn.Sequential
        Dense layer.
    kwargs
        Block parameters.
    """

    def __init__(
            self,
            embedding_layer: Union[
                gluon.nn.Embedding,
                OneHotEncoder,
            ],
            rnn_layer: Union[
                gluon.rnn.GRU,
                gluon.rnn.LSTM,
                gluon.rnn.RNN,
            ],
            dense_layer: Union[
                gluon.nn.Dense,
                gluon.nn.Sequential,
            ],
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.embedding_layer = embedding_layer
        self.rnn_layer = rnn_layer
        self.dense_layer = dense_layer

    def begin_state(
            self,
            batch_size: int = 0,
            **kwargs,
    ) -> List[np.ndarray]:
        """Return initial state for each element in mini-batch.

        Parameters
        ----------
        batch_size : int, default 0
            Batch size.
        **kwargs
            Additional arguments for RNN layer's `begin_state` method including
            callable `func` argument for creating initial state.

        Returns
        -------
        state : list
            Initial state.
        """
        return self.rnn_layer.begin_state(batch_size=batch_size, **kwargs)

    def forward(
            self,
            inputs: np.ndarray,
            state: List[np.ndarray],
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Run forward computation.

        Parameters
        ----------
        inputs : mxnet.np.ndarray, shape = (batch_size, n_steps)
            Input samples.
        state : list
            Hidden state.

        Returns
        -------
        output : tuple
            X : mxnet.np.ndarray, shape = (n_steps, batch_size, vocab_size)
                Output at current step.
            H : list
                Hidden state output.
        """
        inputs_e = self.embedding_layer(inputs.T)
        outputs, state = self.rnn_layer(inputs_e, state)
        outputs = self.dense_layer(outputs.reshape((-1, outputs.shape[-1])))
        return outputs, state

    def generate(
            self,
            vocabulary: Vocabulary,
            prefix: str = Token.BOS,
            max_length: int = 100,
            ctx: context.Context = context.cpu(),
            state_init_func: Optional[Callable] = None,
    ) -> str:
        """Generate SMILES string using (learned) model weights and states.

        Parameters
        ----------
        vocabulary : Vocabulary
            The Vocabulary instance, which provides id-to-token conversion.
        prefix : str, default: Token.BOS
            The prefix of a SMILES string to generate
        max_length : int, default: 100
            The maximum number of tokens to generate.
        ctx : context.Context, default: context.cpu()
            CPU or GPU.
        state_init_func : callable, default nd.random.uniform(-0.1, 0.1)
            Callable for state initialization.

        Returns
        -------
        smiles : str
            The SMILES string generated by the model.
        """
        state_init_func = state_init_func or _distribution_partial(
            nd.random.uniform,
            low=-0.2,
            high=0.2,
            ctx=ctx,
        )
        state = self.begin_state(batch_size=1, func=state_init_func, ctx=ctx)
        smiles = prefix

        for n_iter in range(max_length):
            input_ = np.array(vocabulary[smiles[-1]], ctx=ctx).reshape(1, 1)
            output, state = self.forward(input_, state)

            token_id = npx.softmax(output).argmax().astype(np.int32).item()
            token = vocabulary.idx_to_token[token_id]
            if token in (Token.EOS, Token.PAD):
                break

            smiles += token

        return smiles.lstrip(Token.BOS)
