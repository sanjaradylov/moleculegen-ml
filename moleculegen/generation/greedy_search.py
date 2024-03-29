"""
SMILES samplers to generate new molecules using various greedy search methods.

Classes
-------
GreedySearch
    Generate new SMILES strings using greedy search methods.
"""

__all__ = (
    'GreedySearch',
)


import inspect
import warnings
from typing import Any, Callable, List

from mxnet import context, np, npx
from mxnet.random import multinomial

from ..base import Token
from ..data.vocabulary import SMILESVocabulary


def _nd_multinomial(probabilities: np.ndarray) -> int:
    sample = multinomial(probabilities.as_nd_ndarray())
    sample = sample.as_np_ndarray().item()
    return sample


def _mxnet_ctx(
        array: np.ndarray,
        ctx: context.Context = context.cpu()
) -> np.ndarray:
    return array.as_in_ctx(ctx)


def _numpy_data(token_id: List[int]) -> np.ndarray:
    return np.array(token_id).astype(int).reshape(1, 1)


# noinspection PyUnresolvedReferences
_numpy_softmax = npx.softmax


class GreedySearch:
    """Generate new SMILES strings using greedy search methods. Basically, the
    strategy is to search for the next token with the highest conditional
    probability. By default, we implement softmax with temperature followed
    by sampling from multinomial distribution.

    Each parameter is an optional callable. For mxnet backend, the defaults
    are appropriate for use, so one does not have to reimplement anything.
    But using another backend (e.g. tensorflow), one must carefully redefine
    each callable.

    Parameters
    ----------
    data_type : callable, list of int -> any, default _numpy_data
        A function that accepts a list of token ID and converts it to the main
        backend ndarray. By default, we use mxnet.np.ndarray.
    normalizer : callable, any, float -> any, default _numpy_softmax
        A function that accepts model's outputs `data` and a sensitivity
        parameter `temperature`, and returns probabilities. By default, we use
        mxnet's softmax.
    distribution : callable, any -> int, default _nd_multinomial
        A function that accepts an ndarray of probabilities and returns the ID
        of a sampled token. By default, we sample from an
        mxnet.numpy-compatible multinomial distribution.
    ctx : callable, any -> any, default _mxnet_ctx
        A function that accepts an ndarray, which is transferred into a
        specified context (CPU/GPU/TPU). By default, we explicitly specify an
        mxnet.context.Context instance.

    Notes
    -----
    Here, the `any` type denotes an ndarray type of a desired backend, e.g.
    mxnet.np.ndarray or tensorflow.Tensor.
    See also the docs for the `__call__` method.
    """

    def __init__(
            self,
            data_type: Callable = _numpy_data,
            normalizer: Callable = _numpy_softmax,
            distribution: Callable = _nd_multinomial,
            ctx: Callable = _mxnet_ctx,
    ):
        warnings.warn(
            message=(
                f'{self.__class__.__name__} is deprecated; '
                f'wil be removed in 1.1.0.'
                f'consider `moleculegen.generation.SoftmaxSearch` instead.'
            ),
            category=DeprecationWarning,
        )
        if not all(
                hasattr(obj, '__call__') for obj in (
                    data_type, normalizer, distribution, ctx,
                )
        ):
            raise TypeError(
                'Every parameter must be callable; see the documentation for '
                'the detailed description of the formal parameters.'
            )

        normalizer_args = inspect.getfullargspec(normalizer).args
        for arg in 'data', 'temperature':
            if arg not in normalizer_args:
                raise ValueError(
                    f'`normalizer` must accept parameter `{arg}`; see the '
                    f'documentation for the detailed description of the '
                    f'formal parameters.'
                )

        self._distribution = distribution
        self._data_type = data_type
        self._normalizer = normalizer
        self._ctx = ctx

    def __call__(
            self,
            model: Any,
            states: List,
            vocabulary: SMILESVocabulary,
            prefix: str = Token.BOS,
            max_length: int = 80,
            temperature: float = 1.0,
    ):
        """Generate SMILES string using (learned) model weights and states.

        Parameters
        ----------
        model : any
            A (trained) language model, e.g. mxnet.gluon.Block or
            tf.keras.Model instance.
        states : list
            `model`s hidden states. A list of mxnet.np.ndarray or another
            backend type.
        vocabulary : Vocabulary
            The Vocabulary instance, which provides id-to-token conversion.
        prefix : str, default: Token.BOS
            The prefix of a SMILES string to generate
        max_length : int, default: 80
            The maximum number of tokens to generate.
        temperature : float, default 1.0
            A sensitivity parameter.

        Returns
        -------
        smiles : str
            The SMILES string generated by the model.
        """
        # Tokenize sequence correctly.
        tokens: List[str] = Token.tokenize(prefix)
        # Save predicted token IDs.
        token_ids: List[int] = [vocabulary[tokens[0]]]

        # Update `states` on prefix tokens (no prediction is made).
        for token in tokens[1:]:
            token_id = self._ctx(self._data_type([token_ids[-1]]))
            _, states = model(token_id, states)
            token_ids.append(vocabulary[token])

        # Predict the next token IDs.
        for n_iter in range(max_length):
            inputs = self._ctx(self._data_type([token_ids[-1]]))
            outputs, states = model(inputs, states)

            # Normalize outputs to get probabilities (e.g. using softmax).
            probabilities = self._normalizer(
                data=outputs[0],
                temperature=temperature,
            )

            # Sample from a distribution (e.g. multinomial).
            next_token_id = self._distribution(probabilities)

            # Interrupt, if End-of-SMILES token is generated.
            token = vocabulary.idx_to_token[next_token_id]
            if token in Token.EOS:
                break

            token_ids.append(next_token_id)

        smiles = ''.join(vocabulary.idx_to_token[id_] for id_ in token_ids)
        # Remove special tokens.
        smiles = Token.crop(smiles)

        return smiles
