"""
SMILES samplers to generate new molecules using various greedy search methods.

Classes
-------
BaseSearch
    The base SMILES sampler class.

ArgmaxSearch
    Generate new SMILES strings using greedy search (argmax) method.

SoftmaxSearch
    Softmax with a sensitivity parameter followed by sampling from multinomial
    distribution.
GumbelSoftmaxSearch
    Gumbel-Softmax with a sensitivity parameter followed by sampling from multinomial
    distribution.
"""

__all__ = (
    'ArgmaxSearch',
    'BaseSearch',
    'GumbelSoftmaxSearch',
    'SoftmaxSearch',
)


import functools
from typing import Callable, List, Optional

import mxnet as mx

from ..base import Token
from ..data.vocabulary import SMILESVocabulary
from ..estimation.base import SMILESEncoderDecoderABC


class BaseSearch:
    """The base SMILES sampler class.

    Parameters
    ----------
    model : SMILESEncoderDecoderABC
        A (trained) language model. Generates one token ID in one step.
    vocabulary : SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str
        The prefix of a SMILES string to generate.
    max_length : int
        The maximum number of tokens in the SMILES string being generated.
    state_initializer : callable, any -> mx.nd.NDArray
        The model's hidden state initializer function (e.g. mx.nd.ones).
    normalizer : callable, any -> mx.np.ndarray
        A function that accepts model's outputs (logits) and returns probabilities.
    distribution : callable, any -> int
        A function that accepts an ndarray of probabilities and returns the ID
        of a sampled token.
    """

    def __init__(
            self,
            model: SMILESEncoderDecoderABC,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str,
            max_length: int,
            state_initializer: Callable[..., mx.nd.NDArray],
            normalizer: Callable,
            distribution: Callable,
    ):
        self._model = model
        self._vocabulary = vocabulary
        self._normalizer = normalizer
        self._distribution = distribution

        self.prefix = prefix
        self.max_length = max_length
        self.state_initializer = state_initializer

    @property
    def prefix(self) -> str:
        return self.__prefix

    @prefix.setter
    def prefix(self, prefix: str):
        """Set new `prefix`. Check for validity using moleculegen's available tokens.

        Parameters
        ----------
        prefix : str
            SMILES prefix of length >= 1.

        Raises
        ------
        ValueError
            If there is an invalid token in `prefix`.
        """
        all_tokens = Token.get_all_tokens()
        prefix_tokens: List[str] = Token.tokenize(prefix)

        for token in prefix_tokens:
            if token not in all_tokens:
                raise ValueError(f'token {token!r} is invalid')
        else:
            self.__prefix = prefix

    @property
    def max_length(self) -> int:
        return self.__max_length

    @max_length.setter
    def max_length(self, max_length: int):
        """Set new `max_length`. Must be greater than 1.

        Parameters
        ----------
        max_length : int

        Raises
        ------
        ValueError
            If `max_length` < 2
        """
        if max_length < 2:
            raise ValueError('max_length must be greater than 1')

        self.__max_length = max_length

    @property
    def model(self) -> SMILESEncoderDecoderABC:
        return self._model

    @property
    def vocabulary(self) -> SMILESVocabulary:
        return self._vocabulary

    @property
    def normalizer(self) -> Callable[..., mx.np.ndarray]:
        return self._normalizer

    @property
    def distribution(self) -> Callable[..., int]:
        return self._distribution

    def _np_token_data(self, token_id: List[int]) -> mx.np.ndarray:
        return mx.np.array(token_id, ctx=self._model.ctx).astype(int).reshape(1, 1)

    def __call__(self) -> str:
        """Generate a SMILES string.

        Returns
        -------
        smiles : str
        """
        # Initialize hidden states.
        states = self._model.begin_state(batch_size=1, func=self.state_initializer)
        # Tokenize sequence correctly.
        tokens: List[str] = Token.tokenize(self.prefix)
        # Save predicted token IDs.
        token_ids: List[int] = [self._vocabulary[tokens[0]]]

        # Update `states` on prefix tokens (no prediction is made).
        for token in tokens[1:]:
            token_id = self._np_token_data([token_ids[-1]])
            _, states = self._model(token_id, states)
            token_ids.append(self._vocabulary[token])

        # Predict the next token IDs.
        for n_iter in range(self.max_length):
            inputs = self._np_token_data([token_ids[-1]])
            outputs, states = self._model(inputs, states)

            # Normalize outputs to get probabilities (e.g. using softmax).
            probabilities: mx.np.ndarray = self._normalizer(outputs[0])

            # Sample from a distribution (e.g. multinomial).
            next_token_id: int = self._distribution(probabilities)

            # Interrupt, if End-of-SMILES token is generated.
            token: str = self._vocabulary.idx_to_token[next_token_id]
            if token == Token.EOS:
                break

            token_ids.append(next_token_id)

        smiles = ''.join(self._vocabulary.idx_to_token[id_] for id_ in token_ids)
        # Remove special tokens.
        smiles = Token.crop(smiles)

        return smiles


def _identity(data: mx.np.ndarray) -> mx.np.ndarray:
    return data


def _argmax(logits: mx.np.ndarray) -> int:
    return logits.argmax().item()


class ArgmaxSearch(BaseSearch):
    """Generate new SMILES strings using greedy search (argmax) method. Basically, the
    strategy is to search for the next token with the highest conditional (logit)
    probability.

    Parameters
    ----------
    model : SMILESEncoderDecoderABC
        A (trained) language model. Generates one token ID in one step.
    vocabulary : SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str
        The prefix of a SMILES string to generate.
    max_length : int
        The maximum number of tokens in the SMILES string being generated.
    state_initializer : callable, any -> mx.nd.NDArray
        The model's hidden state initializer function (e.g. mx.nd.ones).
    """

    def __init__(
            self,
            model: SMILESEncoderDecoderABC,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str = Token.BOS,
            max_length: int = 80,
            state_initializer: Callable[..., mx.nd.NDArray] = mx.nd.uniform,
    ):
        super().__init__(
            model=model,
            vocabulary=vocabulary,
            prefix=prefix,
            max_length=max_length,
            state_initializer=state_initializer,
            normalizer=_identity,
            distribution=_argmax,
        )


def _nd_multinomial(probabilities: mx.np.ndarray) -> int:
    sample = mx.random.multinomial(probabilities.as_nd_ndarray())
    sample = sample.as_np_ndarray().item()
    return sample


# noinspection PyUnresolvedReferences
_numpy_softmax = mx.npx.softmax


class SoftmaxSearch(BaseSearch):
    """Softmax with a sensitivity parameter followed by sampling from multinomial
    distribution.

    Parameters
    ----------
    model : SMILESEncoderDecoderABC
        A (trained) language model. Generates one token ID in one step.
    vocabulary : SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str, default Token.BOS
        The prefix of a SMILES string to generate.
    max_length : int, default 80
        The maximum number of tokens in the SMILES string being generated.
    temperature : float, default 1.0
        A sensitivity parameter.
    state_initializer : callable, any -> mx.nd.NDArray
        The model's hidden state initializer function (e.g. mx.nd.ones).
    """
    def __init__(
            self,
            model: SMILESEncoderDecoderABC,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str = Token.BOS,
            max_length: int = 80,
            temperature: float = 1.0,
            state_initializer: Callable[..., mx.nd.NDArray] = mx.nd.zeros,
            normalizer: Optional[Callable[..., mx.np.ndarray]] = None,
            distribution: Optional[Callable[..., int]] = None,
    ):
        normalizer = normalizer or _numpy_softmax
        distribution = distribution or _nd_multinomial
        normalizer_part = functools.partial(normalizer, temperature=temperature)

        super().__init__(
            model=model,
            vocabulary=vocabulary,
            prefix=prefix,
            max_length=max_length,
            state_initializer=state_initializer,
            normalizer=normalizer_part,
            distribution=distribution,
        )

        self.temperature = temperature

    @property
    def temperature(self) -> float:
        """Return the sensitivity parameter.
        """
        return self.__temperature

    @temperature.setter
    def temperature(self, temperature: float):
        """Set a new sensitivity parameter. Change also the `normalizer`s temperature.

        Parameters
        ----------
        temperature : float

        Raises
        ------
        ValueError
            If `temperature` <= 0.
        """
        if temperature <= 0:
            raise ValueError('temperature must be greater than zero')

        self._normalizer = functools.partial(self._normalizer, temperature=temperature)
        self.__temperature = temperature


def _gumbel_trick(logits: mx.np.ndarray, temperature: float) -> mx.np.ndarray:
    gumbel = mx.np.random.gumbel(size=logits.shape[0], ctx=logits.ctx)
    gumbel_logits = (logits - gumbel) / temperature
    # noinspection PyUnresolvedReferences
    probabilities = mx.npx.softmax(gumbel_logits, temperature=temperature)

    return probabilities


class GumbelSoftmaxSearch(SoftmaxSearch):
    """Similar to `SoftmaxSearch` but uses the Gumbel-Softmax trick.

    Parameters
    ----------
    model : SMILESEncoderDecoderABC
        A (trained) language model. Generates one token ID in one step.
    vocabulary : SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str, default Token.BOS
        The prefix of a SMILES string to generate.
    max_length : int, default 80
        The maximum number of tokens in the SMILES string being generated.
    temperature : float, default 1.0
        A sensitivity parameter.
    state_initializer : callable, any -> mx.nd.NDArray
        The model's hidden state initializer function (e.g. mx.nd.ones).
    """

    def __init__(
            self,
            model: SMILESEncoderDecoderABC,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str = Token.BOS,
            max_length: int = 80,
            temperature: float = 1.0,
            state_initializer: Callable[..., mx.nd.NDArray] = mx.nd.zeros,
            normalizer: Optional[Callable[..., mx.np.ndarray]] = None,
            distribution: Optional[Callable[..., int]] = None,
    ):
        normalizer = normalizer or _gumbel_trick

        super().__init__(
            model=model,
            vocabulary=vocabulary,
            prefix=prefix,
            max_length=max_length,
            temperature=temperature,
            state_initializer=state_initializer,
            normalizer=normalizer,
            distribution=distribution,
        )
