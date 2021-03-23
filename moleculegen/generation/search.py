"""
SMILES samplers to generate new molecules using various greedy search methods.

Classes:
    BaseSearch: The base SMILES sampler class.
    ArgmaxSearch: Generate new SMILES strings using greedy search (argmax) method.
    SoftmaxSearch: Softmax with a sensitivity parameter followed by sampling from
        multinomial distribution.
    GumbelSoftmaxSearch: Gumbel-Softmax with a sensitivity parameter followed by sampling
        from multinomial distribution.
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
from ..estimation.base import SMILESLM


class BaseSearch:
    """The base SMILES sampler class.

    Parameters
    ----------
    model : moleculegen.estimation.SMILESLM
        A (trained) language model. Generates one token ID in one step.
    vocabulary : moleculegen.data.SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str
        The prefix of a SMILES string to generate.
    max_length : int
        The maximum number of tokens in the SMILES string being generated.
    normalizer : callable, any -> mx.np.ndarray
        A function that accepts model's outputs (logits) and returns probabilities.
    distribution : callable, any -> int
        A function that accepts an ndarray of probabilities and returns the ID
        of a sampled token.
    """

    def __init__(
            self,
            model: SMILESLM,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str,
            max_length: int,
            normalizer: Callable[..., mx.np.ndarray],
            distribution: Callable[..., int],
    ):
        self._model = model
        self._vocabulary = vocabulary
        self._normalizer = normalizer
        self._distribution = distribution

        self.prefix = prefix
        self.max_length = max_length

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
    def model(self) -> SMILESLM:
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
        # If the inherited model class has a hidden state, `empty_state` is
        # expected to free its content (e.g. set to None) prior to training
        # and `begin_state` to reinitialize its content.
        getattr(self._model, 'empty_state', lambda: None)()
        getattr(self._model, 'begin_state', lambda *a: None)(1)

        # Tokenize sequence correctly.
        tokens: List[str] = Token.tokenize(self.prefix)
        # Save predicted token IDs.
        token_ids: List[int] = [self._vocabulary[tokens[0]]]

        # Update `states` on prefix tokens (no prediction is made).
        for token in tokens[1:]:
            token_id = self._np_token_data([token_ids[-1]])
            self._model(token_id)
            token_ids.append(self._vocabulary[token])

        # Predict the next token IDs.
        for n_iter in range(self.max_length):
            inputs = self._np_token_data([token_ids[-1]])  # shape=(1, 1)
            outputs = self._model(inputs)  # shape=(1, 1, len(self._vocabulary))

            # Normalize outputs to get probabilities (e.g. using softmax);
            # probabilities.shape == (1, len(self._vocabulary)).
            probabilities: mx.np.ndarray = self._normalizer(outputs[0])

            # Sample from a distribution (e.g. multinomial).
            next_token_id: int = self._distribution(probabilities[0])

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
    model : moleculegen.estimation.SMILESLM
        A (trained) language model. Generates one token ID in one step.
    vocabulary : moleculegen.data.SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str, default=moleculegen.Token.BOS
        The prefix of a SMILES string to generate.
    max_length : int, default=100
        The maximum number of tokens in the SMILES string being generated.
    """

    def __init__(
            self,
            model: SMILESLM,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str = Token.BOS,
            max_length: int = 100,
    ):
        super().__init__(
            model=model,
            vocabulary=vocabulary,
            prefix=prefix,
            max_length=max_length,
            normalizer=_identity,
            distribution=_argmax,
        )


def _multinomial(probabilities: mx.np.ndarray) -> int:
    return mx.random.multinomial(probabilities.as_nd_ndarray()).asscalar()


def _p_multinomial(probabilities: mx.np.ndarray, p: float) -> int:
    sorted_idx = mx.np.argsort(probabilities)[::-1]
    # noinspection PyUnresolvedReferences
    cum_sum = mx.np.cumsum(probabilities[sorted_idx])
    remove_mask = cum_sum > p
    remove_mask[1:][:] = remove_mask[:-1]
    select_mask = ~remove_mask
    probabilities[sorted_idx[remove_mask]] = 0.
    top_p = probabilities[sorted_idx[select_mask]]
    return _multinomial(probabilities / top_p.sum())


def _k_multinomial(probabilities: mx.np.ndarray, k: int) -> int:
    nd_prob = probabilities.as_nd_ndarray()
    top_k = mx.nd.topk(nd_prob, k=k, dtype=int)
    top_k_prob = nd_prob[top_k]
    top = _multinomial(top_k_prob / top_k_prob.sum())
    return top_k[top].asscalar()


def _get_multinomial(strategy: Optional[float]) -> Callable[..., int]:
    if strategy is None:
        return _multinomial
    elif isinstance(strategy, float):
        return functools.partial(_p_multinomial, p=strategy)
    elif isinstance(strategy, int):
        return functools.partial(_k_multinomial, k=strategy)


# noinspection PyUnresolvedReferences
_numpy_softmax = mx.npx.softmax


class SoftmaxSearch(BaseSearch):
    """Softmax with a sensitivity parameter followed by
    multinomial distribution/top-p/top-k sampling.

    Parameters
    ----------
    model : moleculegen.estimation.SMILESLM
        A (trained) language model. Generates one token ID in one step.
    vocabulary : moleculegen.data.SMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str, default=moleculegen.Token.BOS
        The prefix of a SMILES string to generate.
    max_length : int, default=100
        The maximum number of tokens in the SMILES string being generated.
    temperature : float, default=1.0
        A sensitivity parameter.
    strategy : float or int, default=None
        If None, sample from multinomial.
        If float, top-p (nucleus) sampling.
        If int, top-k sampling.

    References
    ----------
    .. [1] A. Holtzman et al. The Curious Case of Neural Text Degeneration.
    """
    def __init__(
            self,
            model: SMILESLM,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str = Token.BOS,
            max_length: int = 100,
            temperature: float = 1.0,
            normalizer: Optional[Callable[..., mx.np.ndarray]] = None,
            distribution: Optional[Callable[..., int]] = None,
            strategy: Optional[float] = None,
    ):
        normalizer = normalizer or _numpy_softmax
        distribution = distribution or _get_multinomial(strategy)

        super().__init__(
            model=model,
            vocabulary=vocabulary,
            prefix=prefix,
            max_length=max_length,
            normalizer=functools.partial(normalizer, temperature=temperature),
            distribution=distribution,
        )

        self.temperature = temperature
        self.strategy = strategy

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
        if not isinstance(temperature, (int, float)) or temperature <= 0:
            raise ValueError(
                f'`temperature` must be a number greater than zero, not {temperature!r}.'
            )

        self._normalizer = functools.partial(self._normalizer, temperature=temperature)
        self.__temperature = temperature

    @property
    def strategy(self) -> Optional[float]:
        return self.__strategy

    @strategy.setter
    def strategy(self, strategy: Optional[float]):
        if isinstance(strategy, float):
            if not (0. < strategy <= 1.):
                raise ValueError(f'`strategy` must be in (0., 1.], not {strategy!r}.')
        elif isinstance(strategy, int):
            if strategy <= 0:
                raise ValueError(f'`strategy` must be greater than 0, not {strategy!r}.')
        elif strategy is not None:
            raise TypeError(
                f'`strategy` must be None or float, not {type(strategy)}.'
            )

        self._distribution = _get_multinomial(strategy)
        self.__strategy = strategy


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
    model : moleculegen.estimation.SMILESLM
        A (trained) language model. Generates one token ID in one step.
    vocabulary : moleculegen.dataSMILESVocabulary
        A SMILES vocabulary. Provides id-to-token conversion.
    prefix : str, default=moleculegen.Token.BOS
        The prefix of a SMILES string to generate.
    max_length : int, default=100
        The maximum number of tokens in the SMILES string being generated.
    temperature : float, default=1.0
        A sensitivity parameter.
    strategy : float or int, default=None
        If None, sample from multinomial.
        If float, top-p (nucleus) sampling.
        If int, top-k sampling.

    References
    ----------
    .. [1] A. Holtzman et al. The Curious Case of Neural Text Degeneration.
    """

    def __init__(
            self,
            model: SMILESLM,
            vocabulary: SMILESVocabulary,
            *,
            prefix: str = Token.BOS,
            max_length: int = 100,
            temperature: float = 1.0,
            strategy: Optional[float] = None,
    ):
        super().__init__(
            model=model,
            vocabulary=vocabulary,
            prefix=prefix,
            max_length=max_length,
            temperature=temperature,
            normalizer=_gumbel_trick,
            strategy=strategy,
        )
