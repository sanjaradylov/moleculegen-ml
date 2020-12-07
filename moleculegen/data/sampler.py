"""
Utilities to sample single instances or batches of SMILES subsequences.

Classes
-------
BatchSampler
    An ABC to check the validity of custom batch samplers.
SMILESBatchSampler
    Generate batches of SMILES sequences using specified sampler.
SMILESBatchColumnSampler
    Generate batches of SMILES subsequences "column-wise".
SMILESBatchRandomSampler
    Generate batches of SMILES subsequences randomly.
SMILESConsecutiveSampler
    Generate samples of SMILES subsequences "consecutively".
SMILESRandomSampler
    Sample SMILES sequences randomly.
StateInitializerMixin
    A mixin class for specific state initialization techniques during
    model training.
"""

__all__ = (
    'BatchSampler',
    'SMILESBatchSampler',
    'SMILESBatchColumnSampler',
    'SMILESBatchRandomSampler',
    'SMILESConsecutiveSampler',
    'SMILESRandomSampler',
    'StateInitializerMixin',
)


import abc
import collections
import dataclasses
import functools
import random
import warnings
from typing import Any, Callable, Generator, Iterator, List, Optional, Union, Tuple

import mxnet as mx

from .vocabulary import SMILESVocabulary


@dataclasses.dataclass(eq=False, frozen=True)
class Batch:
    """Stores a (mini-)batch of
    - input token sequences of shape (batch size, time steps);
    - output token sequences of shape (batch size, time steps);
    - valid lengths of samples of shape (batch size,);
    - a flag to (re-)initialize or keep the hidden states of the model being trained.
    """

    inputs: mx.np.ndarray         # Input tokens.
    outputs: mx.np.ndarray        # Output tokens.
    valid_lengths: mx.np.ndarray  # The number of valid tokens in `outputs`.
    init_state: bool = False      # Whether to (re-)initialize hidden states.

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the mini-batch.
        """
        return self.inputs.shape

    def as_in_ctx(self, ctx: mx.context.Context = mx.context.cpu()) -> 'Batch':
        """Change the context of the mxnet.np.ndarray fields of the instance.

        Parameters
        ----------
        ctx : mxnet.context.Context, default mxnet.context.cpu()
            CPU or GPU.

        Returns
        -------
        batch : Batch
        """
        return self.__class__(**{
            key: (
                value.as_in_ctx(ctx)
                if isinstance(value, mx.np.ndarray)
                else value
            )
            for key, value in dataclasses.asdict(self).items()
        })


class BatchSampler(metaclass=abc.ABCMeta):
    """An ABC to check the validity of custom batch samplers.

    Since some of our batch samplers do not inherit from gluon's BatchSampler, but
    provide valid API, we cannot check API correspondence with
    `isinstance(custom_sampler, mxnet.gluon.data.BatchSampler)` etc. Hence, we implement
    a class method `__subclasshook__` for `isinstance(custom_sampler, BatchSampler)`
    checks.

    Notes
    -----
    One does not have to use this class as a superclass. This is rather a reminder of
    the main batch sampler API, as well as a convenient type/instance checker.
    """

    @classmethod
    def __subclasshook__(cls, other_class: type) -> bool:
        """Check if `other_class` supports the main Batch Sampler API, i.e. implements
        `__iter__`, `__len__`, and `init_states` methods.

        Parameters
        ----------
        other_class : type
            A class to check.

        Returns
        -------
        bool
            True, if `other_class` inherits mxnet.gluon.data.BatchSampler and
            StateInitializerMixin, or if it supports their abstract methods.

        Notes
        -----
        The behavior of the method is not inherited.
        """
        if cls is BatchSampler:
            if (
                issubclass(other_class, mx.gluon.data.BatchSampler)
                and issubclass(other_class, StateInitializerMixin)
            ):
                return True

            attributes = collections.ChainMap(
                *(superclass.__dict__ for superclass in other_class.__mro__)
            )
            methods = ('__iter__', '__len__', 'init_states', 'init_state_func')

            return all(method in attributes for method in methods)

        return NotImplemented

    @abc.abstractmethod
    def __iter__(self):
        """Iterate over batches."""

    @abc.abstractmethod
    def __len__(self):
        """Return the number of batches."""

    @classmethod
    @abc.abstractmethod
    def init_states(cls, *args, **kwargs):
        """See moleculegen.StateInitializerMixin for the description of the method."""

    @staticmethod
    @abc.abstractmethod
    def init_state_func(*args, **kwargs):
        """See moleculegen.StateInitializerMixin for the description of the method."""


class StateInitializerMixin:
    """A mixin class for specific state initialization techniques during
    model training.

    The main purpose is to provide additional API for batch samplers that
    encourage nontrivial ways of state (re-)initialization during training.
    For example, while sampling a mini-batch from SMILESBatchColumnSampler,
    one should reinitialize states when finally all the samples from the
    mini-batch encounter Token.EOS.
    """

    @classmethod
    def init_states(
            cls,
            model: Any,
            mini_batch: Any,
            states: Optional[List[mx.np.ndarray]] = None,
            detach: bool = True,
            init_state_func: Optional[Callable[[Any], mx.nd.ndarray.NDArray]] = None,
            *args,
            **kwargs,
    ) -> List[mx.np.ndarray]:
        """(Re-)initialize the hidden state(s) of `model`.

        Parameters
        ----------
        model : any
            A Gluon model.
        mini_batch : any
            A (mini-)batch instance. Basically, tuples or dataclasses.
        states : list of mxnet.nd.ndarray.NDArray, default None
            The previous hidden states of `model`.
            If None, then a new state list will be initialized.
        detach : bool, default True
            Whether to detach the sates from the computational graph.
        init_state_func : callable, any -> mxnet.nd.ndarray.NDArray,
                default None
            A distribution function to initialize `states`.
            If None, the previous states will be returned.
            Recommended to use `StateInitializerMixin.init_state_func` method
            to declare this callable.

        Returns
        -------
        states : list of mxnet.np.ndarray
            A state list.

        Raises
        ------
        AttributeError
            If `mini_batch` does not have `shape` attribute.
            If `model` does not implement `begin_state` method.

        Notes
        -----
        The context of the hidden states is the same as the `model`s context.
        """
        if not hasattr(mini_batch, 'shape'):
            raise AttributeError(
                '`mini_batch` must store `shape` attribute.'
            )
        if not hasattr(model, 'begin_state'):
            raise AttributeError(
                '`model` must implement `begin_state` method.'
            )

        if states is None:
            states = model.begin_state(
                batch_size=mini_batch.shape[0],
                func=init_state_func,
            )

        if detach:
            states = [state.detach() for state in states]

        return states

    @staticmethod
    def init_state_func(
            func: Callable[[Any], mx.nd.ndarray.NDArray] = mx.nd.zeros,
            **func_kwargs,
    ) -> Callable[[Any], mx.nd.ndarray.NDArray]:
        """Return distribution callable `func` with arbitrary
        (non-default) arguments `func_kwargs` specified in advance. Use
        primarily for state initialization.

        Parameters
        ----------
        func : callable, any -> mxnet.nd.ndarray.NDArray
            One of the distribution functions from mxnet.nd.random or
            functions like nd.zeros.
        func_kwargs : dict, default None
            Parameters of `func` excluding `shape`.

        Returns
        -------
        func : callable
            Partial function callable.

        Raises
        ------
        ValueError
            If `shape` parameter is included in `distribution_args`.
            This parameter will be used separately in state initialization.
        """
        if 'shape' in func_kwargs or 'ctx' in func_kwargs:
            raise ValueError(
                '`shape` and `ctx` parameters should not be passed, since '
                'they are processed internally by the model.'
            )

        return functools.partial(func, **func_kwargs)


class SMILESBatchColumnSampler(StateInitializerMixin):
    """Generate batches of SMILES subsequences. Collect SMILES sequences of
    size `batch_size`, divide them into mini-batches of shape
    (`batch_size`, `n_steps`), and generate them successively. The generated
    objects are of type `Batch`.

    Parameters
    ----------
    corpus : list of list of int
        The original data corpus loaded from a vocabulary.
    batch_size : int
        The number of samples to generate.
    n_steps : int
        The number of (time) steps.
    shuffle : bool, default True
        Whether to shuffle the corpus before sampling.

    Attributes
    ----------
    corpus : list of list of int
    batch_size : int
    n_steps : int

    Examples
    --------
    >>> import tempfile
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> smiles_strings = (
    'CN=C=O\n'
    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
    'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N\n'
    'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
    )
    >>> with tempfile.NamedTemporaryFile(mode='w') as temp_fh:
    ...     temp_fh.write(smiles_strings)
    ...     temp_fh.seek(0)
    ...     dataset = SMILESDataset(temp_fh.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESBatchColumnSampler(vocabulary.corpus, 2, 20, shuffle=False)
    >>> print('Input batch samples:')
    >>> for i, batch in enumerate(sampler, start=1):
    ...     print(f'Batch #{i}')
    ...     for sample in batch.inputs:
    ...         print(''.join(vocabulary.get_tokens(sample.tolist())))
    Input batch samples:
    Batch #1
    {CN=C=O}____________
    {CCc1c[n+]2ccc3c4ccc
    Batch #2
    ____________________
    cc4[nH]c3c2cc1}________
    Batch #3
    {OCCc1c(C)[n+](cs1)C
    {CC(=O)NCCC1=CNc2c1c
    Batch #4
    c2cnc(C)nc2N}_______
    c(OC)cc2}___________
    >>> print('Output batch samples:')
    >>> for i, batch in enumerate(sampler, start=1):
    ...     print(f'Batch #{i}')
    ...     for sample in batch.outputs:
    ...         print(''.join(vocabulary.get_tokens(sample.tolist())))
    Output batch samples:
    Batch #1
    CN=C=O}_____________
    CCc1c[n+]2ccc3c4cccc
    Batch #2
    ____________________
    c4[nH]c3c2cc1}_________
    Batch #3
    OCCc1c(C)[n+](cs1)Cc
    CC(=O)NCCC1=CNc2c1cc
    Batch #4
    2cnc(C)nc2N}________
    (OC)cc2}____________
    >>> print('Valid lengths')
    >>> for i, batch in enumerate(sampler, start=1):
    ...     print(f'Batch #{i}')
    ...     print(batch.valid_lengths)
    Batch #1
    [7 20]
    Batch #2
    [0 14]
    Batch #3
    [20 20]
    Batch #4
    [12 8]

    Notes
    -----
    This method is different from SMILESConsecutiveSampler followed by
    mxnet.gluon.data.BatchSampler. See docs or examples for more information.

    See also
    --------
    SMILESBatchSampler
    SMILESConsecutiveSampler
    """

    def __init__(
            self,
            corpus: List[List[int]],
            batch_size: int,
            n_steps: int,
            shuffle: bool = True,
    ):
        self._corpus = corpus
        self._shuffle = shuffle

        self.batch_size = batch_size
        self.n_steps = n_steps

        self._n_samples: Optional[int] = None

    @property
    def corpus(self) -> List[List[int]]:
        """The corpus of the original data set.

        Returns
        -------
        corpus : list of list of int
        """
        return self._corpus

    @property
    def batch_size(self) -> int:
        """The number of samples to generate at each step.

        Returns
        -------
        batch_size : int
        """
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """Set the number of samples (mini-batches).

        Parameters
        ----------
        batch_size : int
        """
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Batch size must be positive non-zero.')

        self.__batch_size = batch_size

    @property
    def n_steps(self) -> int:
        """Return the number of steps.

        Returns
        -------
        n_steps : int
        """
        return self.__n_steps

    @n_steps.setter
    def n_steps(self, n_steps: int):
        """Set the number of (time) steps.

        Parameters
        ----------
        n_steps : int
        """
        if not isinstance(n_steps, int) or n_steps < 1:
            raise ValueError('Number of steps must be positive non-zero.')

        self.__n_steps = n_steps

    def __len__(self) -> int:
        """Return the number of batches.
        """
        if self._n_samples is not None:
            return self._n_samples

        warnings.warn(
            'The iterator of `SMILESBatchColumnSampler` is inherently slow,\n'
            'so is `__len__` method.'
        )

        n_samples = 0
        for n_samples, _ in enumerate(iter(self), start=1):
            pass
        self._n_samples = n_samples

        return n_samples

    def __iter__(self) -> Generator[Batch, None, None]:
        """Iterate over the samples of the corpus.

        Strategy:
        1. Load corpus containing lists of token IDs.
        2. For every iteration, get batch of size `self.batch_size` and
           pad token lists to have equal dimensions.
        3. The modified batch is divided into mini-batches (input, output)
           of size (`self.batch_size`, `self.n_steps`).

        Yields
        ------
        mini_batch : Batch
            Mini-batch containing inputs, outputs, and their corresponding
            valid lengths.
        """
        if self._shuffle:
            random.shuffle(self._corpus)

        n_batches = (
            len(self._corpus) // self.batch_size
            * self.batch_size
        )

        self._n_samples = 0

        for i_batch in range(0, n_batches, self.batch_size):
            curr_slice = slice(i_batch, i_batch + self.batch_size)
            batch = self._pad(self._corpus[curr_slice])

            yield from self._iter_steps(batch)

    def _iter_steps(self, batch: mx.np.ndarray) -> Generator[Batch, None, None]:
        """Generate batches separated by time steps.

        Parameters
        ----------
        batch : np.ndarray, shape = (self.batch_size, max_smiles_len)
            Batch.

        Yields
        ------
        mini_batch : Batch
            Mini-batch containing inputs, outputs, and their corresponding
            valid lengths.
        """

        def get_valid_lengths(sample: mx.np.ndarray) -> mx.np.ndarray:
            """For every entry in `sample`, return the lengths of subsequences
            containing any valid SMILES tokens excluding padding token.

            Parameters
            ----------
            sample : np.ndarray, shape = (self.batch_size - 1, max_len)
                Input or output sample.

            Returns
            -------
            valid_lengths : np.ndarray, shape = self.batch_size - 1
                Valid lengths for every entry.
            """
            lengths: List[int] = []

            # FIXME Iterating over matrix is somewhat brute and ineffective.
            for seq in sample:
                for i, s in enumerate(seq):
                    if s == pad_token_idx:
                        lengths.append(i)
                        break
                else:
                    lengths.append(sample.shape[1])

            return mx.np.array(lengths, dtype=int)

        pad_token_idx = SMILESVocabulary.PAD_ID

        inputs, outputs = batch[:, :-1], batch[:, 1:]

        h = True
        for i_step in range(0, inputs.shape[1], self.n_steps):
            step_slice = (..., slice(i_step, i_step + self.n_steps))
            x, y = inputs[step_slice], outputs[step_slice]

            # If every token in an output batch is PAD, ignore this sample.
            v_y = get_valid_lengths(y)
            if v_y.sum().item() == 0:
                continue

            yield Batch(x, y, v_y, h)

            h = False

            self._n_samples += 1

    def _pad(self, item_lists: List[List[int]]) -> mx.np.ndarray:
        """Get a batch of SMILES token lists and fill with the `PAD` token ids
        those lists with lesser number of tokens.

        Parameters
        ----------
        item_lists : list
            List of token lists, each token list contains token indices and
            constitutes a SMILES string.

        Returns
        -------
        new_item_list : np.ndarray, shape = (self.batch_size, max_len)
            The array of token lists.
        """
        # Get the maximum string length among all entries in a batch.
        max_len = max(map(len, item_lists))

        if max_len % self.n_steps != 0:
            # Make it divisible by self.n_steps so that we can iterate
            # over the batch max_len // self.n_steps times and
            # retrieve self.n_steps tokens.
            max_len = (max_len // self.n_steps + 1) * self.n_steps

        # Increment it since we divide `batch` into
        # (batch[:, :-1], batch[:, 1:]).
        max_len += 1

        pad_token_idx = SMILESVocabulary.PAD_ID
        new_items_list: List[List[int]] = []

        for item_list in item_lists:
            pad_len = max_len - len(item_list)
            pad_list = [pad_token_idx] * pad_len
            new_items_list.append(item_list + pad_list)

        return mx.np.array(new_items_list, dtype=int)

    @classmethod
    def init_states(
            cls,
            model: Any,
            mini_batch: Batch,
            states: Optional[List[mx.np.ndarray]] = None,
            init_state_func: Optional[Callable[[Any], mx.nd.ndarray.NDArray]] = None,
            detach: bool = False,
            *args,
            **kwargs,
    ) -> List[mx.np.ndarray]:
        """(Re-)initialize the hidden state(s) of `model`.

        Parameters
        ----------
        model : any
            A Gluon model.
        mini_batch : Batch
            A (mini-)batch instance. Basically, tuples or dataclasses.
        states : list of mxnet.nd.ndarray.NDArray, default None
            The previous hidden states of `model`.
            If None, then a new state list will be initialized.
        init_state_func : callable, any -> mxnet.nd.ndarray.NDArray,
                default None
            A distribution function to initialize `states`.
            If None, the previous states will be returned.
            Recommended to use `StateInitializerMixin.init_state_func` method
            to declare this callable.
        detach : bool, default False
            Whether to detach `states` from the computational graph.

        Returns
        -------
        states : list of mxnet.np.ndarray
            A state list.

        Raises
        ------
        AttributeError
            If `mini_batch` does not have `shape` attribute.
            If `model` does not implement `begin_state` method.

        Notes
        -----
        The context of the hidden states is the same as the `model`s context.
        """
        if mini_batch.init_state or states is None:
            states = super().init_states(
                model=model,
                mini_batch=mini_batch,
                states=states,
                init_state_func=init_state_func,
                detach=detach,
            )

        if detach:
            states = [state.detach() for state in states]

        return states


class SMILESBatchSampler(mx.gluon.data.BatchSampler, StateInitializerMixin):
    """Generate (mini-)batches of SMILES (sub)sequences. Extends
    mxnet.gluon.data.BatchSampler API by implementing `init_states` for state
    initialization.

    Examples
    --------
    >>> import tempfile
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> smiles_strings = (
    'CN=C=O\n'
    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
    'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N\n'
    'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
    )
    >>> with tempfile.NamedTemporaryFile(mode='w') as temp_fh:
    ...     temp_fh.write(smiles_strings)
    ...     temp_fh.seek(0)
    ...     dataset = SMILESDataset(temp_fh.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESConsecutiveSampler(vocabulary.corpus, 20, shuffle=False)
    >>> batch_sampler = SMILESBatchSampler(sampler, 2)
    >>> for batch_i, batch in enumerate(batch_sampler, start=1):
    ...     print(f'Batch {batch_i}:')
    ...     for sample_i in range(batch.shape[0]):
    ...         input_sample = batch.inputs[sample_i].tolist()
    ...         output_sample = batch.outputs[sample_i].tolist()
    ...         print(''.join(vocabulary.get_tokens(input_sample)))
    ...         print(''.join(vocabulary.get_tokens(output_sample)))
    ...     print(f'Valid lengths: {batch.valid_lengths}.')
    ...     print()
    Batch 1:
    {OCCc1c(C)[n+](cs1)C
    OCCc1c(C)[n+](cs1)Cc
    c2cnc(C)nc2N}_______
    2cnc(C)nc2N}________
    Valid lengths: [20 12].

    Batch 2:
    {CCc1c[n+]2ccc3c4ccc
    CCc1c[n+]2ccc3c4cccc
    cc4[nH]c3c2cc1}________
    c4[nH]c3c2cc1}_________
    Valid lengths: [20 14].

    Batch 3:
    {CC(=O)NCCC1=CNc2c1c
    CC(=O)NCCC1=CNc2c1cc
    c(OC)cc2}___________
    (OC)cc2}____________
    Valid lengths: [20  8].

    Batch 4:
    {CN=C=O}____________
    CN=C=O}_____________
    Valid lengths: [7].

    See also
    --------
    SMILESBatchColumnSampler
    SMILESBatchRandomSampler
    """

    def __iter__(self) -> Generator[Batch, None, None]:
        """Run superclass' iterator and covert the yielded list of samples into Batch
        instances.

        Yields
        ------
        batch : Batch
        """
        batch_class = getattr(self._sampler, 'Batch', Batch)

        for batch in super().__iter__():
            inputs = (sample.inputs for sample in batch)
            outputs = (sample.outputs for sample in batch)
            valid_lengths = (sample.valid_length for sample in batch)

            yield batch_class(
                inputs=mx.np.array(list(inputs), dtype=int),
                outputs=mx.np.array(list(outputs), dtype=int),
                valid_lengths=mx.np.array(list(valid_lengths), dtype=int),
            )


class SMILESConsecutiveSampler(mx.gluon.data.Sampler):
    """Consecutively generate SMILES (sub)sequences of length `n_steps`,
    padding the lacking tokens if necessary. By default, the generated objects
    are of type Sample, which has attributes `inputs` (input sequences),
    `outputs` (output sequences shifted by one step), and `valid_length`
    (the number of valid tokens in `outputs`).

    This class generates only one sample at a time. To sample mini-batches,
    refer to mxnet.gluon.data.BatchSampler.

    Parameters
    ----------
    corpus : list of list of int
        The original data corpus loaded from a vocabulary.
    n_steps : int, default None
        The length of a substring.
        If None, it equals to the maximum string length in the corpus minus 1.
    shuffle : bool, default True
        Whether to shuffle the corpus before sampling.
    sample_type : {'sample', 'tuple'}, default 'sample'
        The type of an object being generated.
        If 'sample', objects are of type SMILESConsecutiveSampler.Sample,
        which has attributes `inputs` (input sequences), `outputs` (output
        sequences shifted by one step), and `valid_length` (the number of
        valid tokens in `outputs`).
        If 'tuple', return a tuple comprising `inputs` and `outputs`, without
        `valid_length`. This option is created for tensorflow.data.Dataset-
        like API to create a dataset from this generator and fit a tf.keras
        model.

    Examples
    --------
    >>> import tempfile
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> smiles_string = 'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1'
    >>> with tempfile.NamedTemporaryFile(mode='w') as temp_fh:
    ...     temp_fh.write(smiles_string)
    ...     temp_fh.seek(0)
    ...     dataset = SMILESDataset(temp_fh.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESConsecutiveSampler(vocabulary.corpus, n_steps=20)
    >>> len(sampler)
    2
    >>> for sample in sampler:
    ...     print(''.join(vocabulary.get_tokens(sample.inputs)))
    ...     print(''.join(vocabulary.get_tokens(sample.outputs)))
    ...     print(sample.valid_length)
    {CCc1c[n+]2ccc3c4ccc
    CCc1c[n+]2ccc3c4cccc
    20
    cc4[nH]c3c2cc1}________
    c4[nH]c3c2cc1}_________
    14
    """

    @dataclasses.dataclass(eq=False, frozen=True)
    class Sample:
        inputs: List[int]   # Input tokens.
        outputs: List[int]  # Output tokens.
        valid_length: int   # The number of valid tokens in `outputs`.

    def __init__(
            self,
            corpus: List[List[int]],
            n_steps: Optional[int] = None,
            shuffle: bool = True,
            *,
            sample_type: str = 'sample',
    ):
        self._corpus = corpus
        self._shuffle = shuffle
        self._n_steps = n_steps or max(map(len, self._corpus))

        if sample_type == 'sample':
            self._sample_type = lambda inputs, outputs, valid_length: \
                self.Sample(inputs, outputs, valid_length)
        elif sample_type == 'tuple':
            self._sample_type = lambda inputs, outputs, ignore: \
                tuple((inputs, outputs))
        else:
            raise ValueError(
                f"`sample_type` must be either 'sample' or 'tuple'."
            )

        self._n_samples: Optional[int] = None

    def __len__(self) -> int:
        """Return the number of samples.
        """
        if self._n_samples is not None:
            return self._n_samples

        n_samples = 0
        for n_samples, _ in enumerate(iter(self), start=1):
            pass
        self._n_samples = n_samples

        return n_samples

    def __iter__(self) -> Generator[
            Union[Sample, Tuple[List[int], List[int]]], None, None]:
        """Generate samples of type SMILESConsecutiveSampler.Sample or tuple
        (depending on the formal parameter `sample_type`) comprising input
        sequences, output sequences, and (optionally) valid lengths.

        Yields
        ------
        sample : SMILESConsecutiveSampler.Sample or tuple of list of int
            Input-output sequences.
        """
        if self._shuffle:
            random.shuffle(self._corpus)

        self._n_samples = 0

        # Iterate over the corpus of SMILES token indices.
        for tokens in self._corpus:
            step_i = 0  # Starting index of a subsequence.
            step_len = step_i + self._n_steps  # Ending index.

            # Iterate over subsequences of length `self._n_steps`.
            # Discard the last subsequence if its length < `self._n_steps`.
            while step_len + 1 <= len(tokens):
                yield self._sample_type(
                    tokens[step_i: step_len],
                    tokens[step_i+1: step_len+1],
                    self._n_steps,
                )

                step_i += self._n_steps
                step_len += self._n_steps

                self._n_samples += 1

            # If the last subsequence is of length < `self._n_steps`,
            remainder_len = step_len - len(tokens) + 1
            if remainder_len > 0 and remainder_len != self._n_steps:
                # fill the lacking tokens with Token.PAD index.
                pad = [SMILESVocabulary.PAD_ID]

                yield self._sample_type(
                    tokens[step_i: step_len] + pad*(remainder_len-1),
                    tokens[step_i+1: step_len+1] + pad*remainder_len,
                    self._n_steps - remainder_len,
                )

                self._n_samples += 1

    def __call__(self) -> Iterator[Union[Sample, Tuple[List[int], List[int]]]]:
        """Return a sampler iterator.
        No need to use this syntactic sugar explicitly w/ mxnet API. Created
        only for tensorflow's Dataset or similar APIs.
        """
        return iter(self)


class SMILESBatchRandomSampler(SMILESBatchSampler):
    """Same as `SMILESBatchSampler` but `init_states` always returns newly initialized
    states and they are not detached from a computational graph.
    Passing a `SMILESRandomSampler` instance creates a batch sampler that yields
    mini-batches of random subsequences w/ possible repetitions.

    See also
    --------
    SMILESBatchSampler
    SMILESRandomSampler
    """

    @classmethod
    def init_states(
            cls,
            model: Any,
            mini_batch: Any,
            init_state_func: Optional[Callable[[Any], mx.nd.ndarray.NDArray]] = None,
            *args,
            **kwargs,
    ) -> List[mx.np.ndarray]:
        return super().init_states(
            model=model,
            mini_batch=mini_batch,
            init_state_func=init_state_func,
            states=None,
            detach=False,
        )


class SMILESRandomSampler(mx.gluon.data.Sampler):
    """Load SMILES (sub)sequences from `SMILESConsecutiveSampler` and sample
    `n_samples` sequences w/ replacement.

    Parameters
    ----------
    corpus : list of list of int
        The original data corpus loaded from a vocabulary.
    n_steps : int, default None
        The length of a substring.
        If None, it equals to the maximum string length in the corpus minus 1.
    n_samples : int, default None
        The number of samples to yield.
        If None, it equals to the number of loaded sequences from
        `SMILESConsecutiveSampler`.

    Examples
    --------
    >>> import tempfile
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> smiles_strings = (
    'CN=C=O\n'
    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
    'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N\n'
    'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
    )
    >>> with tempfile.NamedTemporaryFile(mode='w') as temp_fh:
    ...     temp_fh.write(smiles_strings)
    ...     temp_fh.seek(0)
    ...     dataset = SMILESDataset(temp_fh.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESRandomSampler(vocabulary.corpus, n_steps=20, n_samples=10)
    >>> len(sampler)
    2
    >>> for sample in sampler:
    ...     print(''.join(vocabulary.get_tokens(sample.inputs)))
    ...     print(''.join(vocabulary.get_tokens(sample.outputs)))
    ...     print(sample.valid_length)
    {OCCc1c(C)[n+](cs1)C
    OCCc1c(C)[n+](cs1)Cc
    20
    {CN=C=O}____________
    CN=C=O}_____________
    7
    c(OC)cc2}___________
    (OC)cc2}____________
    8
    cc4[nH]c3c2cc1}_____
    c4[nH]c3c2cc1}______
    14
    c2cnc(C)nc2N}_______
    2cnc(C)nc2N}________
    12
    c(OC)cc2}___________
    (OC)cc2}____________
    8
    {OCCc1c(C)[n+](cs1)C
    OCCc1c(C)[n+](cs1)Cc
    20
    cc4[nH]c3c2cc1}_____
    c4[nH]c3c2cc1}______
    14
    {OCCc1c(C)[n+](cs1)C
    OCCc1c(C)[n+](cs1)Cc
    20
    cc4[nH]c3c2cc1}_____
    c4[nH]c3c2cc1}______
    14

    See also
    --------
    SMILESConsecutiveSampler
    """

    def __init__(
            self,
            corpus: List[List[int]],
            n_steps: Optional[int] = None,
            n_samples: Optional[int] = None,
    ):
        self._corpus = corpus
        self._n_steps = n_steps or max(map(len, self._corpus))

        self.__data = tuple(iter(
            SMILESConsecutiveSampler(self._corpus, self._n_steps, shuffle=False)))
        self._n_samples = n_samples or len(self.__data)

    def __len__(self) -> int:
        """Return the number of samples.
        """
        return self._n_samples

    def __iter__(self) -> Generator[SMILESConsecutiveSampler.Sample, None, None]:
        """Generate samples of type `SMILESConsecutiveSampler.Sample`
        comprising input sequences, output sequences, and valid lengths.

        Yields
        ------
        sample : SMILESConsecutiveSampler.Sample
            Input-output sequences.
        """
        return (random.choice(self.__data) for _ in range(self._n_samples))
