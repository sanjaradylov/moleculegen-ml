"""
Utilities to sample single instances or batches of SMILES subsequences.

Classes
-------
SMILESBatchSampler
    Generate batches of SMILES sequences using specified sampler.
SMILESBatchColumnSampler
    Generate batches of SMILES subsequences "column-wise".
SMILESConsecutiveSampler
    Generate samples of SMILES subsequences "consecutively".
"""

__all__ = (
    'SMILESBatchSampler',
    'SMILESBatchColumnSampler',
    'SMILESConsecutiveSampler',
)


import dataclasses
import random
from typing import (
    Any, Callable, Generator, Iterator, List, Optional, Union, Tuple)

from mxnet import context, nd, np
from mxnet.gluon.data import BatchSampler, Sampler

from ..base import StateInitializerMixin, Token
from .vocabulary import SMILESVocabulary


@dataclasses.dataclass(eq=False, frozen=True)
class Batch:
    """Stores a (mini-)batch of
    - input token sequences of shape (batch size, time steps);
    - output token sequences of shape (batch size, time steps);
    - valid lengths of samples of shape (batch size,);
    - a flag to (re-)initialize or keep the hidden states of the model being
      trained.
    """

    inputs: np.ndarray         # Input tokens.
    outputs: np.ndarray        # Output tokens.
    valid_lengths: np.ndarray  # The number of valid tokens in `outputs`.
    init_state: bool = False   # Whether to (re-)initialize hidden states.

    @property
    def shape(self) -> Tuple[int, ...]:
        """Return the shape of the mini-batch.
        """
        return self.inputs.shape


class SMILESBatchColumnSampler(StateInitializerMixin):
    """Generate batches of SMILES subsequences. Collect SMILES sequences of
    size `batch_size`, divide them into mini-batches of shape
    (`batch_size`, `n_steps`), and generate them successively. The generated
    objects are of type `Batch`.

    Parameters
    ----------
    vocabulary : SMILESVocabulary
        The vocabulary of the original data corpus.
    batch_size : int
        The number of samples to generate.
    n_steps : int
        The number of (time) steps.
    shuffle : bool, default True
        Whether to shuffle the corpus before sampling.

    Attributes
    ----------
    vocabulary : SMILESVocabulary
    batch_size : int
    n_steps : int

    Examples
    --------
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> from moleculegen.tests.utils import TempSMILESFile
    >>> smiles_strings = (
    'CN=C=O\n'
    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
    'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N\n'
    'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
    )
    >>> with TempSMILESFile(smiles_strings=smiles_strings) as temp_fh:
    ...     dataset = SMILESDataset(temp_fh.file_handler.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESBatchColumnSampler(vocabulary, 2, 20, shuffle=False)
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
    {CC(=O)NCCC1=CNc2c1cc(O
    Batch #4
    c2cnc(C)nc2N}_______
    C)cc2}______________
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
    CC(=O)NCCC1=CNc2c1cc(OC
    Batch #4
    2cnc(C)nc2N}________
    )cc2}_______________
    >>> print('Valid lengths')
    >>> for i, batch in enumerate(sampler, start=1):
    ...     print(f'Batch #{i}')
    ...     print(batch.valid_lengths)
    Batch #1
    [7 20]
    Batch #2
    [0 11]
    Batch #3
    [20 20]
    Batch #4
    [12 5]

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
            vocabulary: SMILESVocabulary,
            batch_size: int,
            n_steps: int,
            shuffle: bool = True,
    ):
        self._vocabulary = vocabulary
        self._shuffle = shuffle

        self.batch_size = batch_size
        self.n_steps = n_steps

    @property
    def vocabulary(self) -> SMILESVocabulary:
        """The corpus of the original data set.

        Returns
        -------
        vocabulary : SMILESVocabulary
        """
        return self._vocabulary

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
        """Return the number of samples.
        """
        n_samples = 0
        for n_samples, _ in enumerate(iter(self), start=1):
            pass
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
            random.shuffle(self._vocabulary.corpus)

        n_batches = (
            len(self._vocabulary.corpus) // self.batch_size
            * self.batch_size
        )

        for i_batch in range(0, n_batches, self.batch_size):
            curr_slice = slice(i_batch, i_batch + self.batch_size)
            batch = self._pad(self._vocabulary.corpus[curr_slice])

            yield from self._iter_steps(batch)

    def _iter_steps(self, batch: np.ndarray) -> Generator[Batch, None, None]:
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

        def get_valid_lengths(sample: np.ndarray) -> np.ndarray:
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

            return np.array(lengths, dtype=int)

        pad_token_idx: int = self._vocabulary[Token.PAD]

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

    def _pad(self, item_lists: List[List[int]]) -> np.ndarray:
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

        pad_token_idx: int = self._vocabulary[Token.PAD]
        new_items_list: List[List[int]] = []

        for item_list in item_lists:
            pad_len = max_len - len(item_list)
            pad_list = [pad_token_idx] * pad_len
            new_items_list.append(item_list + pad_list)

        return np.array(new_items_list, dtype=int)

    @classmethod
    def init_states(
            cls,
            model: Any,
            mini_batch: Batch,
            states: Optional[List[np.ndarray]] = None,
            init_state_func: Optional[
                Callable[[Any], nd.ndarray.NDArray]] = None,
            ctx: context.Context = context.cpu(),
            detach: bool = False,
            *args,
            **kwargs,
    ) -> List[np.ndarray]:
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
        ctx : mxnet.context.Context, default mxnet.context.cpu()
            CPU or GPU.
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
        """
        if mini_batch.init_state or states is None:
            states = super().init_states(
                model=model,
                mini_batch=mini_batch,
                states=states,
                init_state_func=init_state_func,
                ctx=ctx,
            )

        if detach:
            states = [state.detach() for state in states]

        return states


class SMILESBatchSampler(BatchSampler, StateInitializerMixin):
    """Generate (mini-)batches of SMILES (sub)sequences. Extends
    mxnet.gluon.data.BatchSampler API by implementing `init_states` for state
    initialization.

    Examples
    --------
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> from moleculegen.tests.utils import TempSMILESFile
    >>> smiles_strings = (
    'CN=C=O\n'
    'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1\n'
    'OCCc1c(C)[n+](cs1)Cc2cnc(C)nc2N\n'
    'CC(=O)NCCC1=CNc2c1cc(OC)cc2'
    )
    >>> with TempSMILESFile(smiles_strings=smiles_strings) as temp_fh:
    ...     dataset = SMILESDataset(temp_fh.file_handler.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESConsecutiveSampler(vocabulary, 20, shuffle=False)
    >>> batch_sampler = SMILESBatchSampler(sampler, 2)
    >>> for batch_i, batch in enumerate(batch_sampler, start=1):
    ...     print(f'Batch {batch_i}:')
    ...     for sample_i in range(batch.shape[0]):
    ...         input_sample = batch.inputs[sample_i].tolist())
    ...         output_sample = batch.outputs[sample_i].tolist()
    ...         print(''.join(vocabulary.get_tokens(input_sample))
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
    Valid lengths: [20 11].

    Batch 3:
    {CC(=O)NCCC1=CNc2c1cc(O
    CC(=O)NCCC1=CNc2c1cc(OC
    C)cc2}______________
    )cc2}_______________
    Valid lengths: [20  5].

    Batch 4:
    {CN=C=O}____________
    CN=C=O}_____________
    Valid lengths: [7].

    See also
    --------
    SMILESBatchColumnSampler
    """

    def __iter__(self) -> Generator[Batch, None, None]:
        for batch in super().__iter__():
            inputs = (sample.inputs for sample in batch)
            outputs = (sample.outputs for sample in batch)
            valid_lengths = (sample.valid_length for sample in batch)

            yield Batch(
                inputs=np.array(list(inputs), dtype=int),
                outputs=np.array(list(outputs), dtype=int),
                valid_lengths=np.array(list(valid_lengths), dtype=int),
            )


class SMILESConsecutiveSampler(Sampler):
    """Consecutively generate SMILES (sub)sequences of length `n_steps`,
    padding the lacking tokens if necessary. By default, the generated objects
    are of type Sample, which has attributes `inputs` (input sequences),
    `outputs` (output sequences shifted by one step), and `valid_length`
    (the number of valid tokens in `outputs`).

    This class generates only one sample at a time. To sample mini-batches,
    refer to mxnet.gluon.data.BatchSampler.

    Parameters
    ----------
    vocabulary : SMILESVocabulary
        The SMILES vocabulary containing the loaded corpus.
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
    >>> from moleculegen.data import SMILESDataset, SMILESVocabulary
    >>> from moleculegen.tests.utils import TempSMILESFile
    >>> smiles_string = 'CCc1c[n+]2ccc3c4ccccc4[nH]c3c2cc1'
    >>> with TempSMILESFile(smiles_strings=smiles_string) as temp_fh:
    ...     dataset = SMILESDataset(temp_fh.file_handler.name)
    >>> vocabulary = SMILESVocabulary(dataset, need_corpus=True)
    >>> sampler = SMILESConsecutiveSampler(vocabulary, n_steps=20)
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
    11
    """

    @dataclasses.dataclass(eq=False, frozen=True)
    class Sample:
        inputs: List[int]   # Input tokens.
        outputs: List[int]  # Output tokens.
        valid_length: int   # The number of valid tokens in `outputs`.

    def __init__(
            self,
            vocabulary: SMILESVocabulary,
            n_steps: Optional[int] = None,
            shuffle: bool = True,
            *,
            sample_type: str = 'sample',
    ):
        self._vocabulary = vocabulary
        self._shuffle = shuffle
        self._n_steps = n_steps or max(map(len, self._vocabulary.corpus))

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

    def __len__(self) -> int:
        """Return the number of samples.
        """
        n_samples = 0
        for n_samples, _ in enumerate(iter(self), start=1):
            pass
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
            random.shuffle(self._vocabulary.corpus)

        # Iterate over the corpus of SMILES token indices.
        for tokens in self._vocabulary.corpus:
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

            # If the last subsequence is of length < `self._n_steps`,
            remainder_len = step_len - len(tokens) + 1
            if remainder_len > 0 and remainder_len != self._n_steps:
                # fill the lacking tokens with Token.PAD index.
                pad = [self._vocabulary[Token.PAD]]

                yield self._sample_type(
                    tokens[step_i: step_len] + pad*(remainder_len-1),
                    tokens[step_i+1: step_len+1] + pad*remainder_len,
                    self._n_steps - remainder_len,
                )

    def __call__(self) -> Iterator[Union[Sample, Tuple[List[int], List[int]]]]:
        """Return a sampler iterator.
        No need to use this syntactic sugar explicitly w/ mxnet API. Created
        only for tensorflow's Dataset or similar APIs.
        """
        return iter(self)
