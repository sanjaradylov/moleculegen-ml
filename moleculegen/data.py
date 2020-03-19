"""
Utilities to load SMILES data sets.

Classes
-------
SMILESDataset
    Load text data set containing SMILES strings.
SMILESDataLoader
    Generate data samples from data set.
"""

import io
import warnings
from typing import AnyStr, Iterator, List, Optional, Tuple

from mxnet import np
from mxnet.gluon.data import SimpleDataset

from .utils import Batch, SpecialTokens
from .vocab import Vocabulary


class SMILESDataset(SimpleDataset):
    """Text data set comprising SMILES strings. Every line is presented as
    a SMILES string.

    Parameters
    ----------
    filename : str
        Path to the text file.
    encoding : {'ascii', 'utf8'}, default 'ascii'
        File encoding format.
    """

    def __init__(
            self,
            filename: AnyStr,
            encoding: str = 'ascii',
    ):
        self.__filename = filename
        self.__encoding = encoding

        smiles_strings = self._read()
        super().__init__(smiles_strings)

    @property
    def filename(self):
        """The filename of the data set to read.

        Returns
        -------
        filename : str
        """
        return self.__filename

    @property
    def encoding(self):
        """The file encoding format.

        Returns
        -------
        encoding : {'ascii', 'utf8'}
        """
        return self.__encoding

    def _read(self) -> List[str]:
        """Read SMILES strings from the specified filename.

        Returns
        -------
        smiles_strings : list
            Non-empty SMILES strings.
        """
        smiles_strings: List[str] = []

        with io.open(self.__filename, encoding=self.__encoding) as fh:
            for line in fh:
                if not line:
                    continue
                smiles_strings.append(line.strip() + SpecialTokens.EOS.value)

        return smiles_strings


class SMILESDataLoader:
    """Iterator to load SMILES strings.

    Parameters
    ----------
    batch_size : int
        Number of samples to generate.
    n_steps : int
        Number of (time) steps.
    dataset : SMILESDataset, default None
        Data set containing SMILES strings.
        Either `dataset` or `vocab` must be specified.
    vocab : Vocabulary, default None
        Vocabulary instance of original data.
        Either `dataset` or `vocab` must be specified.
    """

    def __init__(
            self,
            batch_size: int,
            n_steps: int,
            dataset: Optional[SMILESDataset] = None,
            vocab: Optional[Vocabulary] = None,
    ):
        assert dataset or vocab, 'Pass either `dataset` or `vocab`.'

        self._vocab = vocab or Vocabulary(dataset=dataset, need_corpus=True)

        self.batch_size = batch_size
        self.n_steps = n_steps

    @property
    def vocab(self) -> Vocabulary:
        """The corpus of the original data set.

        Returns
        -------
        vocab : Vocabulary
        """
        return self._vocab

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
        assert batch_size >= 1, 'Batch size must be positive non-zero.'
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
        assert n_steps >= 1, 'Number of steps must be positive non-zero.'
        self.__n_steps = n_steps

    def sequential_sample(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Iterate over the samples of the corpus.
        Strategy: sequential partitioning.

        Yields
        ------
        sample : tuple
            inputs : mxnet.np.ndarray
            outputs : mxnet.np.ndarray

        Warns
        -----
        DeprecationWarning
            For now, this sampling method is considered to be ineffective.
        """
        warnings.warn(
            "This sampling method might be ineffective; try using iter.",
            category=DeprecationWarning,
        )

        offset = np.random.randint(0, self.n_steps).asscalar()
        n_idx = (
            (
                (len(self._vocab.corpus) - offset - 1)
                // self.batch_size
            )
            * self.batch_size
        )

        inputs = np.array(self._vocab.corpus[offset:(offset+n_idx)])
        inputs = inputs.reshape((self.batch_size, -1))
        outputs = np.array(self._vocab.corpus[(offset+1):(offset+1+n_idx)])
        outputs = outputs.reshape((self.batch_size, -1))

        self.n_batches = inputs.shape[1] // self.n_steps
        for i in range(0, self.n_batches * self.n_steps, self.n_steps):
            yield inputs[:, i:i+self.n_steps], outputs[:, i:i+self.n_steps]

    # TODO Still, revise sampling methods.
    def __iter__(self) -> Iterator[Batch]:
        """Iterate over the samples of the corpus.

        Strategy:
        1. Load corpus containing lists of token IDs.
        2. For every iteration, get batch of size `self.batch_size` and
           pad token lists to have equal dimensions.
        3. The modified batch is divided into mini-batches (input, output)
           of size (`self.batch_size`, `self.n_steps`). Previous mini-batch
           differs from the current in exactly one timestep.

        Yields
        ------
        mini_batch : Batch
            Mini-batch containing inputs, outputs, and their corresponding
            valid lengths.
        """
        for i_batch in range(0, len(self._vocab.corpus), self.batch_size):
            curr_slice = slice(i_batch, i_batch + self.batch_size)
            batch = self._pad(self._vocab.corpus[curr_slice])

            yield from self._iter_steps(batch)

    def _iter_steps(self, batch: np.ndarray) -> Iterator[Batch]:
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

        # TODO There is a solution not involving matrix lookup.
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
                    if s == len(self._vocab) - 1:
                        lengths.append(i)
                        break
                else:
                    lengths.append(sample.shape[1])

            return np.array(lengths, dtype=int)

        inputs, outputs = batch[:, :-1], batch[:, 1:]

        h = True
        for i_step in range(0, inputs.shape[1], self.n_steps):
            step_slice = (..., slice(i_step, i_step + self.n_steps))
            x, y = inputs[step_slice], outputs[step_slice]

            yield Batch(x, y, get_valid_lengths(x), get_valid_lengths(y), h)

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
        # Make it divisible by self.n_steps so that we can iterate over the
        # batch max_len // self.n_steps times and retrieve self.n_steps tokens.
        max_len = (max_len // self.n_steps + 1) * self.n_steps
        # Increment it since we divide `batch` into
        # (batch[:, :-1], batch[:, 1:]).
        max_len += 1

        pad_token_idx = len(self._vocab) - 1
        new_items_list: List[List[int]] = []

        for item_list in item_lists:
            if len(item_list) != max_len:
                pad_len = max_len - len(item_list)
                pad_list = [pad_token_idx] * pad_len
                new_items_list.append(item_list + pad_list)
            else:
                new_items_list.append(item_list)

        return np.array(new_items_list, dtype=int)
