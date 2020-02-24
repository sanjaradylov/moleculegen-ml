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

from mxnet import nd
from mxnet.gluon.data import SimpleDataset

from .utils import EOF
from .vocab import Vocabulary


class SMILESDataset(SimpleDataset):
    """Text data set comprising SMILES strings. Every line is presented as
    a SMILES string.

    Parameters
    ----------
    filename : str
        Path to the text file.
    encoding : str, default 'ascii'
        File encoding format.
    """

    def __init__(
            self,
            filename: AnyStr,
            encoding: Optional[str] = 'ascii',
    ):
        self._filename = filename
        self._encoding = encoding

        smiles_strings = self._read()
        super().__init__(smiles_strings)

    def _read(self) -> List[str]:
        """Read SMILES strings from the specified filename.

        Returns
        -------
        smiles_strings : list
            Non-empty SMILES strings.
        """
        smiles_strings = []

        with io.open(self._filename, encoding=self._encoding) as fh:
            for line in fh:
                if not line:
                    continue
                smiles_strings.append(line.strip() + EOF)

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
            dataset: SMILESDataset = None,
            vocab: Vocabulary = None,
    ):
        assert dataset or vocab, 'Pass either `dataset` or `vocab`.'

        self.__vocab = vocab or Vocabulary(dataset=dataset, need_corpus=True)

        self.batch_size = batch_size
        self.n_steps = n_steps

    @property
    def vocab(self) -> Vocabulary:
        """Return the corpus of the original data set.
        """
        return self.__vocab

    @property
    def batch_size(self) -> int:
        """Return the number of samples to generate at each step.
        """
        return self._batch_size

    @batch_size.setter
    def batch_size(self, batch_size: int):
        """Update the number of samples (mini-batches).
        """
        assert batch_size >= 1, 'Batch size must be positive non-zero.'
        self._batch_size = batch_size

    @property
    def n_steps(self) -> int:
        """Return the number of steps.
        """
        return self._n_steps

    @n_steps.setter
    def n_steps(self, n_steps: int):
        """Update the number of (time) steps.
        """
        assert n_steps >= 1, 'Number of steps must be positive non-zero.'
        self._n_steps = n_steps

    def sequential_sample(self) -> Iterator[Tuple[nd.NDArray, nd.NDArray]]:
        """Iterate over the samples of the corpus.
        Strategy: sequential partitioning.

        Yields
        ------
        sample : tuple
            inputs : mxnet.nd.NDArray
            outputs : mxnet.nd.NDArray
        """
        warnings.warn(
            "This sampling method might be ineffective; try using iter.",
            category=DeprecationWarning,
        )

        offset = nd.random.randint(0, self.n_steps).asscalar()
        n_idx = (
            (
                (len(self.__vocab.corpus) - offset - 1)
                // self.batch_size
            )
            * self.batch_size
        )

        inputs = nd.array(self.__vocab.corpus[offset:(offset+n_idx)])
        inputs = inputs.reshape((self.batch_size, -1))
        outputs = nd.array(self.__vocab.corpus[(offset+1):(offset+1+n_idx)])
        outputs = outputs.reshape((self.batch_size, -1))

        self.n_batches = inputs.shape[1] // self.n_steps
        for i in range(0, self.n_batches * self.n_steps, self.n_steps):
            yield inputs[:, i:i+self.n_steps], outputs[:, i:i+self.n_steps]

    # TODO Still, revise sampling methods.
    def __iter__(self) -> Iterator[Tuple[nd.NDArray, nd.NDArray]]:
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
        mini_batch : tuple
            inputs : nd.NDArray, shape = (`self.batch_size`, `self.n_steps`)
            outputs : nd.NDArray, shape = (`self.batch_size`, `self.n_steps`)
        """
        for i_batch in range(0, len(self.vocab.corpus), self.batch_size):
            curr_slice = slice(i_batch, i_batch + self.batch_size)
            batch = self.__pad(self.vocab.corpus[curr_slice])

            yield from self._iter_steps(batch)

    def _iter_steps(self, batch):
        """Generate batches separated by time steps.

        Parameters
        ----------
        batch : nd.NDArray, shape = (self.batch_size, max_smiles_len)
            Batch.

        Yields
        ------
        result : tuple
            inputs : nd.NDArray, shape = (self.batch_size, self.n_steps)
            outputs : nd.NDArray, shape = (self.batch_size, self.n_steps)
        """
        inputs = batch[:, :-1]
        outputs = batch[:, 1:]

        for i_step in range(0, batch.shape[1] - self.n_steps):
            step_slice = slice(i_step, i_step + self.n_steps)
            yield inputs[:, step_slice], outputs[:, step_slice]

    def __pad(self, item_lists: List[List[int]]) -> nd.NDArray:
        """Get a batch of SMILES token lists and fill with the `PAD` token ids
        those lists with lesser number of tokens.

        Parameters
        ----------
        item_lists : list
            List of token lists, each token list contains token indices and
            constitutes a SMILES string.

        Returns
        -------
        new_item_list : nd.NDArray
             The array of token lists.
        """
        max_len = max(map(len, item_lists))
        pad_token_idx = len(self.vocab)
        new_items_list = []

        for item_list in item_lists:
            if len(item_list) != max_len:
                pad_len = max_len - len(item_list)
                pad_list = [pad_token_idx] * pad_len
                new_items_list.append(item_list + pad_list)
            else:
                new_items_list.append(item_list)

        new_items_list = nd.array(new_items_list, dtype=int)
        return new_items_list
