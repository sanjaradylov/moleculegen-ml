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
from typing import AnyStr, Iterator, List, Optional, Tuple

from mxnet import nd
from mxnet.gluon.data import SimpleDataset

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
                smiles_strings.append(line)

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

    # TODO Revise sampling methods
    def __iter__(self) -> Iterator[Tuple[nd.NDArray, nd.NDArray]]:
        """Iterate over the samples of the corpus.
        Strategy: sequential partitioning.

        Yields
        ------
        sample : tuple
            inputs : mxnet.nd.NDArray
            outputs : mxnet.nd.NDArray
        """
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
