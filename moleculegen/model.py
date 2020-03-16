"""
Language models for generation of novel molecules.

Classes
-------
SMILESRNNModel
    Recurrent neural network to encode-decode SMILES strings.
"""

from typing import List, Tuple, Union
from mxnet import gluon, np, npx


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
