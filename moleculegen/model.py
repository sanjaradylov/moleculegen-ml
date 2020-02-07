"""
Language models for generation of novel molecules.

Classes
-------
SMILESRNNModel
    Recurrent neural network to encode-decode SMILES strings.
"""

from typing import Callable, List, Tuple, Union
from mxnet import gluon, nd


class SMILESRNNModel(gluon.Block):
    """Recurrent neural network generating SMILES strings of novel molecules.
    Follows gluon's Block API.

    Parameters
    ----------
    rnn_layer : gluon.rnn._RNNLayer
        Recurrent layer.
    dense_layer : gluon.nn.Dense or gluon.nn.Sequential
        Dense layer.
    vocab_size : int
        Number of unique tokens.
    kwargs
        Block parameters.
    """

    def __init__(
            self,
            rnn_layer: Union[
                gluon.rnn.GRU,
                gluon.rnn.LSTM,
                gluon.rnn.RNN,
            ],
            dense_layer: Union[
                gluon.nn.Dense,
                gluon.nn.Sequential,
            ],
            vocab_size: int,
            **kwargs,
    ):
        super().__init__(**kwargs)

        self.rnn_layer = rnn_layer
        self.dense_layer = dense_layer
        self.vocab_size = vocab_size

    def begin_state(
            self,
            batch_size: int = 0,
            func: Callable = nd.zeros,
            **kwargs,
    ) -> List[nd.NDArray]:
        """Return initial state for each element in mini-batch.

        Parameters
        ----------
        batch_size : int, default 0
            Batch size.
        func : callable, default mxnet.nd.zeros
            Function to create initial state.

        Returns
        -------
        state : list
            Initial state.
        """
        return self.rnn_layer.begin_state(
            batch_size=batch_size, func=func, **kwargs)

    def forward(
            self,
            inputs: nd.NDArray,
            state: List[nd.NDArray],
    ) -> Tuple[nd.NDArray, List[nd.NDArray]]:
        """Run forward computation.

        Parameters
        ----------
        inputs : mxnet.nd.NDArray, shape = (n_steps, batch_size, vocab_size)
            Input samples.
        state : list
            Hidden state.

        Returns
        -------
        output : tuple
            X : mxnet.nd.NDArray, shape = (n_steps, batch_size, vocab_size)
                Output at current step.
            H : list
                Hidden state output.
        """
        inputs_oh = nd.one_hot(inputs.T, self.vocab_size)
        outputs, state = self.rnn_layer(inputs_oh, state)
        outputs = self.dense_layer(outputs.reshape((-1, outputs.shape[-1])))
        return outputs, state
