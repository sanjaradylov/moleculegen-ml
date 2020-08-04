"""
Gluon objects and sequential models used in the subpackage.
"""

from typing import Dict

from mxnet import gluon


# Available RNNs.
RNN_MAP: Dict[str, type] = {
    'vanilla': gluon.rnn.RNN,
    'lstm': gluon.rnn.LSTM,
    'gru': gluon.rnn.GRU,
}


def dropout_mlp(
        n_layers: int,
        n_units: int,
        activation: str,
        output_dim: int,
        dtype: str,
        dropout: float,
        **kwargs,
) -> gluon.Block:
    """A dropout-regularized MLP built with mxnet.gluon.nn.Sequential.
    """
    net = gluon.nn.Sequential()

    for _ in range(n_layers-1):
        net.add(gluon.nn.Dropout(dropout))
        net.add(gluon.nn.Dense(
            units=n_units,
            dtype=dtype,
            activation=activation,
            flatten=False,
        ))

    net.add(gluon.nn.Dropout(dropout))
    net.add(gluon.nn.Dense(
        units=output_dim,
        dtype=dtype,
        flatten=False,
    ))

    return net


def mlp(
        n_layers: int,
        n_units: int,
        activation: str,
        output_dim: int,
        dtype: str,
        **kwargs,
) -> gluon.Block:
    """A single dense layer or an MLP built with mxnet.gluon.nn.Sequential.
    """
    output_dense = gluon.nn.Dense(
        units=output_dim,
        dtype=dtype,
        flatten=False,
    )

    if n_layers == 1:
        return output_dense

    net = gluon.nn.Sequential()

    for _ in range(n_layers-1):
        net.add(gluon.nn.Dense(
            units=n_units,
            dtype=dtype,
            activation=activation,
            flatten=False,
        ))

    net.add(output_dense)

    return net
