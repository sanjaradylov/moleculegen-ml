"""
Gluon objects and sequential models used in the subpackage.
"""

from typing import Dict

import mxnet as mx
from mxnet import gluon


# Available RNNs.
RNN_MAP: Dict[str, type] = {
    'vanilla': gluon.rnn.RNN,
    'lstm': gluon.rnn.LSTM,
    'gru': gluon.rnn.GRU,
}

INIT_MAP: Dict[str, mx.init.Initializer] = {
    'normal': mx.init.Normal(),
    'orthogonal': mx.init.Orthogonal(),
    'uniform': mx.init.Uniform(),
    'xavier': mx.init.Xavier(),
}

CTX_MAP: Dict[str, mx.context.Context] = {
    'cpu': mx.context.cpu(),
    'gpu': mx.context.gpu(),
}


def dropout_mlp(
        *,
        n_layers: int,
        n_units: int,
        activation: str,
        output_dim: int,
        dtype: str,
        dropout: float,
        prefix: str,
        **kwargs,
) -> gluon.Block:
    """A dropout-regularized MLP built with mxnet.gluon.nn.HybridSequential.
    """
    net = gluon.nn.HybridSequential(prefix=prefix)

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
        *,
        n_layers: int,
        n_units: int,
        activation: str,
        output_dim: int,
        dtype: str,
        prefix: str,
        **kwargs,
) -> gluon.Block:
    """A single dense layer or an MLP built with mxnet.gluon.nn.HybridSequential.
    """
    output_dense = gluon.nn.Dense(
        units=output_dim,
        dtype=dtype,
        flatten=False,
        prefix=prefix if n_layers == 1 else None,
    )

    if n_layers == 1:
        return output_dense

    net = gluon.nn.HybridSequential(prefix=prefix)

    for _ in range(n_layers-1):
        net.add(gluon.nn.Dense(
            units=n_units,
            dtype=dtype,
            activation=activation,
            flatten=False,
        ))

    net.add(output_dense)

    return net
