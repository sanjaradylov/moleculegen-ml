"""
Gluon objects and sequential models used in the subpackage.
"""

from typing import Callable, Dict, List, Optional, Union

import mxnet as mx
from mxnet import gluon

from .._types import ActivationT, ContextT, InitializerT


# Available RNNs.
RNN_MAP: Dict[str, type] = {
    'vanilla': gluon.rnn.RNN,
    'lstm': gluon.rnn.LSTM,
    'gru': gluon.rnn.GRU,
}

INIT_MAP: Dict[str, mx.init.Initializer] = {
    'normal': mx.init.Normal(),
    'uniform': mx.init.Uniform(),
    'orthogonal_uniform': mx.init.Orthogonal(rand_type='uniform'),
    'orthogonal_normal': mx.init.Orthogonal(rand_type='normal'),
    'xavier_uniform': mx.init.Xavier(rnd_type='uniform'),
    'xavier_normal': mx.init.Xavier(rnd_type='gaussian')
}


CTX_MAP: Dict[str, Callable[[int], mx.context.Context]] = {
    'cpu': mx.context.cpu,
    'gpu': mx.context.gpu,
}


def get_ctx(ctx: str) -> ContextT:
    if ctx is None or isinstance(ctx, mx.context.Context):
        return ctx or mx.context.cpu()

    ctx_name, *ctx_ids = ctx.split(':')
    ctx_f = CTX_MAP[ctx_name]

    if len(ctx_ids) == 0:  # ctx == 'gpu'
        return ctx_f(0)
    elif len(ctx_ids) == 1:  # e.g. ctx == 'gpu:0'
        return ctx_f(int(ctx_ids[0]))
    else:  # e.g. ctx == 'gpu:0:1:2'
        return [ctx_f(int(id_)) for id_ in ctx_ids]


def get_init(initializer: InitializerT) -> Optional[mx.init.Initializer]:
    if isinstance(initializer, mx.init.Initializer):
        return initializer
    return initializer and INIT_MAP[initializer]


def mlp(
        *,
        n_layers: int,
        n_units: Union[int, List[int]],
        activation: Union[ActivationT, List[ActivationT]],
        output_dim: int,
        dtype: str,
        dropout: Union[float, List[float]],
        init: InitializerT,
        prefix: Optional[str] = 'decoder_',
        params: Optional[mx.gluon.ParameterDict] = None,
) -> Union[gluon.nn.Dense, gluon.nn.HybridSequential]:
    """A single dense layer or an MLP built with mxnet.gluon.nn.HybridSequential.
    """
    n_hidden_layers = n_layers - 1

    if isinstance(n_units, int):
        n_units = [n_units] * n_hidden_layers
    elif len(n_units) != n_hidden_layers:
        raise ValueError(f'`n_units` must contain exactly `n_layers-1` elements.')

    if not isinstance(activation, list):
        activation = [activation] * n_hidden_layers
    elif len(activation) != n_hidden_layers:
        raise ValueError(f'`activation` must contain exactly `n_layers-1` elements.')

    if isinstance(dropout, float):
        dropout = [dropout] * n_hidden_layers
    elif len(dropout) != n_hidden_layers:
        raise ValueError(f'`dropout` must contain exactly `n_layers-1` elements.')

    output_dense = gluon.nn.Dense(
        units=output_dim,
        dtype=dtype,
        flatten=False,
        weight_initializer=init,
        prefix=prefix if n_layers == 1 else f'{prefix}out_',
        params=params,
    )

    if n_layers == 1:
        return output_dense

    net = gluon.nn.HybridSequential(prefix=f'{prefix.rstrip("_")}seq_')
    for i in range(n_hidden_layers):
        net.add(gluon.nn.Dense(
            units=n_units[i],
            activation=activation[i],
            dtype=dtype,
            flatten=False,
            weight_initializer=init,
            prefix=f'{prefix}l{i}_',
        ))
        if dropout != 0.:
            net.add(gluon.nn.Dropout(dropout[i]))

    net.add(output_dense)

    return net
