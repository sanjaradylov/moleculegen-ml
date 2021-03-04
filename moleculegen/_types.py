"""
Common data type annotations used in the project.
"""

from typing import Callable, List, Literal, Union

from mxnet import context, gluon, init, nd, optimizer


Activations = Literal['sigmoid', 'tanh', 'relu', 'softrelu', 'softsign']
ActivationT = Union[
    None, Activations, gluon.nn.Activation, Callable[[nd.NDArray], nd.NDArray]]

ContextT = Union[context.Context, List[context.Context]]

Initializers = Literal['uniform', 'normal', 'orthogonal', 'xavier']
InitializerT = Union[None, Initializers, init.Initializer]

Optimizers = Literal[
    'sgd', 'nag', 'adagrad', 'rmsprop', 'adadelta', 'adam', 'nadam', 'ftml']
OptimizerT = Union[Optimizers, optimizer.Optimizer]
