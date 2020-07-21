"""
Language models for generation of novel molecules.

Classes
-------
OneHotEncoder
    One-hot-encoder functor.
SMILESRNNModel
    Recurrent neural network to encode-decode SMILES strings.
"""

import statistics
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

from mxnet import autograd, context, gluon, nd, np, npx, optimizer

from .base import Token
from .data.sampler import SMILESBatchColumnSampler
from .data.vocabulary import SMILESVocabulary
from .description.common import OneHotEncoder
from .evaluation.loss import get_mask_for_loss


def _distribution_partial(
        distribution: Callable,
        **distribution_args: Any,
) -> Callable:
    """Return distribution callable `distribution` with arbitrary
    (non-default) arguments `distribution_args` specified in advance. Use
    primarily for state initialization.

    Parameters
    ----------
    distribution : callable
        One of the distribution functions from mxnet.nd.random.
    distribution_args : dict, default None
        Parameters of mxnet.nd.random.uniform excluding `shape`.

    Returns
    -------
    func : callable
        Partial distribution callable.

    Raises
    ------
    ValueError
        If `shape` parameter is included in `distribution_args`.
        This parameter will be used separately in state initialization.
    """
    if 'shape' in distribution_args:
        raise ValueError('`shape` parameter should be not be specified.')

    return partial(distribution, **distribution_args)


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

    def train(
            self,
            optimizer_: optimizer.Optimizer,
            dataloader: SMILESBatchColumnSampler,
            loss_fn: gluon.loss.SoftmaxCELoss(),
            ctx: context.Context = context.cpu(),
            verbose: int = 0,
    ):
        """Train the model on the data from `dataloader` using optimization
        technique `optimizer_` minimizing `loss_fn`.

        Parameters
        ----------
        optimizer_ : mxnet.optimizer.Optimizer
            MXNet optimizer instance.
        dataloader : SMILESDataLoader
            SMILES data loader.
        loss_fn : gluon.loss.Loss, default gluon.loss.SoftmaxCELoss()
            Loss function.
        ctx : mxnet.context.Context, default context.cpu(0)
            CPU or GPU.
        verbose : int, default 0
            Print logs every `verbose` steps.
        """
        trainer = gluon.trainer.Trainer(self.collect_params(), optimizer_)
        loss_list: List[float] = []
        init_state_func: Callable = dataloader.init_state_func()
        states: Optional[List[np.ndarray]] = None

        for batch_no, batch in enumerate(dataloader, start=1):
            states = dataloader.init_states(
                model=self,
                mini_batch=batch,
                states=states,
                init_state_func=init_state_func,
                ctx=ctx,
                detach=True,
            )

            inputs = batch.inputs.as_in_ctx(ctx)
            outputs = batch.outputs.T.reshape(-1).as_in_ctx(ctx)
            valid_lengths = batch.valid_lengths.as_in_ctx(ctx)

            with autograd.record():
                p_outputs, states = self.forward(inputs, states)

                # Get a label mask, which labels 1 for any valid token and 0
                # for padding token `Token.PAD`.
                label_mask = get_mask_for_loss(inputs.shape, valid_lengths)
                label_mask = label_mask.T.reshape(-1)

                loss = loss_fn(p_outputs, outputs, label_mask)

            loss.backward()
            trainer.step(valid_lengths.sum())

            if (batch_no - 1) % verbose == 0:
                mean_loss = loss.mean().item()
                loss_list.append(mean_loss)
                print(f'Batch: {batch_no:>6}, Loss: {mean_loss:>3.3f}')

                smiles = self.generate(dataloader.vocabulary, ctx=ctx)
                print(f'Molecule: {smiles}')

        if verbose > 0:
            print(
                f'\nMean loss: '
                f'{statistics.mean(loss_list):.3f} '
                f'(+/- {statistics.stdev(loss_list):.3f})'
            )

    def generate(
            self,
            vocabulary: SMILESVocabulary,
            prefix: str = Token.BOS,
            max_length: int = 100,
            ctx: context.Context = context.cpu(),
            state_init_func: Optional[Callable] = None,
    ) -> str:
        """Generate SMILES string using (learned) model weights and states.

        Parameters
        ----------
        vocabulary : Vocabulary
            The Vocabulary instance, which provides id-to-token conversion.
        prefix : str, default: Token.BOS
            The prefix of a SMILES string to generate
        max_length : int, default: 100
            The maximum number of tokens to generate.
        ctx : context.Context, default: context.cpu()
            CPU or GPU.
        state_init_func : callable, default nd.random.uniform(-0.1, 0.1)
            Callable for state initialization.

        Returns
        -------
        smiles : str
            The SMILES string generated by the model.
        """
        state_init_func = state_init_func or _distribution_partial(
            nd.random.uniform,
            low=-0.2,
            high=0.2,
            ctx=ctx,
        )
        state = self.begin_state(batch_size=1, func=state_init_func, ctx=ctx)
        smiles = prefix

        for n_iter in range(max_length):
            input_ = np.array(vocabulary[smiles[-1]], ctx=ctx).reshape(1, 1)
            output, state = self.forward(input_, state)

            # noinspection PyUnresolvedReferences
            token_id = npx.softmax(output).argmax().astype(np.int32).item()
            token = vocabulary.idx_to_token[token_id]
            if token in (Token.EOS, Token.PAD):
                break

            smiles += token

        return smiles.lstrip(Token.BOS)
