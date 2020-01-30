#!/usr/bin/env python3

"""
Learn language models on SMILES strings of given molecules. Generate and
evaluate new molecules.
"""

__author__ = 'Sanjar Ad[iy]lov'


import argparse
from typing import Any, Dict, Union

from mxnet import autograd, context, gluon, init, metric, nd, optimizer

from moleculegen import (
    EOF,
    SMILESDataset,
    SMILESDataLoader,
    SMILESRNNModel,
    Vocabulary,
)


def main():
    """Main function: load data comprising molecules, create RNN,
    fit RNN with the data, and predict novel molecules.

    Command line options:

    positional arguments:
        filename        The path to the training data containing SMILES
                        strings.

    optional arguments:
        -h, --help      show this help message and exit
        -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The number of batches to generate at every iteration.
                        (default: 32)
        -s N_STEPS, --n_steps N_STEPS
                        The number of time steps. (default: 40)
        -u HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        The number of units in a network's hidden state.
                        (default: 256)
        -n N_LAYERS, --n_layers N_LAYERS
                        The number of hidden layers. (default: 1)
        -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate. (default: 1.0)
        -e N_EPOCHS, --n_epochs N_EPOCHS
                        The number of epochs. (default: 2000)
        -p PREDICT_EPOCH, --predict_epoch PREDICT_EPOCH
                        Predict new strings every p epochs (default: 20)
        -v VERBOSE, --verbose VERBOSE
                        Print logs every v iterations. (default: 10)
        -c {cpu,CPU,gpu,GPU}, --ctx {cpu,CPU,gpu,GPU}
                        CPU or GPU (default: cpu)
        -r PREFIX, --prefix PREFIX
                        Initial symbol(s) of a SMILES string to generate.
                        (default: C)
    """
    options = process_options()

    dataset = SMILESDataset(filename=options.filename)
    dataloader = SMILESDataLoader(
        batch_size=options.batch_size,
        n_steps=options.n_steps,
        dataset=dataset,
    )

    rnn_layer = gluon.rnn.GRU(
        hidden_size=options.hidden_size,
        num_layers=options.n_layers,
    )
    model = SMILESRNNModel(
        rnn_layer=rnn_layer,
        vocab_size=len(dataloader.vocab),
    )
    optimizer_params = {
        'learning_rate': options.learning_rate,
        # TODO Add gradient clipping
        # 'clip_gradient': 1,
    }
    ctx = {
        'cpu': context.cpu(0),
        'gpu': context.gpu(0),
    }

    train(
        dataloader=dataloader,
        model=model,
        optimizer_params=optimizer_params,
        n_epochs=options.n_epochs,
        predict_epoch=options.predict_epoch,
        verbose=options.verbose,
        ctx=ctx[options.ctx.lower()],
        prefix=options.prefix,
    )


def train(
        dataloader: SMILESDataLoader,
        model: SMILESRNNModel,
        optimizer_params: Dict[str, Any],
        opt: Union[str, optimizer.Optimizer] = 'adam',
        n_epochs: int = 1,
        predict_epoch: int = 20,
        loss_fn: gluon.loss.Loss = gluon.loss.SoftmaxCrossEntropyLoss(),
        verbose: int = 0,
        ctx: context.Context = context.cpu(0),
        prefix: str = 'C',
):
    """Fit `model` with `data`.

    Parameters
    ----------
    dataloader : SMILESDataLoader
        SMILES data loader.
    model : SMILESRNNModel
        Language model to fit.
    optimizer_params : dict
        gluon.Trainer optimizer_params.
    opt : str or optimizer.Optimizer, default 'adam'
        Optimizer constructor.
    n_epochs : int, default 1
        Number of train epochs.
    predict_epoch : int, default 20
        Predict a new string every 20 iterations.
    ctx : mxnet.context.Context, default context.cpu(0)
        CPU or GPU.
    loss_fn : gluon.loss.Loss, default gluon.loss.SoftmaxCrossEntropyLoss()
        Loss function.
    verbose : int, default 0
        Print logs every `verbose` steps.
    prefix : str, default 'C'
        The initial tokens of the string being generated.
    """
    model.initialize(ctx=ctx, force_reinit=True, init=init.Xavier())
    trainer = gluon.Trainer(model.collect_params(), opt, optimizer_params)
    state = None

    for epoch in range(1, n_epochs + 1):

        perplexity = metric.Perplexity(ignore_label=None)

        for inputs, outputs in dataloader:
            if state is None:
                state = model.begin_state(
                    batch_size=dataloader.batch_size, ctx=ctx)
            else:
                for unit in state:
                    unit.detach()

            inputs = inputs.as_in_context(ctx)
            outputs = outputs.T.reshape((-1,)).as_in_context(ctx)

            with autograd.record():
                p_outputs, state = model(inputs, state)
                p_outputs = p_outputs.softmax()
                loss = loss_fn(p_outputs, outputs).mean()

            loss.backward()
            trainer.step(batch_size=1)
            perplexity.update(outputs, p_outputs)

        if epoch % verbose == 0:
            print(f'Epoch {epoch:>4}, perplexity {perplexity.get()[1]:>4.3f}')

        if epoch % predict_epoch == 0:
            print(predict(prefix, model, dataloader.vocab, 50, ctx))


def predict(
        prefix: str,
        model: SMILESRNNModel,
        vocab: Vocabulary,
        n_steps: int,
        ctx: context.Context = context.cpu(0),
) -> str:
    """Predict SMILES string starting with `prefix`.

    Parameters
    ----------
    prefix : str
        The initial tokens of the string being generated.
    model : SMILESRNNModel
        RNN.
    vocab : Vocabulary
        Vocabulary.
    n_steps : int
        The number of time steps.
    ctx : mxnet.context.Context
        CPU or GPU.

    Returns
    -------
    s : str
        Predicted (generated) SMILES string.
    """

    def get_input() -> nd.NDArray:
        """Get the last token from the output at current step.

        Returns
        -------
        token_nd : mxnet.nd.NDArray
            The last token at current step.
        """
        nonlocal outputs
        return nd.array([outputs[-1]], ctx=ctx).reshape((1, 1))

    state = model.begin_state(batch_size=1, ctx=ctx)
    outputs = [vocab[prefix[0]]]

    for token in prefix[1:]:
        output, state = model(get_input(), state)
        outputs.append(vocab[token])

    for step in range(n_steps):
        output, state = model(get_input(), state)

        output_id = int(output.argmax(axis=1).reshape(1).asscalar())
        char = vocab.idx_to_token[output_id]
        if char == EOF:
            break

        outputs.append(output_id)

    return ''.join(vocab.idx_to_token[i] for i in outputs)


def process_options() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    namespace : argparse.Namespace
        Command line attributes and its values.
    """
    parser = argparse.ArgumentParser(
        description='Generate novel molecules with recurrent neural networks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'filename',
        help='The path to the training data containing SMILES strings.',
    )
    parser.add_argument(
        '-b', '--batch_size',
        help='The number of batches to generate at every iteration.',
        type=int,
        default=32,
    )
    parser.add_argument(
        '-s', '--n_steps',
        help='The number of time steps.',
        type=int,
        default=40,
    )
    parser.add_argument(
        '-u', '--hidden_size',
        help="The number of units in a network's hidden state.",
        type=int,
        default=256,
    )
    parser.add_argument(
        '-n', '--n_layers',
        help='The number of hidden layers.',
        type=int,
        default=1,
    )
    parser.add_argument(
        '-l', '--learning_rate',
        help='The learning rate.',
        type=float,
        default=1.0,
    )
    parser.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs.',
        type=int,
        default=2000,
    )
    parser.add_argument(
        '-p', '--predict_epoch',
        help='Predict new strings every p epochs.',
        type=int,
        default=20,
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Print logs every v iterations.',
        type=int,
        default=10,
    )
    parser.add_argument(
        '-c', '--ctx',
        help='CPU or GPU.',
        default='cpu',
        choices=('cpu', 'CPU', 'gpu', 'GPU'),
    )
    parser.add_argument(
        '-r', '--prefix',
        help='Initial symbol(s) of a SMILES string to generate.',
        default='C',
    )

    return parser.parse_args()


main()
