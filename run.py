#!/usr/bin/env python3

"""
Learn language models on SMILES strings of given molecules. Generate and
evaluate new molecules.
"""

__author__ = 'Sanjar Ad[iy]lov'


import argparse
import datetime
import statistics
import time
from typing import Any, Dict, Union

from mxnet import autograd, context, gluon, init, npx, optimizer, random

from moleculegen import (
    SpecialTokens,
    get_mask_for_loss,
    OneHotEncoder,
    SMILESDataset,
    SMILESDataLoader,
    SMILESRNNModel,
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
                        (default: 128)
        -s N_STEPS, --n_steps N_STEPS
                        The number of time steps. (default: 20)
        -u HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        The number of units in a network's hidden state.
                        (default: 1024)
        -n N_LAYERS, --n_layers N_LAYERS
                        The number of hidden layers. (default: 3)
        -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate. (default: 0.5)
        -e N_EPOCHS, --n_epochs N_EPOCHS
                        The number of epochs. (default: 20)
        -p PREDICT_EPOCH, --predict_epoch PREDICT_EPOCH
                        Predict new strings every p epochs (default: 50)
        -v VERBOSE, --verbose VERBOSE
                        Print logs every v iterations. (default: 50)
        -c {cpu,CPU,gpu,GPU}, --ctx {cpu,CPU,gpu,GPU}
                        CPU or GPU (default: gpu)
        -r PREFIX, --prefix PREFIX
                        Initial symbol(s) of a SMILES string to generate.
                        (default: ^)
        -m MAX_GEN_LENGTH, --max_gen_length MAX_GEN_LENGTH
                        Maximum number of tokens to generate. (default: 100)
        -g GRAD_CLIP_LENGTH, --grad_clip_length GRAD_CLIP_LENGTH
                        The radius by which a gradient's length is
                        constrained. (default: 5.0)
    """
    options = process_options()

    dataset = SMILESDataset(filename=options.filename)
    dataloader = SMILESDataLoader(
        batch_size=options.batch_size,
        n_steps=options.n_steps,
        dataset=dataset,
    )

    embedding_layer = OneHotEncoder(len(dataloader.vocab))
    rnn_layer = gluon.rnn.LSTM(
        hidden_size=options.hidden_size,
        num_layers=options.n_layers,
        dropout=0.2,
    )
    dense_layer = gluon.nn.Sequential()
    dense_layer.add(
        gluon.nn.Dense(len(dataloader.vocab)),
    )
    model = SMILESRNNModel(
        embedding_layer=embedding_layer,
        rnn_layer=rnn_layer,
        dense_layer=dense_layer,
    )
    optimizer_params = {
        'learning_rate': options.learning_rate,
        'clip_gradient': options.grad_clip_length,
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
        max_gen_length=options.max_gen_length,
    )


def train(
        dataloader: SMILESDataLoader,
        model: SMILESRNNModel,
        optimizer_params: Dict[str, Any],
        opt: Union[str, optimizer.Optimizer] = 'adam',
        n_epochs: int = 1,
        predict_epoch: int = 20,
        loss_fn: gluon.loss.Loss = gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False,
            sparse_label=True,
        ),
        verbose: int = 0,
        ctx: context.Context = context.cpu(0),
        prefix: str = SpecialTokens.BOS.value,
        max_gen_length: int = 100,
):
    """Fit `model` with data from `dataloader`.

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
    max_gen_length : int, default 100
        Maximum number of tokens to generate.
    """
    model.initialize(ctx=ctx, force_reinit=True, init=init.Normal(sigma=0.1))
    trainer = gluon.Trainer(model.collect_params(), opt, optimizer_params)
    time_list = []

    for epoch in range(1, n_epochs + 1):

        start_time = None
        if verbose > 0:
            print(f'\nEpoch: {epoch:>3}\n')
            start_time = time.time()

        states = None
        loss_list = []

        for batch_no, batch in enumerate(dataloader, start=1):
            curr_batch_size = batch.x.shape[0]

            if batch.s:
                states = model.begin_state(batch_size=curr_batch_size, ctx=ctx)
            else:
                states = [state.detach() for state in states]

            inputs = batch.x.as_in_context(ctx)
            outputs = batch.y.T.reshape((-1,)).as_in_context(ctx)

            with autograd.record():
                p_outputs, states = model(inputs, states)
                label_mask = get_mask_for_loss(inputs.shape, batch.v_y)
                label_mask = label_mask.T.reshape((-1,)).as_in_context(ctx)
                loss = loss_fn(p_outputs, outputs, label_mask)

            loss.backward()
            trainer.step(batch_size=curr_batch_size)

            if (batch_no - 1) % verbose == 0:
                mean_loss = loss.mean().item()
                loss_list.append(mean_loss)
                print(
                    f'Batch: {batch_no:>6}, '
                    f'Loss: {mean_loss:>3.3f}'
                )

            if (batch_no - 1) % predict_epoch == 0:
                smiles = model.generate(
                    dataloader.vocab,
                    prefix=prefix,
                    max_length=max_gen_length,
                    ctx=ctx,
                )
                print(f'Molecule:\n    {smiles}')

        if verbose > 0:
            print(
                f'\nMean loss: '
                f'{statistics.mean(loss_list):.3f} '
                f'(+/- {statistics.stdev(loss_list):.3f})'
            )

            seconds_left = time.time() - start_time
            time_list.append(seconds_left)
            exec_time = datetime.timedelta(seconds=seconds_left)
            print(f'Execution time: {exec_time}')

    if verbose > 0:
        total_time = datetime.timedelta(seconds=sum(time_list))
        print(f'\nTotal execution time: {total_time}.')


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
        default=128,
    )
    parser.add_argument(
        '-s', '--n_steps',
        help='The number of time steps.',
        type=int,
        default=50,
    )
    parser.add_argument(
        '-u', '--hidden_size',
        help="The number of units in a network's hidden state.",
        type=int,
        default=1024,
    )
    parser.add_argument(
        '-n', '--n_layers',
        help='The number of hidden layers.',
        type=int,
        default=2,
    )
    parser.add_argument(
        '-l', '--learning_rate',
        help='The learning rate.',
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs.',
        type=int,
        default=20,
    )
    parser.add_argument(
        '-p', '--predict_epoch',
        help='Predict new strings every p iterations.',
        type=int,
        default=50,
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Print logs every v iterations.',
        type=int,
        default=50,
    )
    parser.add_argument(
        '-c', '--ctx',
        help='CPU or GPU.',
        default='gpu',
        choices=('cpu', 'CPU', 'gpu', 'GPU'),
    )
    parser.add_argument(
        '-r', '--prefix',
        help='Initial symbol(s) of a SMILES string to generate.',
        default=SpecialTokens.BOS.value,
    )
    parser.add_argument(
        '-m', '--max_gen_length',
        help='Maximum number of tokens to generate.',
        type=int,
        default=100,
    )
    parser.add_argument(
        '-g', '--grad_clip_length',
        help="The radius by which a gradient's length is constrained.",
        type=float,
        default=5.0,
    )

    return parser.parse_args()


if __name__ == '__main__':
    npx.set_np()
    random.seed(0)
    main()
