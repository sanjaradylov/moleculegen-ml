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
    SpecialTokens,
    get_mask_for_loss,
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

    rnn_layer = gluon.rnn.LSTM(
        hidden_size=options.hidden_size,
        num_layers=options.n_layers,
        dropout=0.2,
    )
    dense_layer = gluon.nn.Sequential()
    dense_layer.add(
        # gluon.nn.Dense(256, activation='relu'),
        # gluon.nn.Dropout(0.2),
        gluon.nn.Dense(len(dataloader.vocab)),
    )
    model = SMILESRNNModel(
        rnn_layer=rnn_layer,
        dense_layer=dense_layer,
        vocab_size=len(dataloader.vocab),
    )
    optimizer_params = {
        'learning_rate': options.learning_rate,
        'clip_gradient': 5,
        'wd': 1,
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
        loss_fn: gluon.loss.Loss = gluon.loss.SoftmaxCrossEntropyLoss(
            from_logits=False,
            sparse_label=True,
        ),
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

    for epoch in range(1, n_epochs + 1):

        state = None
        perplexity = metric.Perplexity(ignore_label=None)

        for batch_no, batch in enumerate(dataloader, start=1):
            if batch.s:
                state = model.begin_state(
                    batch_size=batch.x.shape[0], ctx=ctx)
            else:
                for unit in state:
                    unit.detach()

            inputs = batch.x.as_in_context(ctx)
            outputs = batch.y.T.reshape((-1,)).as_in_context(ctx)

            with autograd.record():
                p_outputs, state = model(inputs, state)
                label_mask = get_mask_for_loss(inputs.shape, batch.v_y)
                label_mask = label_mask.T.reshape((-1,)).as_in_context(ctx)
                loss = loss_fn(p_outputs, outputs, label_mask).mean()

            loss.backward()
            trainer.step(batch_size=1)
            perplexity.update(outputs, p_outputs.softmax())

            if batch_no % verbose == 0:
                print(
                    f'Loss: {loss.asscalar():>4.3f}, '
                    f'Perplexity: {perplexity.get()[1]:>4.3f}'
                )

            if batch_no % predict_epoch == 0:
                generated_molecule = predict(
                    prefix, model, dataloader.vocab, 50, ctx)
                generated_molecule = generated_molecule.strip(
                    SpecialTokens.EOS.value)
                print(f'Molecule: {generated_molecule}')

        if verbose != 0:
            print(f'\nEpoch: {epoch:>4}\n')


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
    outputs = vocab[prefix[:]]

    for token in prefix[1:]:
        output, state = model(get_input(), state)
        outputs.append(vocab[token])

    for step in range(n_steps):
        output, state = model(get_input(), state)

        output_id = int(output.argmax(axis=1).reshape(1).asscalar())
        char = vocab.idx_to_token[output_id]
        if char in (SpecialTokens.EOS.value, SpecialTokens.PAD.value):
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
        default=128,
    )
    parser.add_argument(
        '-s', '--n_steps',
        help='The number of time steps.',
        type=int,
        default=20,
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
        default=3,
    )
    parser.add_argument(
        '-l', '--learning_rate',
        help='The learning rate.',
        type=float,
        default=.5,
    )
    parser.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs.',
        type=int,
        default=10,
    )
    parser.add_argument(
        '-p', '--predict_epoch',
        help='Predict new strings every p iterations.',
        type=int,
        default=20,
    )
    parser.add_argument(
        '-v', '--verbose',
        help='Print logs every v iterations.',
        type=int,
        default=20,
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
        default='C',
    )

    return parser.parse_args()


main()
