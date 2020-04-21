#!/usr/bin/env python3

"""
Learn language models on SMILES strings of given molecules. Generate and
evaluate new molecules.
"""

__author__ = 'Sanjar Ad[iy]lov'


import argparse
import datetime
import pathlib
import statistics
import time
from typing import Any, Dict, IO, Optional, Union

from mxnet import autograd, context, gluon, init, np, npx, optimizer

from moleculegen import (
    Token,
    get_mask_for_loss,
    OneHotEncoder,
    SMILESDataset,
    SMILESDataLoader,
    SMILESRNNModel,
)


# Some useful constants to define file names.
DATE = datetime.datetime.now().strftime('%m_%d_%H_%M')
DIRECTORY = pathlib.Path(__file__).resolve().parent
PREDICTIONS_OUT_DEF = DIRECTORY / 'data' / f'{DATE}__predictions.csv'


def main():
    """Main function: load data comprising molecules, create RNN,
    fit RNN with the data, and predict novel molecules.

    Command line options:

    input/output arguments:
        stage1_data     The path to the training data containing SMILES
                        strings.
        -L MODEL_PARAMS_IN, --model_params_in MODEL_PARAMS_IN
                        The path to the model parameters. The model will load
                        these parameters and proceed fitting. (default: None)
        -S MODEL_PARAMS_OUT, --model_params_out MODEL_PARAMS_OUT
                        Save learned model parameters in this file.
                        (default: None)
        -O PREDICTIONS, --predictions PREDICTIONS
                        Save predicted molecules in this file.
                        (default: PARENT_PATH/data/DATE__predictions.csv)

    model arguments:
        -u HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        The number of units in a network's hidden state.
                        (default: 1024)
        -n N_LAYERS, --n_layers N_LAYERS
                        The number of hidden layers. (default: 2)

    hyperparameters:
        -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The number of batches to generate at every iteration.
                        (default: 128)
        -s N_STEPS, --n_steps N_STEPS
                        The number of time steps. (default: 50)
        -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate. (default: 0.0001)
        -e N_EPOCHS, --n_epochs N_EPOCHS
                        The number of epochs. (default: 20)
        -g GRAD_CLIP_LENGTH, --grad_clip_length GRAD_CLIP_LENGTH
                        The radius by which a gradient's length is
                        constrained. (default: 5.0)

    logging options:
        -v VERBOSE, --verbose VERBOSE
                        Print logs every v iterations. (default: 50)
        -r PREFIX, --prefix PREFIX
                        Initial symbol(s) of a SMILES string to generate.
                        (default: {)
        -m MAX_GEN_LENGTH, --max_gen_length MAX_GEN_LENGTH
                        Maximum number of tokens to generate. (default: 100)
        -p PREDICT_EPOCH, --predict_epoch PREDICT_EPOCH
                        Predict new strings every p iterations. (default: 50)
        -k N_PREDICTIONS, --n_predictions N_PREDICTIONS
                        The number of molecules to generate and save after
                        training. (default: 0)

    other options:
        -c {cpu,CPU,gpu,GPU}, --ctx {cpu,CPU,gpu,GPU}
                        CPU or GPU. (default: gpu)
        --help          Show this help message and exit.
        --version       Show version information.
    """
    # Process command line arguments.
    options = process_options()

    # Load raw SMILES data.
    dataset = SMILESDataset(filename=options.stage1_data)

    # Define data loader, which generates mini-batches for training.
    dataloader = SMILESDataLoader(
        batch_size=options.batch_size,
        n_steps=options.n_steps,
        dataset=dataset,
    )

    # Define model architecture.
    embedding_layer = OneHotEncoder(len(dataloader.vocab))
    rnn_layer = gluon.rnn.LSTM(
        hidden_size=options.hidden_size,
        num_layers=options.n_layers,
        dropout=0.2,
    )
    dense_layer = gluon.nn.Dense(len(dataloader.vocab))
    model = SMILESRNNModel(
        embedding_layer=embedding_layer,
        rnn_layer=rnn_layer,
        dense_layer=dense_layer,
    )

    # Define (hyper)parameters for model training.
    optimizer_params = {
        'learning_rate': options.learning_rate,
        'clip_gradient': options.grad_clip_length,
    }

    # Use CPU or GPU.
    ctx = {
        'cpu': context.cpu(0),
        'gpu': context.gpu(0),
    }

    # Begin fitting the model and generating novel molecules.
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
        model_params_in=options.model_params_in,
        model_params_out=options.model_params_out,
        n_predictions=options.n_predictions,
        predictions_out=options.predictions,
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
        prefix: str = Token.BOS.token,
        max_gen_length: int = 100,
        model_params_in: Union[IO, str] = None,
        model_params_out: Union[IO, str] = None,
        n_predictions: int = 0,
        predictions_out: Union[IO, str] = PREDICTIONS_OUT_DEF,
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
    model_params_in : file-like, default None
        Binary file with pre-trained model parameters.
    model_params_out : file-like, default None
        Binary file to save trained model parameters.
    n_predictions : int, default 0
        The number of molecules to generate and save after training.
    predictions_out : file-like, default PARENT_PATH/data/DATE__predictions.csv
        Text file to save generated predictions.
    """
    # Initialize model weights.
    if model_params_in is not None:
        if verbose > 0:
            print(f'Loading model weights from {model_params_in!r}.')
        model.load_parameters(model_params_in, ctx=ctx)
    else:
        model.initialize(
            ctx=ctx,
            force_reinit=True,
            init=init.Normal(sigma=0.1),
        )

    for key, value in model.collect_params().items():
        data = value.data()
        print(f'{key!r} weights norm: {np.linalg.norm(data).item():.3f}')
        print(f'{key!r} grad norm: {np.linalg.norm(data.grad).item():.3f}')

    # Define trainer.
    trainer = gluon.Trainer(model.collect_params(), opt, optimizer_params)

    # Define the list that stores the execution time per epoch.
    time_list = []

    for epoch in range(1, n_epochs + 1):

        # (Optional) Begin timer.
        start_time = None
        if verbose > 0:
            print(f'\nEpoch: {epoch:>3}\n')
            start_time = time.time()

        # Define a state list of the model.
        states = None

        # Define the list that stores loss per iteration.
        loss_list = []

        # Generate `Batch` instances for training.
        for batch_no, batch in enumerate(dataloader, start=1):
            curr_batch_size = batch.x.shape[0]

            # Every mini-batch entry is a substring of (padded) SMILES string.
            # If entries begin with beginning-of-SMILES token
            # `Token.BOS.token` (i.e. our model has not seen any part
            # of this mini-batch), then we initialize a new state list.
            # Otherwise, we keep the previous state list and detach it from
            # the computation graph.
            if batch.s:
                states = model.begin_state(batch_size=curr_batch_size, ctx=ctx)
            else:
                states = [state.detach() for state in states]

            inputs = batch.x.as_in_context(ctx)
            outputs = batch.y.T.reshape((-1,)).as_in_context(ctx)

            with autograd.record():
                # Run forward computation.
                p_outputs, states = model(inputs, states)

                # Get a label mask, which labels 1 for any valid token and 0
                # for padding token `Token.PAD.token`.
                label_mask = get_mask_for_loss(inputs.shape, batch.v_y)
                label_mask = label_mask.T.reshape((-1,)).as_in_context(ctx)

                # Compute loss using predictions, labels, and the label mask.
                loss = loss_fn(p_outputs, outputs, label_mask)

            loss.backward()
            trainer.step(batch_size=curr_batch_size)

            # (Optional) Print training statistics (batch).
            if (batch_no - 1) % verbose == 0:
                mean_loss = loss.mean().item()
                loss_list.append(mean_loss)
                print(
                    f'Batch: {batch_no:>6}, '
                    f'Loss: {mean_loss:>3.3f}'
                )

            # (Optional) Generate and print new molecules.
            if (batch_no - 1) % predict_epoch == 0:
                smiles = model.generate(
                    dataloader.vocab,
                    prefix=prefix,
                    max_length=max_gen_length,
                    ctx=ctx,
                )
                print(f'Molecule:\n    {smiles}')

        # (Optional) Print training statistics (epoch).
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

    # (Optional) Save model weights.
    if model_params_out is not None:
        if verbose > 0:
            print(f'\nSaving model parameters to {model_params_out!r}.')
        model.save_parameters(model_params_out)

    # (Optional) Print total execution time.
    if verbose > 0:
        total_time = datetime.timedelta(seconds=sum(time_list))
        print(f'\nTotal execution time: {total_time}.')

    # (Optional) After full training process, generate new molecules.
    if n_predictions > 0:
        if verbose > 0:
            print(
                f"\nGenerating novel molecules and saving results in "
                f"'{predictions_out}'."
            )

        with open(predictions_out, 'w') as fh:
            for _ in range(n_predictions):
                smiles = model.generate(
                    dataloader.vocab,
                    prefix=prefix,
                    max_length=max_gen_length,
                    ctx=ctx,
                )
                fh.write(f'{smiles}\n')

    if verbose > 0:
        print('\nDone!\n')


def process_options() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    namespace : argparse.Namespace
        Command line attributes and its values.
    """

    class PositiveInteger(int):
        """Requires integer to be positive.

        Raises
        ------
        ValueError
            If requirement is not satisfied.
        """
        def __new__(
                cls,
                value: int,
                *args,
                **kwargs
        ) -> int:
            try:
                value = int(value)
                if value <= 0:
                    raise ValueError
            except ValueError:
                raise

            return super().__new__(cls, value, *args, **kwargs)

    class ValidFileAction(argparse.Action):
        """Requires filename to be valid.
        """
        def __call__(
                self,
                parser_: argparse.ArgumentParser,
                namespace: argparse.Namespace,
                values: str,
                option_string: Optional[str] = None
        ):
            """Check if file exists.

            Raises
            ------
            OSError
                If file does not exist.
            """
            if not pathlib.Path(values).exists():
                raise OSError(f'no such file {values!r}.')

            setattr(namespace, self.dest, values)

    parser = argparse.ArgumentParser(
        description=(
            'Generate novel molecules with recurrent neural networks. '
            'The script has inclusive argument groups representing options '
            'for model fitting, molecule prediction, and logging.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False,
    )

    file_options = parser.add_argument_group('input/output arguments')
    file_options.add_argument(
        'stage1_data',
        help='The path to the training data containing SMILES strings.',
        action=ValidFileAction,
    )
    file_options.add_argument(
        '-L', '--model_params_in',
        help=(
            'The path to the model parameters. The model will load these '
            'parameters and proceed fitting.'
        ),
        action=ValidFileAction,
    )
    file_options.add_argument(
        '-S', '--model_params_out',
        help='Save learned model parameters in this file.',
        action=ValidFileAction,
    )
    file_options.add_argument(
        '-O', '--predictions',
        help='Save predicted molecules in this file.',
        default=PREDICTIONS_OUT_DEF,
    )

    model_options = parser.add_argument_group('model arguments')
    model_options.add_argument(
        '-u', '--hidden_size',
        help="The number of units in a network's hidden state.",
        type=PositiveInteger,
        default=1024,
    )
    model_options.add_argument(
        '-n', '--n_layers',
        help='The number of hidden layers.',
        type=PositiveInteger,
        default=2,
    )

    fit_options = parser.add_argument_group('hyperparameters')
    fit_options.add_argument(
        '-b', '--batch_size',
        help='The number of batches to generate at every iteration.',
        type=PositiveInteger,
        default=128,
    )
    fit_options.add_argument(
        '-s', '--n_steps',
        help='The number of time steps.',
        type=PositiveInteger,
        default=50,
    )
    fit_options.add_argument(
        '-l', '--learning_rate',
        help='The learning rate.',
        type=float,
        default=0.0001,
    )
    fit_options.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs.',
        type=PositiveInteger,
        default=20,
    )
    fit_options.add_argument(
        '-g', '--grad_clip_length',
        help="The radius by which a gradient's length is constrained.",
        type=float,
        default=5.0,
    )

    log_options = parser.add_argument_group('logging options')
    log_options.add_argument(
        '-v', '--verbose',
        help='Print logs every v iterations.',
        type=int,
        default=50,
    )
    log_options.add_argument(
        '-r', '--prefix',
        help='Initial symbol(s) of a SMILES string to generate.',
        default=Token.BOS.token,
    )
    log_options.add_argument(
        '-m', '--max_gen_length',
        help='Maximum number of tokens to generate.',
        type=PositiveInteger,
        default=100,
    )
    log_options.add_argument(
        '-p', '--predict_epoch',
        help='Predict new strings every p iterations.',
        type=PositiveInteger,
        default=50,
    )
    log_options.add_argument(
        '-k', '--n_predictions',
        help='The number of molecules to generate and save after training.',
        type=int,
        default=0,
    )

    other_options = parser.add_argument_group('other options')
    other_options.add_argument(
        '-c', '--ctx',
        help='CPU or GPU.',
        default='gpu',
        choices=('cpu', 'CPU', 'gpu', 'GPU'),
    )
    other_options.add_argument(
        '--help',
        help='Show this help message and exit.',
        action='help',
    )
    other_options.add_argument(
        '--version',
        help='Show version information.',
        action='version',
        version='%(prog)s beta',
    )

    return parser.parse_args()


if __name__ == '__main__':
    npx.set_np()
    main()
