#!/usr/bin/env python3

"""
Learn language models on SMILES strings of given molecules. Generate and
evaluate new molecules.
"""

__author__ = 'Sanjar Ad[iy]lov'

import argparse
import pathlib
from typing import Optional

import mxnet as mx
from mxnet import gluon
from rdkit.RDLogger import DisableLog

import moleculegen as mg


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
                        (default: 256)
        -n N_LAYERS, --n_layers N_LAYERS
                        The number of hidden layers. (default: 2)

    hyperparameters:
        -b BATCH_SIZE, --batch_size BATCH_SIZE
                        The number of batches to generate at every iteration.
                        (default: 64)
        -s N_STEPS, --n_steps N_STEPS
                        The number of time steps. (default: 32)
        -l LEARNING_RATE, --learning_rate LEARNING_RATE
                        The learning rate. (default: 0.005)
        -e N_EPOCHS, --n_epochs N_EPOCHS
                        The number of epochs. (default: 20)
        -g GRAD_CLIP_LENGTH, --grad_clip_length GRAD_CLIP_LENGTH
                        The radius by which a gradient's length is
                        constrained. (default: 16.0)

    logging options:
        -v VERBOSE, --verbose VERBOSE
                        Print logs every v iterations. (default: 500)
        -r PREFIX, --prefix PREFIX
                        Initial symbol(s) of a SMILES string to generate.
                        (default: {)
        -m MAX_GEN_LENGTH, --max_gen_length MAX_GEN_LENGTH
                        Maximum number of tokens to generate. (default: 80)
        -p PREDICT_EPOCH, --predict_epoch PREDICT_EPOCH
                        Predict new strings every p iterations. (default: 500)
        -k N_PREDICTIONS, --n_predictions N_PREDICTIONS
                        The number of molecules to generate and save after
                        training. (default: 10000)

    other options:
        -c {cpu,CPU,gpu,GPU}, --ctx {cpu,CPU,gpu,GPU}
                        CPU or GPU. (default: gpu)
        --help          Show this help message and exit.
        --version       Show version information.
    """

    def has_prohibited_tokens(smiles: str) -> bool:
        """Check if `smiles` has any prohibited token declared in
        `PROHIBITED_TOKENS` constant.

        Parameters
        ----------
        smiles : str
            SMILES string.

        Returns
        -------
        check : bool

        Notes
        -----
        !!! Use tokenization on `smiles` if any prohibited token matches
        subtokens from a valid token set (e.g. 'Fe' is valid, but 'F' isn't).
        """
        for token in prohibited_tokens:
            if token in smiles:
                return True
        return False

    options = process_options()

    dataset = mg.data.SMILESDataset(filename=options.stage1_data)
    prohibited_tokens = frozenset(
        [
            'Sn', 'K', 'Al', 'Te', 'te', 'Li', 'As', 'Na', 'Se', 'se',
        ]
    )
    dataset = dataset.filter(lambda smiles: not has_prohibited_tokens(smiles))

    vocabulary = mg.data.SMILESVocabulary(dataset, need_corpus=True)

    sequence_sampler = mg.data.SMILESConsecutiveSampler(
        vocabulary,
        n_steps=options.n_steps,
        shuffle=True,
    )
    batch_sampler = mg.data.SMILESBatchSampler(
        sequence_sampler,
        batch_size=options.batch_size,
        last_batch='rollover',
    )

    ctx_map = {
        'cpu': mx.context.cpu(0),
        'gpu': mx.context.gpu(0),
    }
    ctx = ctx_map[options.ctx.lower()]

    model = mg.estimation.SMILESEncoderDecoder(
        len(vocabulary),
        embedding_dim=32,
        n_rnn_layers=options.n_layers,
        n_rnn_units=options.hidden_size,
        rnn_dropout=0.5,
        dense_dropout=0.25,
        ctx=ctx,
    )

    lr_scheduler = mx.lr_scheduler.FactorScheduler(
        factor=0.9,
        stop_factor_lr=1e-5,
        base_lr=options.learning_rate,
        step=len(batch_sampler),
    )
    optimizer = mx.optimizer.Adam(
        learning_rate=options.learning_rate,
        clip_gradient=options.grad_clip_length,
        lr_scheduler=lr_scheduler,
    )
    loss_fn = gluon.loss.SoftmaxCELoss()
    callbacks = [
        mg.callback.EpochMetricScorer(
            metrics=[
                mg.evaluation.RAC(name='RUAC', count_unique=True),
            ],
            predictor=mg.generation.GreedySearch(ctx=lambda array: array.as_in_ctx(ctx)),
            vocabulary=vocabulary,
            train_dataset=[mg.Token.crop(smiles) for smiles in dataset],
        ),
        mg.callback.ProgressBar(),
    ]

    model.fit(
        batch_sampler=batch_sampler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=options.n_epochs,
        callbacks=callbacks,
    )


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
        default='data/sets/predictions.csv',
    )

    model_options = parser.add_argument_group('model arguments')
    model_options.add_argument(
        '-u', '--hidden_size',
        help="The number of units in a network's hidden state.",
        type=PositiveInteger,
        default=256,
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
        default=64,
    )
    fit_options.add_argument(
        '-s', '--n_steps',
        help='The number of time steps.',
        type=PositiveInteger,
        default=64,
    )
    fit_options.add_argument(
        '-l', '--learning_rate',
        help='The learning rate.',
        type=float,
        default=0.005,
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
        default=8.0,
    )

    log_options = parser.add_argument_group('logging options')
    log_options.add_argument(
        '-v', '--verbose',
        help='Print logs every v iterations.',
        type=int,
        default=500,
    )
    log_options.add_argument(
        '-r', '--prefix',
        help='Initial symbol(s) of a SMILES string to generate.',
        default=mg.Token.BOS,
    )
    log_options.add_argument(
        '-m', '--max_gen_length',
        help='Maximum number of tokens to generate.',
        type=PositiveInteger,
        default=80,
    )
    log_options.add_argument(
        '-p', '--predict_epoch',
        help='Predict new strings every p iterations.',
        type=PositiveInteger,
        default=500,
    )
    log_options.add_argument(
        '-k', '--n_predictions',
        help='The number of molecules to generate and save after training.',
        type=int,
        default=10000,
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
    DisableLog('rdApp.*')
    mx.npx.set_np()
    main()
