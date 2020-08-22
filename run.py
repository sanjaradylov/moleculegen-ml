#!/usr/bin/env python3

"""
Train a language model on a general set of SMILES strings of molecules.
Fine-tune the model on a focused set of compounds.
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
    """
    options = process_options()

    stage1_dataset = mg.data.SMILESDataset(filename=options.stage1_data)
    stage1_vocab = mg.data.SMILESVocabulary(
        load_from_pickle=f'{options.checkpoint}/stage1_vocabulary.pkl',
    )
    stage1_sequence_sampler = mg.data.SMILESConsecutiveSampler(
        stage1_vocab.corpus,
        n_steps=options.n_steps,
        shuffle=True,
    )
    stage1_batch_sampler = mg.data.SMILESBatchSampler(
        stage1_sequence_sampler,
        batch_size=options.batch_size,
        last_batch='rollover',
    )

    stage2_dataset = mg.data.SMILESDataset(filename=options.stage2_data)
    stage2_corpus = stage1_vocab.get_token_id_corpus(stage2_dataset)
    stage2_sequence_sampler = mg.data.SMILESConsecutiveSampler(
        stage2_corpus,
        n_steps=options.n_steps,
        shuffle=True,
    )
    stage2_batch_sampler = mg.data.SMILESBatchSampler(
        stage2_sequence_sampler,
        batch_size=options.batch_size,
        last_batch='rollover',
    )

    model = mg.estimation.SMILESEncoderDecoder.from_config(
        f'{options.checkpoint}/config.json',
    )
    lr_scheduler = mx.lr_scheduler.FactorScheduler(
        factor=0.8,
        stop_factor_lr=1e-5,
        base_lr=options.learning_rate,
        step=len(stage1_batch_sampler),
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
            predictor=mg.generation.GreedySearch(ctx=lambda a: a.as_in_ctx(model.ctx)),
            vocabulary=stage1_vocab,
            train_dataset=[mg.Token.crop(smiles) for smiles in stage1_dataset],
        ),
        mg.callback.ProgressBar(),
    ]
    # noinspection PyTypeChecker
    model.fit(
        batch_sampler=stage1_batch_sampler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=options.n_epochs,
        # callbacks=callbacks,
    )

    fine_tuner = mg.estimation.SMILESEncoderDecoderFineTuner(
        model=model,
        output_dim=len(stage1_vocab),
        dense_dropout=0.25,
        ctx=model.ctx,
    )
    lr_scheduler_fine_tune = mx.lr_scheduler.FactorScheduler(
        factor=0.7,
        stop_factor_lr=1e-6,
        base_lr=options.learning_rate_fine_tune,
        step=len(stage2_batch_sampler),
    )
    optimizer_fine_tune = mx.optimizer.Adam(
        learning_rate=options.learning_rate_fine_tune,
        clip_gradient=options.grad_clip_length,
        lr_scheduler=lr_scheduler_fine_tune,
    )
    # noinspection PyTypeChecker
    fine_tuner.fit(
        batch_sampler=stage2_batch_sampler,
        optimizer=optimizer_fine_tune,
        loss_fn=loss_fn,
        n_epochs=options.n_epochs_fine_tune,
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
        description='Train and fine-tune a language model on SMILES data.',
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
        'stage2_data',
        help='The path to the data for fine-tuning containing SMILES strings.',
        action=ValidFileAction,
    )
    file_options.add_argument(
        'checkpoint',
        help='The path to the directory of a vocabulary and model configuration.',
        action=ValidFileAction,
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
        help='The learning rate (training).',
        type=float,
        default=0.005,
    )
    fit_options.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs (training).',
        type=PositiveInteger,
        default=10,
    )
    fit_options.add_argument(
        '-L', '--learning_rate_fine_tune',
        help='The learning rate (fine-tuning).',
        type=float,
        default=0.01,
    )
    fit_options.add_argument(
        '-E', '--n_epochs_fine_tune',
        help='The number of epochs (fine-tuning).',
        type=PositiveInteger,
        default=30,
    )
    fit_options.add_argument(
        '-g', '--grad_clip_length',
        help="The radius by which a gradient's length is constrained.",
        type=float,
        default=8.0,
    )

    other_options = parser.add_argument_group('other options')
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
