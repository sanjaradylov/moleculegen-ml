#!/usr/bin/env python3

"""
Train a language model on a general set of SMILES strings of molecules.
Fine-tune the model on a focused set of compounds.
"""

__author__ = 'Sanjar Ad[iy]lov'


import argparse
import itertools
import pathlib
from typing import Optional

import mxnet as mx
from sklearn.manifold import TSNE

import moleculegen as mg


def main():
    """Main function:
    - load data for training;
    - load a model configuration and a vocabulary from a checkpoint;
    - train an encoder-decoder model on the training data and monitor the progress;
    - generate descriptors for training and test sets and save t-SNE projection;
    - load data for fine-tuning;
    - fine-tune the model on the fine-tuning data and monitor the progress.
    """
    options = process_options()

    # Data loaders for training.
    stage1_data = mg.data.SMILESDataset(filename=options.stage1_data)
    stage1_vocab = mg.data.SMILESVocabulary(
        load_from_pickle=f'{options.checkpoint}/vocabulary.pkl',
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

    model = mg.estimation.SMILESEncoderDecoder.from_config(
        f'{options.checkpoint}/config.json',
    )
    # A compound generator and RUAC metric.
    predictor = mg.generation.GumbelSoftmaxSearch(
        model=model,
        vocabulary=stage1_vocab,
        prefix=options.prefix,
        max_length=options.max_length,
        temperature=options.temperature,
    )
    train_dataset = [mg.Token.crop(smiles) for smiles in stage1_data]
    ruac = mg.evaluation.RAC(name='RUAC', count_unique=True)

    # Callbacks for training.
    epoch_metric_scorer = mg.callback.EpochMetricScorer(
        metrics=[
            mg.evaluation.Novelty(),
            mg.evaluation.Uniqueness(),
            mg.evaluation.Validity(),
        ],
        predictor=predictor,
        n_predictions=options.n_predictions,
        train_dataset=train_dataset,
    )
    progressbar = mg.callback.ProgressBar()
    early_stopping = mg.callback.EarlyStopping(
        min_delta=0.003,
        patience=3,
        restore_best_weights=True,
    )
    generator = mg.callback.Generator(
        filename=f'{options.checkpoint}/predictions_stage1',
        predictor=predictor,
        metric=ruac,
        on_interrupt=True,
        n_predictions=options.n_predictions*10,
        train_dataset=train_dataset,
    )
    plotter = mg.callback.PhysChemDescriptorPlotter(
        transformer=TSNE(n_components=2, n_jobs=-1),
        train_data=train_dataset[:10_000],
        image_file_prefix=f'{options.checkpoint}/descriptors_stage1',
        valid_data_file_prefix=f'{options.checkpoint}/predictions_stage1',
    )
    callbacks = [
        progressbar,
        epoch_metric_scorer,
        early_stopping,
        generator,
        plotter,
    ]

    # An optimizer, loss function, and learning rate scheduler.
    lr_scheduler = mx.lr_scheduler.CosineScheduler(
        max_update=options.n_epochs*len(stage1_batch_sampler),
        base_lr=options.learning_rate,
        final_lr=0.0005,
    )
    optimizer = mx.optimizer.Adam(
        learning_rate=options.learning_rate,
        clip_gradient=options.grad_clip_length,
        lr_scheduler=lr_scheduler,
    )
    loss_fn = mx.gluon.loss.SoftmaxCELoss()
    # Train the main model.
    # noinspection PyTypeChecker
    model.fit(
        batch_sampler=stage1_batch_sampler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=options.n_epochs,
        callbacks=callbacks,
    )

    # Data loaders for fine-tuning.
    stage2_data = mg.data.SMILESDataset(filename=options.stage2_data)
    stage2_corpus = stage1_vocab.get_token_id_corpus(stage2_data)
    stage2_sequence_sampler = mg.data.SMILESConsecutiveSampler(
        stage2_corpus,
        n_steps=options.n_steps,
        shuffle=True,
    )
    stage2_batch_sampler = mg.data.SMILESBatchSampler(
        stage2_sequence_sampler,
        batch_size=options.batch_size_fine_tune,
        last_batch='rollover',
    )

    # Callbacks for fine-tuning.
    train_dataset = [
        mg.Token.crop(smiles)
        for smiles in itertools.chain(stage1_data, stage2_data)
    ]
    epoch_metric_scorer = mg.callback.EpochMetricScorer(
        metrics=[
            mg.evaluation.Novelty(),
            mg.evaluation.Uniqueness(),
            mg.evaluation.Validity(),
            ruac,
        ],
        predictor=predictor,
        n_predictions=options.n_predictions,
        train_dataset=train_dataset,
    )
    batch_metric_scorer = mg.callback.BatchMetricScorer(
        metrics=[mg.evaluation.Perplexity()],
    )
    generator = mg.callback.Generator(
        filename=f'{options.checkpoint}/predictions_stage2',
        predictor=predictor,
        metric=ruac,
        on_interrupt=False,
        epoch=1,
        n_predictions=len(stage2_data),
        train_dataset=train_dataset,
    )
    plotter = mg.callback.PhysChemDescriptorPlotter(
        transformer=TSNE(n_components=2, n_jobs=-1),
        train_data=train_dataset[:10_000],
        image_file_prefix=f'{options.checkpoint}/descriptors_stage2',
        epoch=1,
        valid_data_file_prefix=f'{options.checkpoint}/predictions_stage2',
    )
    callbacks = [
        progressbar,
        batch_metric_scorer,
        epoch_metric_scorer,
        generator,
        plotter,
    ]

    # An optimizer, loss function, and learning rate scheduler.
    lr_scheduler = mx.lr_scheduler.CosineScheduler(
        max_update=options.n_epochs_fine_tune*len(stage2_batch_sampler),
        base_lr=options.learning_rate_fine_tune,
        final_lr=1e-5,
    )
    optimizer = mx.optimizer.Adam(
        learning_rate=options.learning_rate_fine_tune,
        clip_gradient=options.grad_clip_length,
        lr_scheduler=lr_scheduler,
    )
    # Fine-tune the model.
    fine_tuner = mg.estimation.SMILESEncoderDecoderFineTuner(
        model=model,
        output_dim=len(stage1_vocab),
        ctx=model.ctx,
    )
    # noinspection PyTypeChecker
    fine_tuner.fit(
        batch_sampler=stage2_batch_sampler,
        optimizer=optimizer,
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

        def __new__(cls, value: int, *args, **kwargs) -> int:
            value = int(value)
            if value <= 0:
                raise ValueError

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
        description='Train a language model on SMILES data.',
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
        default=0.0025,
    )
    fit_options.add_argument(
        '-e', '--n_epochs',
        help='The number of epochs (training).',
        type=PositiveInteger,
        default=6,
    )
    fit_options.add_argument(
        '-g', '--grad_clip_length',
        help="The radius by which a gradient's length is constrained.",
        type=float,
        default=8.0,
    )
    fit_options.add_argument(
        '-L', '--learning_rate_fine_tune',
        help='The learning rate (fine-tuning).',
        type=float,
        default=0.005,
    )
    fit_options.add_argument(
        '-B', '--batch_size_fine_tune',
        help='The number of batches to generate at every iteration (fine-tuning).',
        type=PositiveInteger,
        default=16,
    )
    fit_options.add_argument(
        '-E', '--n_epochs_fine_tune',
        help='The number of epochs (fine-tuning).',
        type=PositiveInteger,
        default=8,
    )

    generate_options = parser.add_argument_group('generation')
    generate_options.add_argument(
        '-n', '--n_predictions',
        help='The number of compounds to generate.',
        type=PositiveInteger,
        default=1000,
    )
    generate_options.add_argument(
        '-p', '--prefix',
        help='The prefix of a SMILES string to generate.',
        default=mg.Token.BOS,
    )
    generate_options.add_argument(
        '-m', '--max_length',
        help='The maximum number of tokens to generate.',
        type=PositiveInteger,
        default=80,
    )
    generate_options.add_argument(
        '-t', '--temperature',
        help='A sensitivity parameter',
        type=float,
        default=0.6,
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
    from rdkit.RDLogger import DisableLog

    DisableLog('rdApp.*')
    mx.npx.set_np()
    main()
