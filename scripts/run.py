#!/usr/bin/env python3

"""
Train and evaluate a SMILESRNN language model.

See `main()`, `HYPERPARAMETERS`, and `CONFIG` for model, sampler, optimizer, and callback
choices.
Run ```python3 run.py --help``` to print help message.
"""

__author__ = 'Sanjar Ad[iy]lov'


import argparse
import pathlib

import mxnet as mx
from sklearn.manifold import MDS

import moleculegen as mg


HYPERPARAMETERS = {
    'vocab': {
        'min_count': 20,
        'match_bracket_atoms': False,
        'allowed_tokens': frozenset(),
    },
    'sampler': {
        'time_span': 64,
        'max_offset': 2,
        'fraction': 0.8,
        'batch_size': 128,
    },
    'model': {
        'embedding_dim': 32,
        'embedding_dropout': 0.4,
        'embedding_dropout_axes': 1,
        'embedding_init': 'xavier_uniform',
        'rnn': 'lstm',
        'rnn_n_layers': 2,
        'rnn_n_units': 256,
        'rnn_dropout': 0.6,
        'rnn_i2h_init': 'xavier_uniform',
        'rnn_h2h_init': 'orthogonal_normal',
        'rnn_state_init': 'zeros',
        'rnn_reinit_state': True,
        'rnn_detach_state': True,
        'dense_n_layers': 1,
        'dense_init': 'xavier_uniform',
        'tie_weights': False,
        'ctx': mx.context.gpu(),
    },
    'predictor': {
        'prefix': mg.Token.BOS,
        'max_length': 101,
        'temperature': 1.0,
        'strategy': 0.8,
        'n_predictions': 1_000,
    },
    'early_stop': {
        'min_delta': 0.002,
        'patience': 3,
    },
    'train': {
        'n_epochs': 12,
        'start_learning_rate': 0.001,
        'final_learning_rate': 0.0001,
        'clip_gradient': 10.0,
    }
}

CONFIG = {
    'vocab_filename': 'vocabulary.pkl',
    'model_param_filename': 'weights.params',
    'predictor_file_prefix': 'predictions',
    'plotter_file_prefix': 'descriptors',
    'log_filename': None,
    'verbose': True,
}


def main():
    """Main function:
    - load data for training and evaluation;
    - tokenize training data and create batch sampler using `random sampling`;
    - train `SMILESRNN` model;
    - evaluate generated data using `GumbelSoftmaxSearch` and 7 metrics;
    - generate MDS plot of physicochemical descriptors;
    - save model parameters and training logs.
    """
    options = process_options()

    train_data = mg.data.SMILESDataset(options.train_data)
    reference_data = mg.data.SMILESDataset(options.reference_data, augment=False)
    vocabulary = mg.data.SMILESVocabulary(
        train_data,
        need_corpus=True,
        min_count=HYPERPARAMETERS['vocab']['min_count'],
        match_bracket_atoms=HYPERPARAMETERS['vocab']['match_bracket_atoms'],
        allowed_tokens=HYPERPARAMETERS['vocab']['allowed_tokens'],
    )
    vocabulary.to_pickle(f'{options.checkpoint}/{CONFIG["vocab_filename"]}')
    sequence_sampler = mg.data.SMILESRandomSampler(
        vocabulary.corpus,
        n_steps=HYPERPARAMETERS['sampler']['time_span'],
        max_offset=HYPERPARAMETERS['sampler']['max_offset'],
        samples_fraction=HYPERPARAMETERS['sampler']['fraction'],
    )
    batch_sampler = mg.data.SMILESBatchSampler(
        sequence_sampler,
        batch_size=HYPERPARAMETERS['sampler']['batch_size'],
        last_batch='discard',
    )

    model = mg.estimation.SMILESRNN(
        len(vocabulary),
        use_one_hot=False,
        embedding_dim=HYPERPARAMETERS['model']['embedding_dim'],
        embedding_dropout=HYPERPARAMETERS['model']['embedding_dropout'],
        embedding_dropout_axes=HYPERPARAMETERS['model']['embedding_dropout_axes'],
        embedding_init=HYPERPARAMETERS['model']['embedding_init'],
        rnn=HYPERPARAMETERS['model']['rnn'],
        rnn_n_layers=HYPERPARAMETERS['model']['rnn_n_layers'],
        rnn_n_units=HYPERPARAMETERS['model']['rnn_n_units'],
        rnn_dropout=HYPERPARAMETERS['model']['rnn_dropout'],
        rnn_i2h_init=HYPERPARAMETERS['model']['rnn_i2h_init'],
        rnn_h2h_init=HYPERPARAMETERS['model']['rnn_h2h_init'],
        rnn_state_init=HYPERPARAMETERS['model']['rnn_state_init'],
        rnn_reinit_state=HYPERPARAMETERS['model']['rnn_reinit_state'],
        rnn_detach_state=HYPERPARAMETERS['model']['rnn_detach_state'],
        dense_n_layers=HYPERPARAMETERS['model']['dense_n_layers'],
        dense_init=HYPERPARAMETERS['model']['dense_init'],
        tie_weights=HYPERPARAMETERS['model']['tie_weights'],
        ctx=HYPERPARAMETERS['model']['ctx'],
    )

    predictor = mg.generation.GumbelSoftmaxSearch(
        model=model,
        vocabulary=vocabulary,
        prefix=HYPERPARAMETERS['predictor']['prefix'],
        max_length=HYPERPARAMETERS['predictor']['max_length'],
        temperature=HYPERPARAMETERS['predictor']['temperature'],
        strategy=HYPERPARAMETERS['predictor']['strategy'],
    )

    batch_metric_scorer = mg.callback.BatchMetricScorer([
        mg.evaluation.Perplexity(ignore_label=vocabulary.PAD_ID),
    ])
    epoch_metric_scorer = mg.callback.EpochMetricScorer(
        metrics=[
            mg.evaluation.Validity(),
            mg.evaluation.Uniqueness(),
            mg.evaluation.InternalDiversity(),
            mg.evaluation.Novelty(),
            mg.evaluation.NearestNeighborSimilarity(),
            mg.evaluation.KLDivergence(),
        ],
        predictor=predictor,
        n_predictions=HYPERPARAMETERS['predictor']['n_predictions'],
        train_dataset=reference_data,
    )
    progressbar = mg.callback.ProgressBar()
    generator = mg.callback.Generator(
        filename=f'{options.checkpoint}/{CONFIG["predictor_file_prefix"]}',
        predictor=predictor,
        epoch=None,
        on_interrupt=True,
        n_predictions=HYPERPARAMETERS['predictor']['n_predictions'],
    )
    plotter = mg.callback.PhysChemDescriptorPlotter(
        transformer=MDS(n_components=2, metric=False),
        train_data=reference_data.take_(2*HYPERPARAMETERS['predictor']['n_predictions']),
        epoch=None,
        image_file_prefix=f'{options.checkpoint}/{CONFIG["plotter_file_prefix"]}',
        valid_data_file_prefix=f'{options.checkpoint}/{CONFIG["predictor_file_prefix"]}',
    )
    early_stopping = mg.callback.EarlyStopping(
        min_delta=HYPERPARAMETERS['early_stop']['min_delta'],
        patience=HYPERPARAMETERS['early_stop']['patience'],
        restore_best_weights=True,
    )
    callbacks = [
        progressbar,
        batch_metric_scorer,
        epoch_metric_scorer,
        early_stopping,
        generator,
        plotter,
    ]

    lr_scheduler = mx.lr_scheduler.CosineScheduler(
        max_update=HYPERPARAMETERS['train']['n_epochs'] * len(batch_sampler),
        base_lr=HYPERPARAMETERS['train']['start_learning_rate'],
        final_lr=HYPERPARAMETERS['train']['final_learning_rate'],
    )
    optimizer = mx.optimizer.Adam(
        learning_rate=HYPERPARAMETERS['train']['start_learning_rate'],
        clip_gradient=HYPERPARAMETERS['train']['clip_gradient'],
        lr_scheduler=lr_scheduler,
    )
    loss_fn = mx.gluon.loss.SoftmaxCELoss()

    model.fit(
        batch_sampler=batch_sampler,
        optimizer=optimizer,
        loss_fn=loss_fn,
        n_epochs=HYPERPARAMETERS['train']['n_epochs'],
        callbacks=callbacks,
        verbose=CONFIG['verbose'],
        log_filename=CONFIG['log_filename'],
    )

    model.save_parameters(f'{options.checkpoint}/{CONFIG["model_param_filename"]}')


def process_options() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
    namespace : argparse.Namespace
        Command line attributes and its values.
    """

    class ValidFileAction(argparse.Action):
        """Requires filename to be valid.
        """

        def __call__(
                self,
                parser_: argparse.ArgumentParser,
                namespace: argparse.Namespace,
                values: str,
                option_string: str = None,
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
            'Train and evaluate a SMILESRNN language model.\n\n'
            '    Command line arguments are only for IO operations.\n'
            '    Configuration and hyperparameters are hardcoded in the script since \n'
            '    there are more than 30 values to tune. See docs for more information.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False,
    )

    parser.add_argument(
        'train_data',
        help='The path to the training data containing SMILES strings.',
        action=ValidFileAction,
    )
    parser.add_argument(
        'reference_data',
        help='The path to the reference data for evaluation and comparison.',
        action=ValidFileAction,
    )
    parser.add_argument(
        'checkpoint',
        help='The path to the directory for saving training results.',
        action=ValidFileAction,
    )

    parser.add_argument(
        '--help',
        help='Show this help message and exit.',
        action='help',
    )
    parser.add_argument(
        '--version',
        help='Show version information.',
        action='version',
        version='moleculegen-1.1.0-benchmark',
    )

    return parser.parse_args()


if __name__ == '__main__':
    from rdkit.RDLogger import DisableLog

    DisableLog('rdApp.*')
    mx.npx.set_np()

    main()
