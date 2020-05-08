#!/usr/bin/env python3

"""
Load, post-process, and save SMILES strings for Stage 1.
"""

import argparse
import datetime
import functools
import os
import sqlite3
from typing import Callable, Coroutine, Generator, TextIO, Union

from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.RDLogger import DisableLog


# Some useful constants to define file names.
DATE = datetime.datetime.now().strftime('%m_%d_%H_%M')
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(os.path.dirname(DIRECTORY), 'data')
OUT_FILENAME = os.path.join(DATA_DIRECTORY, f'stage1__{DATE}.csv')
SQL_FILENAME = os.path.join(DATA_DIRECTORY, 'stage1_smiles.sql')


def main():
    """The main function: load SMILES data from a database,
    process the data using filters (length restriction, canonical SMILES
    validation), and save resulting SMILES to an output file.
    """
    options = process_options()

    pipe = consume_valid_smiles(options.output_f)
    pipe = filter_valid_length(
        minimum=options.minimum_length,
        maximum=options.maximum_length,
        receiver=pipe,
    )
    pipe = filter_valid_structure(receiver=pipe)
    produce_query_results(
        query_filename=options.sql_f,
        db_filename=options.database_f,
        receiver=pipe,
    )
    pipe.close()


def produce_query_results(
        query_filename: Union[str, TextIO],
        db_filename: Union[str, TextIO] = os.environ.get('DB_FILENAME'),
        *,
        receiver: Coroutine,
):
    """Fetch Stage1 data from `db_filename` using query from
    `query_filename`.

    Parameters
    ----------
    receiver : callable
        Coroutine.
    query_filename : str or file-like
        Path to SQL file with query.
    db_filename : str, file-like, default $DB_FILENAME
        Path to database.

    Raises
    ------
    FileNotFoundError
        If specified file paths do not exist.
    """
    if db_filename is None or not os.path.exists(db_filename):
        raise FileNotFoundError('database does not exist.')
    if not os.path.exists(query_filename):
        raise FileNotFoundError('query file does not exist.')

    with sqlite3.connect(db_filename) as connection:
        cursor = connection.cursor()

        with open(query_filename) as query_fh:
            query = query_fh.read()

        cursor.execute(query)

        for smiles, in cursor.fetchall():
            receiver.send(smiles)


def coroutine(function: Callable) -> Callable:
    """Decorator that accepts a coroutine and explicitly calls `next` on it to
    commence execution.

    Parameters
    ----------
    function : callable
        Coroutine.

    Returns
    -------
    wrapped_function : callable
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs) -> Generator:
        generator = function(*args, **kwargs)
        next(generator)
        return generator

    return wrapper


@coroutine
def filter_valid_length(
        minimum: int = 15,
        maximum: int = 100,
        *,
        receiver: Coroutine,
) -> Coroutine:
    """Send only SMILES with length >= `minimum` and <= `maximum`.

    Parameters
    ----------
    receiver : coroutine
        Receiver coroutine (either filter or consumer).
    minimum : int, default 15
        Minimum length.
    maximum : int, default 100
        Maximum length.
    """
    try:
        while True:
            smiles = (yield)
            if minimum <= len(smiles) <= maximum:
                receiver.send(smiles)
    except GeneratorExit:
        receiver.close()


@coroutine
def filter_valid_structure(
        *,
        receiver: Coroutine,
) -> Coroutine:
    """Send only valid and canonical SMILES.

    Parameters
    ----------
    receiver : coroutine
        Receiver coroutine (either filter or consumer).
    """
    try:
        while True:
            smiles = (yield)
            molecule = MolFromSmiles(smiles)
            if molecule is not None:
                receiver.send(MolToSmiles(molecule))
    except GeneratorExit:
        receiver.close()


@coroutine
def consume_valid_smiles(out_filename: Union[TextIO, str]) -> Coroutine:
    """Save pipe results to `out_filename`.

    Parameters
    ----------
    out_filename : str or file-like
        File to save results.
    """
    with open(out_filename, 'w') as file_handler:
        try:
            while True:
                smiles = (yield)
                file_handler.write(f'{smiles}\n')
        except GeneratorExit:
            pass


def process_options() -> argparse.Namespace:
    """Process command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run SQL query from file and save results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    file_options = parser.add_argument_group('input/output arguments')
    file_options.add_argument(
        '-s', '--sql_f',
        help='Path to file w/ SQL query.',
        default=SQL_FILENAME,
    )
    file_options.add_argument(
        '-d', '--database_f',
        help='Path to database.',
        default=os.environ.get('DB_FILENAME'),
    )
    file_options.add_argument(
        '-o', '--output_f',
        help='File to save results.',
        default=OUT_FILENAME,
    )

    filter_options = parser.add_argument_group('filter arguments')
    filter_options.add_argument(
        '-a', '--minimum_length',
        help='Lower bound for SMILES string length.',
        type=int,
        default=15,
    )
    filter_options.add_argument(
        '-b', '--maximum_length',
        help='Upper bound for SMILES string length.',
        type=int,
        default=100,
    )

    return parser.parse_args()


if __name__ == '__main__':
    DisableLog('rdApp.*')
    main()
