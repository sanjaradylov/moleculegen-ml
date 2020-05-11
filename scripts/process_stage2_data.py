"""
Retrieve Stage 2 data w/ activity values.
"""

import argparse
import collections
import csv
import os
import sqlite3
from typing import Dict, List, TextIO, Tuple, Union


# Some useful constants to define file names.
DIRECTORY = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(os.path.dirname(DIRECTORY), 'data')
OUT_FILENAME = os.path.join(
    DATA_DIRECTORY, f'stage2_staphylococcus_aureus__unique.csv')
SQL_FILENAME = os.path.join(
    DATA_DIRECTORY, 'stage2_staphylococcus_aureus.sql')


def main():
    """Retrieve query results and save them to a csv file.
    """
    options = process_options()

    smiles_activity_map = get_sql_results(options.sql_f, options.database_f)

    with open(options.output_f, 'w', newline='') as fh:
        header_names = tuple(smiles_activity_map.keys())
        writer = csv.DictWriter(fh, fieldnames=header_names)
        writer.writeheader()
        writer.writerows((
            {
                header_names[0]: smiles,
                header_names[1]: activity,
            }
            for smiles, activity in zip(
                smiles_activity_map[header_names[0]],
                smiles_activity_map[header_names[1]],
            )
        ))


def get_sql_results(
        query_filename: Union[str, TextIO],
        db_filename: Union[str, TextIO] = os.environ.get('DB_FILENAME'),
) -> Dict[str, List]:
    """Fetch Stage2 data from `db_filename` using query from
    `query_filename` and return a set of unique entries.

    Parameters
    ----------
    query_filename : str or file-like
        Path to SQL file with query.
    db_filename : str, file-like, default $DB_FILENAME
        Path to database.

    Returns
    -------
    smiles_to_activity_map : dict
        SMILES string as a key and an activity as a value.

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

        smiles_activity_list: List[Tuple[str, int]] = cursor.fetchall()
        counter = collections.defaultdict(int)
        for smiles, activity in smiles_activity_list:
            counter[smiles] += 1
        smiles_activity_count_list = [
            (smiles, activity, counter[smiles])
            for smiles, activity in smiles_activity_list
        ]
        unique_smiles_activity_list = [
            (smiles, activity)
            for smiles, activity, count in smiles_activity_count_list
            if count == 1
        ]
        smiles_activity_map = {
            'smiles': [smiles for smiles, _ in unique_smiles_activity_list],
            'activity': [
                activity for _, activity in unique_smiles_activity_list],
        }

        return smiles_activity_map


def process_options() -> argparse.Namespace:
    """Process command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Run SQL query from file and save results.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        '-s', '--sql_f',
        help='Path to file w/ SQL query.',
        default=SQL_FILENAME,
    )
    parser.add_argument(
        '-d', '--database_f',
        help='Path to database.',
        default=os.environ.get('DB_FILENAME'),
    )
    parser.add_argument(
        '-o', '--output_f',
        help='File to save results.',
        default=OUT_FILENAME,
    )

    return parser.parse_args()


main()
