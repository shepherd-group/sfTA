#!/usr/bin/env python
''' analyse_vasp.py [options] directory_1 directory_2 ... directory_N

A script to parse output files produced by Vienna Ab initio Simulation Package
(VASP) which are the result of intermediate steps leading up to the generation
of files used by the Coupled Cluster 4 Solds (CC4S) code.
'''

import os
import sys
import argparse
import numpy as np
import pandas as pd

from warnings import warn
from typing import TypeVar, List, Tuple, Type

Array = TypeVar('numpy.ndarray')
Dataframe = TypeVar('pd.core.frame.DataFrame')

# TODO - WZV
# Write docstring, comments throughout, and expand functionality!


class VASP:
    def __init__(self, clargs: str) -> None:
        self.__dict__.update(parse_command_line_arguments(clargs).__dict__)
        self.data = [Outcar(output, self.unique) for output in self.outputs]

        if self.output == 'csv':
            print(self.data[0].data.to_csv())
        else:
            print(self.data[0].data.to_string())


class Outcar:
    exclude_terms = [
            'E-fermi',
            'exchange ACFDT',
            'GAMMA',
            'Pullay stress',
        ]
    delta_terms = [
            'free energy TOTEN eV',
            'energy without entropy',
            'energy(sigma->0)',
        ]

    def __init__(self, filename: str, unique: bool) -> None:
        self.filename = self._isfile(filename)
        self.parseoutcar(self.filename, unique)

    def parseoutcar(self, filename: str, unique: bool) -> None:
        data = {'Iteration': []}
        n = len(data['Iteration'])
        final_data = {}

        with open(filename, 'rt') as stream:
            for line in stream:
                if 'Iteration' in line:
                    n += 1
                    data['Iteration'].append(n)
                elif n > 0 and all(t not in line for t in self.exclude_terms):
                    keys, values = self._parseline(line)

                    if 'free  energy   TOTEN  =' in line:
                        final_data[keys[0]] = values[0]
                    elif 'energy  without entropy=' in line:
                        final_data[keys[0]] = values[0]
                        final_data[keys[1]] = values[1]
                    elif '=' in line:
                        for k, v in zip(keys, values):
                            if k not in data:
                                data[k] = [v]
                            else:
                                data[k].append(v)

        for column in list(data.keys()):
            if column in final_data:
                data[column].append(final_data[column])
            else:
                if unique and n > 2:
                    unique_values = np.unique(data[column])
                    if unique_values.shape[0] <= 1:
                        del data[column]
                        continue

                if 'Iteration' in column:
                    data[column].append(data[column][-1])
                    data[column] = np.array(data[column]).astype(int)
                else:
                    data[column].append(np.nan)

        self.data = pd.DataFrame(data)

        for column in self.delta_terms:
            if column in self.data:
                values = self.data[column].values.flatten()
                deltas = np.concatenate([[np.nan], np.diff(values)], axis=0)
                self.data.drop(columns=column, inplace=True)
                self.data.insert(1, column, values, allow_duplicates=False)
                self.data.insert(2, fr'\Delta {column}',
                                 deltas, allow_duplicates=False)

    @staticmethod
    def _parseline(line: str) -> Tuple[List[str], List[float]]:
        keys = [[]]
        values = []

        for data in line.replace('=', ' = ').split():
            if '=' == data:
                keys[-1] = ' '.join(keys[-1])
                keys.append([])
            else:
                cdata = data.replace('-', '').replace('.', '')

                if len(cdata) == 0:
                    continue
                elif all(c.isdigit() for c in cdata):
                    values.append(float(data))
                else:
                    keys[-1].append(data)

        if len(keys) > 1:
            if len(keys[-1]) > 0:
                keys[-2] += ' ' + ' '.join(keys[-1])

        return keys[:-1], values

    @staticmethod
    def _isfile(filename: str) -> str:
        if not os.path.isfile(filename):
            raise RuntimeError(f'Could not find OUTCAR file: {filename}!')
        return filename


def parse_command_line_arguments(
            arguments: List[str],
        ) -> Type[argparse.ArgumentParser]:
    ''' Parse command-line arguments.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.

    Returns
    -------
    options : :class:`ArgumentParser`
        User options read in from the command-line.
    '''
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument(
            '-o',
            '--output',
            action='store',
            default='txt',
            type=str,
            dest='output',
            help='Provide one of "csv" or "txt" to change the stdout format.',
        )
    parser.add_argument(
            '-u',
            '--unique',
            action='store_true',
            default=True,
            dest='unique',
            help='Drop those columns which do not contain more than one '
                 'unique data point.',
        )
    parser.add_argument(
            'outputs',
            nargs='+',
            help='OUTCAR files to parse.',
        )
    parser.parse_args(args=None if arguments else ['--help'])

    options = parser.parse_args(arguments)

    return options


def main(arguments: List[str]) -> None:
    ''' Run structure factor twist averaging on cc4s outputs.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.
    '''
    VASP(arguments)


if __name__ == '__main__':
    warn('analyse_vasp.py is an incomplete script, '
         'use at your own discretion!', stacklevel=2)
    main(sys.argv[1:])
