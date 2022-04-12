#!/usr/bin/env python
''' analyse_cc4s.py [options] directory_1 directory_2 ... directory_N

Perform sFTA on a list of user provided directories which
contain cc4s structure factor outputs `GridVectors.elements`,
`CoulombPotential.elements`, and `SF.elements`.

The final report provides the directory for the structure factor
data which minimizes the residual of the difference between the
average structure factor and the given individual structure factor.

More details can be found in: https://doi.org/10.1038/s43588-021-00165-1
'''

import os
import sys
import time
import yaml
import typing
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

nparray = typing.TypeVar('np.ndarray')
pddataframe = typing.TypeVar('pd.core.frame.DataFrame')


class StructureFactor:
    ''' A simple class to store all the structure factor data in
    to prevent repeated recalculation and make analysis clean.

    Attributes
    ----------
    initial_time : float
        The seconds from epoch we started our analysis.
    previous_time : float
        The seconds from epoch we started a new timer instance.
    timing_report : dictionary
        Stores the integer count, corresponding note and total time in minutes
        required to perform each step of analysis.
    options : list of str and bool
        Stores the command line arguments parsed using argparse.
    directories : list of str
        Stores the paths with structure factor data
    mp2_df : :class:`pandas.DataFrame`
        Stores energies scrubbed from cc4s yaml output files.
    SFi : list of :class:`pandas.DataFrame`
        Stores the individual structure factors prior to averaging in the
        G vectors within a twist angle, sorted by G.
    aSFi : list of :class:`pandas.DataFrame`
        Stores the individual structure factors averaged within a given
        G vector, sorted by G.
    aSF : :class:`pandas.DataFrame`
        Stores the twist averaged structure factor across all twists
        and within a given G vector, sorted by G.
    ispecial : int
        The index corresponding to the special twist angle in the various
        lists.

    Methods
    -------
    update_timing_report(msg)
        Adds a timing check point to the timing report dictionary.
    end_timing_report()
        Close out the timing report and print out to the user.
    '''
    def __init__(self, clargs: typing.List[str], fmt: str = '%24.16f') -> None:
        ''' Run the general structure factor analysis based on the user
        provided command line arguments.

        Parameters
        ----------
        clargs : list of str
            The user provided command line arguments.
        fmt : str, default="%24.16f"
            Changes the float format for the various reports done as to string.
        '''
        self.initial_time = time.perf_counter()
        self.previous_time = self.initial_time
        self.timing_report = {'step ': [], 'note ': [], 'time (min) ': []}

        self.options = parse_command_line_arguments(clargs)
        directories = self.options.directories
        self.directories = clean_paths_and_simple_checks(directories)
        self.update_timing_report(msg='Input parsing and directory checks')

        if self.options.mp2 or self.options.mp2_write is not None:
            yaml_out_files = find_yaml_outs(self.directories)
            self.mp2_df = extract_mp2_from_yaml(yaml_out_files)
            self.update_timing_report(msg='Yaml checks and parsing')

            if self.options.mp2:
                mp2_str = self.mp2_df.to_string(index=False, float_format=fmt)
                print(mp2_str, file=sys.stderr)
                self.update_timing_report(msg='MP2 energy dumping')

            if self.options.mp2_write is not None:
                msg = f' Saving MP2 energies to: {self.options.mp2_write}'
                print(msg, file=sys.stderr)
                self.mp2_df.to_csv(self.options.mp2_write, index=False)
                self.update_timing_report(msg='MP2 energy storing')

        if self.options.skip_sfta:
            return

        Gvector_files, Coulomb_files, S_G_files = find_SF_outputs(directories)
        self.update_timing_report(msg='Structure factor output search')

        sf_tuple = read_and_average_SF(Gvector_files, Coulomb_files, S_G_files)
        self.update_timing_report(msg='Structure factor parsing and analysis')

        self.SFi, self.aSFi, self.aSF = sf_tuple

        self.ispecial = find_special_twist_angle(self.aSFi, self.aSF)
        self.update_timing_report(msg='Special twist analysis')

        if self.options.legacy_write is not None:
            legacy_output = self.options.legacy_write
            write_sfTA_csv(legacy_output, self.directories, self.SFi)
            self.update_timing_report(msg='Raw structure factor storing')

        if self.options.average:
            SF_str = self.aSF.to_string(index=False, float_format=fmt)
            print(SF_str, file=sys.stderr)
            self.update_timing_report(msg='Twist average dumping')

        if self.options.average_write is not None:
            msg = ' Saving average structure factor to:'
            print(f'{msg} {self.options.average_write}', file=sys.stderr)
            self.aSF.to_csv(self.options.average_write, index=False)
            self.update_timing_report(msg='Twist average storing')

        if self.options.single_write is not None:
            single_output = self.options.single_write
            write_individual_twist_average_csv(single_output, self.aSFi)
            self.update_timing_report(msg='Individual twist average storing')

        plot_names = [
                self.options.sfta_plot, self.options.difference_plot,
                self.options.variance_plot,
            ]
        if any(k is not None for k in plot_names):
            plot_SF(self.options.sfta_plot, self.options.difference_plot,
                    self.options.variance_plot, self.aSFi,
                    self.aSF, self.ispecial)
            self.update_timing_report(msg='Structure factor plotting')

        if self.options.special_write is not None:
            special_output = self.options.special_write
            msg = ' Saving special twist angle structure factor to:'
            print(f'{msg} {special_output}', file=sys.stderr)
            self.aSFi[self.ispecial].to_csv(special_output, index=False)
            self.update_timing_report(msg='Special twist storing')

        self.end_timing_report()

        print('\n Found Special Twist Angle:', file=sys.stderr)
        print(f' {self.directories[self.ispecial]}\n', file=sys.stderr)

    def update_timing_report(self, msg: str) -> None:
        ''' Perform the calculation of the elapsed time for a given
        process and report the index, time and message for the process
        during sfTA analysis.

        Parameters
        ----------
        msg : str
            The message corresponding to the timed process.
        '''
        current_time = time.perf_counter()
        dt = (current_time - self.previous_time)/60
        self.previous_time = current_time
        nstep = len(self.timing_report['step '])+1
        self.timing_report['step '].append(f'{nstep}')
        self.timing_report['note '].append(msg)
        self.timing_report['time (min) '].append(f'{dt:>10.6f}')

    def end_timing_report(self) -> None:
        ''' Close out the timing report by updating with the total
        time elapsed for all processing, then report to the user.
        '''
        self.previous_time = self.initial_time
        self.update_timing_report(msg='All analysis total time')
        timing_report = pd.DataFrame(self.timing_report)
        timing_report['% time '] = timing_report['time (min) '].astype(float)
        timing_report['% time '] /= timing_report['% time '].iloc[-1]/100
        report = timing_report.to_string(index=False, float_format='%.2f')
        print('\n Final script timing report: \n', report, file=sys.stderr)


def parse_command_line_arguments(
            arguments: typing.List[str],
        ) -> typing.Type[argparse.ArgumentParser]:
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
    parser.add_argument('-aw', '--average-write', action='store', default=None,
                        type=str, dest='average_write', help='Provide a file '
                        'to store the average structure factor data in. '
                        'The data format is csv, and the file flag is '
                        'automatically included.')
    parser.add_argument('-sw', '--special-write', action='store', default=None,
                        type=str, dest='special_write', help='Provide a file '
                        'to store the special twist angles structure factor '
                        'data to. The data format is csv.')
    parser.add_argument('-iw', '--individual-write', action='store',
                        default=None, type=str, dest='single_write',
                        help='A file name to write the average individual '
                        'structure factors, i.e. the S(G), V(G) averaged '
                        'over the common G within a single calculation.')
    parser.add_argument('-ew', '--mp2-write', action='store', default=None,
                        type=str, dest='mp2_write', help='A file to write the '
                        'MP2 energies to from the individual calculations.')
    parser.add_argument('-lw', '--legacy-write', action='store', default=None,
                        type=str, dest='legacy_write', help='Provide a file '
                        'name to store the individual structure factor data '
                        'to in a format ammendable to the sfTA.py script.')
    parser.add_argument('-sp', '--sfta-plot', action='store', default=None,
                        type=str, dest='sfta_plot', help='Provide a file '
                        'name to store the plot of the individual structure '
                        'factor data and the average structure factor. '
                        'The format is png, and the extenstion is not '
                        'required in the provided name.')
    parser.add_argument('-dp', '--difference-plot', action='store',
                        default=None, type=str, dest='difference_plot',
                        help='Provide a filename for the plot showing the '
                        'difference between the individual structure factors '
                        'and the twist averaged structure factors.')
    parser.add_argument('-vp', '--variance-plot', action='store', default=None,
                        type=str, dest='variance_plot', help='Provide a file '
                        'name to store the plot of the variance in the '
                        'average structure factor for the G values.')
    parser.add_argument('-a', '--average', action='store_true', default=False,
                        dest='average', help='Print out the average structure '
                        'factor in a nice table to the standard error output.')
    parser.add_argument('-e', '--mp2', action='store_true', default=False,
                        dest='mp2', help='Pull the MP2 energies and print out '
                        'as a table to the standard error output.')
    parser.add_argument('-s', '--skip-sfta', action='store_true',
                        default=False, dest='skip_sfta', help='Skip all forms '
                        'of sfTA analysis. I.E., overrides related settings!')
    parser.add_argument('directories', nargs='+', help='Paths containing '
                        'Structure Factor data to be analyzed.')
    parser.parse_args(args=None if arguments else ['--help'])

    options = parser.parse_args(arguments)

    def __ext_check(filename: str, ext: str) -> str:
        ''' A private function to ensure user provided
        filenames have the corresponding extension.
        '''
        if filename is not None and ext not in filename:
            filename += ext
        return filename

    options.average_write = __ext_check(options.average_write, '.csv')
    options.special_write = __ext_check(options.special_write, '.csv')
    options.single_write = __ext_check(options.single_write, '.csv')
    options.mp2_write = __ext_check(options.mp2_write, '.csv')
    options.legacy_write = __ext_check(options.legacy_write, '.csv')
    options.sfta_plot = __ext_check(options.sfta_plot, '.png')
    options.difference_plot = __ext_check(options.difference_plot, '.png')
    options.variance_plot = __ext_check(options.variance_plot, '.png')

    return options


def plot_SF(sfta_plot: str, difference_plot: str, variance_plot: str,
            raw_aSF: typing.List[pddataframe],
            SF: pddataframe, ispecial: int) -> None:
    ''' Performs all the plotting that can occur based on user input.
    This could be either the individual structure factors, the average
    structure factor and the special twist. Or the variance of the average
    structure factor. Both can also occur at the same time.

    Parameters
    ----------
    sfta_plot : str
        A string for the structure factor plot name, default is None in
        which case no plot is created.
    difference_plot : str
        A string for the plot name of the difference between the
        twist averaged structure factor and the individual structure factors.
    variance_plot : str
        A string for the plot of the variance for the average structure
        factor. Default is None in which case no plot is created.
    raw_aSF : list of :class:`pandas.DataFrame`
        A list of all the average structure factors for a given twist angle.
    SF : :class:`pandas.DataFrame`
        A data frame of the average structure factor.
    ispecial : integer
        The index of the special twist angle. The index is pythonic and
        matches the various lists used throughout.
    '''
    font = {'family': 'serif', 'sans-serif': 'Computer Modern Roman'}
    mpl.rc('font', **font)
    mpl.rc('savefig', dpi=300)
    mpl.rc('lines', lw=2, markersize=5)
    mpl.rc('legend', fontsize=8, numpoints=1)
    mpl.rc(('axes', 'xtick', 'ytick'), labelsize=8)
    mpl.rc('figure', dpi=300, figsize=(3.37, 3.37*(np.sqrt(5)-1)/2))

    if sfta_plot is not None:
        plt.clf()

        for i, aSFi in enumerate(raw_aSF):
            plt.errorbar(
                    aSFi['G'],
                    aSFi['S_G'],
                    aSFi['S_G_error'],
                    label='individual twists' if i == 1 else '',
                    color='#02a642',
                )

            if i == ispecial:
                plt.errorbar(
                        aSFi['G'],
                        aSFi['S_G'],
                        aSFi['S_G_error'],
                        label='special twist',
                        color='#f26003',
                        marker='o',
                        ls='--',
                        markeredgecolor='k',
                        markeredgewidth=1.0,
                        zorder=15,
                    )

        plt.errorbar(
                SF['G'],
                SF['S_G'],
                SF['S_G_error'],
                label='twist averaged',
                color='#2c43fc',
                zorder=10,
            )

        plt.xlabel('G')
        plt.ylabel('S(G)')
        plt.legend(loc='best', ncol=1, handlelength=1.0, handletextpad=0.1)
        print(' Saving structure factor plot to: '
              f'{sfta_plot}', file=sys.stderr)
        plt.savefig(sfta_plot, bbox_inches='tight')

    if difference_plot is not None:
        plt.clf()
        SF = SF.sort_values('G')

        for i, aSFi in enumerate(raw_aSF):

            plt.errorbar(
                    aSFi['G'],
                    aSFi['S_G'] - SF['S_G'],
                    np.sqrt(aSFi['S_G_error']**2 + SF['S_G_error']**2),
                    label='individual twists' if i == 1 else '',
                    color='#02a642',
                )

            if i == ispecial:
                plt.errorbar(
                        aSFi['G'],
                        aSFi['S_G'] - SF['S_G'],
                        np.sqrt(aSFi['S_G_error']**2 + SF['S_G_error']**2),
                        label='special twist',
                        color='#f26003',
                        marker='o',
                        ls='--',
                        markeredgecolor='k',
                        markeredgewidth=1.0,
                        zorder=15,
                    )

        plt.axhline(
                y=0,
                label='twist averaged',
                color='#2c43fc',
                zorder=10,
            )

        plt.xlabel('G')
        plt.ylabel(r'$\Delta$S(G)')
        plt.legend(loc='best', ncol=1, handlelength=1.0, handletextpad=0.1)
        print(' Saving structure difference factor plot to: '
              f'{difference_plot}', file=sys.stderr)
        plt.savefig(difference_plot, bbox_inches='tight')

    if variance_plot is not None:
        plt.clf()

        ntwists = len(raw_aSF)
        sem_error = SF['S_G_error']/((2*(ntwists - 1))**0.5)
        variance = (SF['S_G_error'] * (ntwists)**0.5)**2.0
        variance_error = (sem_error * 2.0 * variance) / SF['S_G_error']

        plt.plot(
                SF['G'],
                variance,
                color='#2c43fc',
                zorder=5,
                lw=1.0,
            )

        plt.fill_between(
                SF['G'],
                variance - variance_error,
                variance + variance_error,
                label='twist averaged',
                color='#2c43fc',
                zorder=10,
                alpha=0.5,
            )

        plt.yscale('log')
        plt.xlabel('G')
        plt.ylabel('S(G) Variance')
        plt.legend(loc='best', ncol=1)
        print(' Saving structure factor variance plot to: '
              f'{variance_plot}', file=sys.stderr)
        plt.savefig(variance_plot, bbox_inches='tight')


def clean_paths_and_simple_checks(
            directories: typing.List[str],
        ) -> typing.List[str]:
    ''' Perform some simple pre checks on the user provided directories.

    Parameters
    ----------
    directories : list of strings
        The directories where structure factor data is contained.

    Returns
    -------
    cleaned_directories : list of strings
        The directories list with files removed from the list, leaving
        only directories.

    Raises
    ------
    UserWarning
        When there are files within the directories list.
    UserWarning
        When there are not 100 paths.
    RuntimeError
        When a directory is found to occur multiple times in directories.
    '''
    cleaned_directories = []
    removed_paths = '\n'
    for path in directories:
        if os.path.isdir(path):
            cleaned_directories.append(path)
        else:
            removed_paths += path + '\n'

    if len(cleaned_directories) != len(directories):
        warnings.warn(f'\nNon-directory paths provided: {removed_paths}'
                      'These are removed!\n', stacklevel=2)

    if len(cleaned_directories) != 100:
        warnings.warn(f'\nThere are {len(cleaned_directories)},'
                      ' not 100 calculations!\n', stacklevel=2)

    if np.unique(directories).shape[0] != len(directories):
        raise RuntimeError('Repeated directories found!')

    return cleaned_directories


def find_yaml_outs(directories: typing.List[str]) -> typing.List[str]:
    ''' Search through the user provided directories and find the
    relevant yaml energy out files containing energy data.

    Parameters
    ----------
    directories : list of strings
        The directories where structure factor data is contained.

    Returns
    -------
    yaml_out_files : list of strings
        A list of the yaml log files with energy data.

    Raises
    ------
    RuntimeError
        When a cc4s yaml output file is absent from a directory.
    '''
    yaml_out_files = []

    for path in directories:
        yaml_out_file = f'{path}/cc4s.out.yaml'

        if not os.path.isfile(yaml_out_file):
            raise RuntimeError(f'{yaml_out_file} does not exist!')
        else:
            yaml_out_files.append(yaml_out_file)

    return yaml_out_files


def get_yaml_as_dict(yaml_file: str) -> dict:
    ''' Read in the yaml file `cc4s.out.yaml` from a cc4s calculation
    and return the information as a dictionary.

    Parameters
    ----------
    yaml_file : str
        A string of a yaml file to read in.

    Returns
    -------
    yaml_dict : :class:`dictionary`
        A dictionary of the cc4s data.
    '''
    with open(yaml_file, 'r') as yaml_stream:
        yaml_dict = yaml.safe_load(yaml_stream)

    return yaml_dict


def extract_mp2_from_yaml(yaml_out_files: typing.List[str]) -> pddataframe:
    ''' Collect the data yaml out files and return the relevent
    energy data from these files.

    Parameters
    ----------
    yaml_out_files : list of strings
        A list of the yaml log files with energy data.

    Returns
    -------
    mp2_df : :class:`pandas.DataFrame`
        A pandas Data Frame of all the MP2 energies from the twist angles

    Raises
    ------
    RuntimeError
        When a cc4s yaml logfile is absent from a directory.
    '''
    def __ekey_check(cstep: str, nkey: str, ekey: str) -> bool:
        ''' A private function to check the yaml for an energy key '''
        iskey = False

        if cstep['name'] == nkey:
            if 'out' in cstep.keys() and 'energy' in cstep['out'].keys():
                iskey = ekey in cstep['out']['energy'].keys()

        return iskey

    mp2_df = {'Twist': []}

    for imp2, yaml_file in enumerate(yaml_out_files):
        steps = get_yaml_as_dict(yaml_file)['steps'].values()

        mp2_df['Twist'].append(imp2+1)

        for step in steps:
            if __ekey_check(step, 'CoupledCluster', 'correlation'):
                if imp2 == 0:
                    mp2_df['Ec'], mp2_df['Ed'], mp2_df['Ex'] = [], [], []

                mp2_df['Ec'].append(step['out']['energy']['correlation'])
                mp2_df['Ed'].append(step['out']['energy']['direct'])
                mp2_df['Ex'].append(step['out']['energy']['exchange'])

            if __ekey_check(step, 'FiniteSizeCorrection', 'correction'):
                if imp2 == 0:
                    mp2_df['FSC'] = []

                mp2_df['FSC'].append(step['out']['energy']['correction'])

            if __ekey_check(step, 'BasisSetCorrection', 'correction'):
                if imp2 == 0:
                    mp2_df['BSC'] = []

                mp2_df['BSC'].append(step['out']['energy']['correction'])

    if any(len(v) != len(yaml_out_files) for k, v in mp2_df.items()):
        raise RuntimeError('cc4s yaml log files are missing data!')

    mp2_df = pd.DataFrame(mp2_df)

    if all(col in mp2_df.columns for col in ['Ec', 'FSC', 'BSC']):
        mp2_df['Ec+FSC+BSC'] = mp2_df['Ec'] + mp2_df['FSC'] + mp2_df['BSC']

    return mp2_df


def find_SF_outputs(
            directories: typing.List[str],
        ) -> typing.Tuple[typing.List[str]]:
    ''' Search through the user provided directories and find the
    relevenant outputs for sFTA.

    Parameters
    ----------
    directories : list of strings
        The directories where structure factor data is contained.

    Returns
    -------
    Gvector_files : list of strings
        A list of files with the G vector data.
    Coulomb_files : list of strings
        A list of files with the Coulomb potential data.
    S_G_files : list of strings
        A list of files with the S_G data.

    Raises
    ------
    RuntimeError
        When one of the GridVectors.elements, CoulombPotential.elements or
        SF.elements is absent from a directory.
    '''
    Gvector_files, Coulomb_files, S_G_files = [], [], []

    for path in directories:
        Gvector_file = f'{path}/GridVectors.elements'
        Coulomb_file = f'{path}/CoulombPotential.elements'
        S_G_file = f'{path}/SF.elements'

        if not os.path.isfile(Gvector_file):
            raise RuntimeError(f'{Gvector_file} does not exist!')
        elif not os.path.isfile(Coulomb_file):
            raise RuntimeError(f'{Coulomb_file} does not exist!')
        elif not os.path.isfile(S_G_file):
            raise RuntimeError(f'{S_G_file} does not exist!')
        else:
            Gvector_files.append(Gvector_file)
            Coulomb_files.append(Coulomb_file)
            S_G_files.append(S_G_file)

    return Gvector_files, Coulomb_files, S_G_files


def read_and_generate_Gvector_magnitudes(Gvector_file: str) -> nparray:
    ''' Read in a GridVectors.elements file generated by cc4s and calculate the
    G magnitudes for sFTA.

    Parameters
    ----------
    Gvector_file : string
        A file which contains the G vectors.

    Returns
    -------
    G : :class:`numpy.ndarray`
        An array of the G magnitudes.
    '''
    raw_g_xyz = np.loadtxt(Gvector_file, dtype=np.float64)
    N_G = int(raw_g_xyz.shape[0] / 3)
    g_xyz = raw_g_xyz.reshape((N_G, 3))
    G = np.sqrt(np.einsum('ij,ij->i', g_xyz, g_xyz))

    return G


def read_Vg(Coulomb_files: str) -> nparray:
    ''' Read in a CoulombPotenial.elements file generated by cc4s.

    Parameters
    ----------
    Coulomb_files : string
        A file which contains the Coulomb energy elements.

    Returns
    -------
    V_G : :class:`numpy.ndarray`
        An array of the Coulomb potential values for the G magnitudes.
    '''
    V_G = np.loadtxt(Coulomb_files, dtype=np.float64)

    return V_G


def read_Sg(S_G_file: str) -> nparray:
    ''' Read in a SF.elements file generated by cc4s.

    Parameters
    ----------
    S_G_file : string
        A file which contains the Structure Factor elements.

    Returns
    -------
    S_G : :class:`numpy.ndarray`
        An array of the Structure Factor values for the G magnitudes.
    '''
    S_G = np.loadtxt(S_G_file, dtype=np.float64)

    return S_G


def read_and_average_SF(
            Gvector_files: typing.List[str],
            Coulomb_files: typing.List[str],
            S_G_files: typing.List[str]
        ) -> typing.Tuple[typing.List[pddataframe], pddataframe]:
    ''' Loop through the relevant files of structure factors and calculate
    the average structure factor.

    Parameters
    ----------
    Gvector_files : list of strings
        A list of files with the G vector data.
    Coulomb_files : list of strings
        A list of files with the Coulomb energy data.
    S_G_files : list of strings
        A list of files with the S_G data.

    Returns
    -------
    raw_SF : list of :class:`pandas.DataFrame`
        A list of all the structure factors.
    raw_aSF : list of :class:`pandas.DataFrame`
        A list of all the average structure factors for a given twist angle.
    SF : :class:`pandas.DataFrame`
        A data frame of the average structure factor.
    '''
    raw_SF = []
    raw_aSF = []
    SF = pd.DataFrame()

    for files in zip(Gvector_files, Coulomb_files, S_G_files):
        aSFi = pd.DataFrame()

        G = read_and_generate_Gvector_magnitudes(files[0]).round(10)
        V_G = read_Vg(files[1])
        S_G = read_Sg(files[2])
        SV_G = S_G*V_G
        SFi = pd.DataFrame({'G': G, 'V_G': V_G, 'S_G': S_G, 'S_G*V_G': SV_G})
        raw_SF.append(SFi)

        group = SFi.groupby('G')
        aSFi['S_G'] = group['S_G'].mean()
        aSFi['S_G_error'] = group['S_G'].sem()
        aSFi['S_G*V_G'] = group['S_G*V_G'].sum()
        aSFi['V_G'] = group['V_G'].sum()
        aSFi.reset_index(drop=False, inplace=True)
        aSFi.sort_values(by='G', inplace=True)
        raw_aSF.append(aSFi)

    group = pd.concat(raw_SF).groupby('G')
    SF['S_G'] = group['S_G'].mean()
    SF['S_G_error'] = group['S_G'].sem()
    SF['S_G*V_G'] = group['S_G*V_G'].sum()/len(Coulomb_files)
    SF['V_G'] = group['V_G'].sum()/len(Coulomb_files)
    SF.reset_index(drop=False, inplace=True)
    SF.sort_values(by='G', inplace=True)

    return (raw_SF, raw_aSF, SF)


def find_special_twist_angle(raw_aSF: typing.List[pddataframe],
                             SF: pddataframe) -> int:
    ''' Find the twist angle corresponding to the minimum residual
    between the twist averaged S_G and a given S_G.

    Parameters
    ----------
    raw_aSF : list of :class:`pandas.DataFrame`
        A list of all the individual average structure factors.
    SF : :class:`pandas.DataFrame`
        A data frame of the average structure factor.

    Returns
    -------
    ispecial : integer
        The index of the special twist angle. The index is pythonic and
        matches the various lists used throughout.

    Raises
    ------
    RuntimeError
        When the average and individual structure factor data sets
        have different G values.
    '''
    residuals = []

    for aSFi in raw_aSF:
        residuals.append(np.power(np.abs(SF['S_G'] - aSFi['S_G']), 2).sum())

        if not np.array_equal(aSFi['G'], SF['G']):
            raise RuntimeError('G value arrays are not equivlent between'
                               'the average SF and an individual SF.'
                               'This should not happen!')

    ispecial = np.argmin(residuals)

    return ispecial


def write_sfTA_csv(csv_file: str, directories: typing.List[str],
                   raw_SF: typing.List[pddataframe]) -> None:
    ''' Write out the raw structure factor data in a format
    which is ammendable to sfTA.py.

    Parameters
    ----------
    csv_file : str
        A user provided name to save the data to.
    directories : list of strings
        The directories where structure factor data is contained.
    raw_SF : list of :class:`pandas.DataFrame`
        A list of all the structure factors.
    '''
    csv_twist = csv_file.replace('.csv', '_Twist_angle_Num_map.csv')

    csv_SF, csv_mp = [], {'Twist angle Num': [], 'directory': []}
    for i, (SFi, directory) in enumerate(zip(raw_SF, directories)):
        itwist = np.repeat(i+1, SFi['G'].shape[0])
        oSFi = pd.DataFrame({'Twist angle Num': itwist})
        oSFi[['G', 'V_G', 'S_G']] = SFi[['G', 'V_G', 'S_G']]
        csv_SF.append(oSFi.sort_values(by='G').reset_index(drop=True))

        csv_mp['Twist angle Num'].append(i+1)
        csv_mp['directory'].append(directory)

    print(f' Saving structure factor data to: {csv_file}', file=sys.stderr)
    pd.concat(csv_SF).to_csv(csv_file, index=False)
    print(f' Saving twist angle index map to: {csv_twist}', file=sys.stderr)
    pd.DataFrame(csv_mp).to_csv(csv_twist, index=False)


def write_individual_twist_average_csv(
            single_write: str, raw_aSF: typing.List[pddataframe],
        ) -> None:
    ''' Write out the average of the individual twist angles to a csv file.

    Parameters
    ----------
    single_write : str
        A user provided name to save the data to.
    raw_aSF : list of :class:`pandas.DataFrame`
        A list of all the individual average structure factors.
    '''
    individual_averages = []

    for i, aSFi in enumerate(raw_aSF):
        itwist = np.repeat(i+1, np.unique(aSFi['G']).shape[0])
        aSFi.insert(0, 'Twist angle Num', itwist)
        individual_averages.append(aSFi)

    pd.concat(individual_averages).to_csv(single_write, index=False)
    print(f' Saving individual averages to: {single_write}', file=sys.stderr)


def main(arguments: typing.List[str]) -> None:
    ''' Run structure factor twist averaging on cc4s outputs.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.
    '''
    StructureFactor(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
