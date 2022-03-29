#!/usr/bin/env python
'''analyse_cc4s.py [options] directory_1 directory_2 ... directory_N

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
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt


class ScriptTimer:
    ''' A simple timer for the analysis script.

    Attributes
    ----------
    start : float
        The start time set when the timer is initalized
    current : float
        The final time when the lap or stop methods are called
    total : float
        The time between start and current, updated by lap or stop
    '''
    def __report(self, msg):
        ''' Private method to report time and msg in a non-intrusive way '''
        print(f'\n {msg}: {self.total:>9.6f} (minutes)\n', file=sys.stderr)

    def __update(self):
        ''' Private method to update the time '''
        self.current = time.perf_counter()
        self.total = (self.current - self.start)/60.0

    @classmethod
    def start(cls):
        ''' Initialize the timer instance. '''
        cls.start = time.perf_counter()

    @classmethod
    def lap(cls, msg='Time for lap'):
        ''' Calculate an intemediate time in minutes, and report

        Parameters
        ----------
        msg : str
            A message the user would like to report when printing timings

        Returns
        -------
        None.
        '''
        cls.__update(cls)
        cls.__report(cls, msg)

    @classmethod
    def stop(cls):
        ''' Calculate the total time in minutes, and report '''
        cls.lap(msg='Script execution time')


def parse_command_line_arguments(arguments):
    ''' Parse command-line arguments.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.

    Returns
    -------
    directories : list of strings
        The directories where structure factor data is contained.
    options : :class:`ArgumentParser`
        User options read in from the command-line.
    '''
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument('-aw', '--average-write', action='store', default=None,
                        type=str, dest='average_write', help='Provide a file '
                        'to store the average structure factor data in. '
                        'The data format is csv, and the file flag is '
                        'automatically included.')
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

    def __ext_filename_check(filename, ext):
        ''' A private function to ensure user provided
        filenames have the corresponding extension.
        '''
        if filename is not None and ext not in filename:
            filename += ext
        return filename

    options.average_write = __ext_filename_check(options.average_write, '.csv')
    options.single_write = __ext_filename_check(options.single_write, '.csv')
    options.mp2_write = __ext_filename_check(options.mp2_write, '.csv')
    options.legacy_write = __ext_filename_check(options.legacy_write, '.csv')
    options.sfta_plot = __ext_filename_check(options.sfta_plot, '.png')
    options.variance_plot = __ext_filename_check(options.variance_plot, '.png')

    return options.directories, options


def plot_SF(sfta_plot, variance_plot, raw_SF, SF, ispecial):
    ''' Performs all the plotting that can occur based on user input.
    This could be either the individual structure factors, the average
    structure factor and the special twist. Or the variance of the average
    structure factor. Both can also occur at the same time.

    Parameters
    ----------
    sfta_plot : str
        A string for the structure factor plot name, default is None in
        which case no plot is created.
    variance_plot : str
        A string for the plot of the variance for the average structure
        factor. Default is None in which case no plot is created.
    raw_SF : list of :class:`pandas.DataFrame`
        A list of all the structure factors.
    SF : :class:`pandas.DataFrame`
        A data frame of the average structure factor.
    ispecial : integer
        The index of the special twist angle. The index is pythonic and
        matches the various lists used throughout.

    Returns
    -------
    None.
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

        for i, SFi in enumerate(raw_SF):
            aSFi = SFi.groupby('G', as_index=False).mean().sort_values('G')
            eSFi = SFi.groupby('G', as_index=False).sem().sort_values('G')

            plt.errorbar(
                    aSFi['G'],
                    aSFi['S_G'],
                    eSFi['S_G'],
                    label='S(G), MP2' if i == 1 else '',
                    color='#02a642',
                )

            if i == ispecial:
                plt.errorbar(
                        aSFi['G'],
                        aSFi['S_G'],
                        eSFi['S_G'],
                        label='S(G), BasisSetData-MP2',
                        color='#f26003',
                        marker='o',
                        ls='--',
                        markeredgecolor='k',
                        markeredgewidth=1.0,
                        zorder=15,
                    )

        SF = SF.sort_values('G')
        plt.errorbar(
                SF['G'],
                SF['S_G'],
                SF['S_G_error'],
                label='S(G), TA-MP2',
                color='#2c43fc',
                zorder=10,
            )

        plt.xlabel('G')
        plt.ylabel('S(G)')
        plt.legend(loc='best', ncol=1, handlelength=1.0, handletextpad=0.1)
        plt.savefig(sfta_plot, bbox_inches='tight')

    if variance_plot is not None:
        plt.clf()

        SF = SF.sort_values('G')
        sem_error = SF['S_G_error']/((2*(len(raw_SF) - 1))**0.5)
        variance = (SF['S_G_error'] * (len(raw_SF))**0.5)**2.0
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
                label='S(G), TA-MP2',
                color='#2c43fc',
                zorder=10,
                alpha=0.5,
            )

        plt.yscale('log')
        plt.xlabel('G')
        plt.ylabel('S(G) Variance')
        plt.legend(loc='best', ncol=1)
        plt.savefig(variance_plot, bbox_inches='tight')


def clean_paths_and_simple_checks(directories):
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


def find_yaml_outs(directories):
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


def get_yaml_as_dict(yaml_file):
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


def extract_mp2_from_yaml(yaml_out_files):
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
    def __ekey_check(cstep, nkey, ekey):
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


def find_SF_outputs(directories):
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


def read_and_generate_Gvector_magnitudes(Gvector_file):
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


def read_Vg(Coulomb_files):
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


def read_Sg(S_G_file):
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


def read_and_average_SF(Gvector_files, Coulomb_files, S_G_files):
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
    SF : :class:`pandas.DataFrame`
        A data frame of the average structure factor.
    '''
    raw_SF = []

    for files in zip(Gvector_files, Coulomb_files, S_G_files):
        G = read_and_generate_Gvector_magnitudes(files[0])
        V_G = read_Vg(files[1])
        S_G = read_Sg(files[2])
        SFi = pd.DataFrame({'G': G.round(10), 'V_G': V_G, 'S_G': S_G})
        raw_SF.append(SFi)

    group = pd.concat(raw_SF).groupby('G', as_index=False)
    SF = group.mean()
    SF[['V_G_error', 'S_G_error']] = group.sem()[['V_G', 'S_G']]

    return raw_SF, SF


def find_special_twist_angle(raw_SF, SF):
    ''' Find the twist angle corresponding to the minimum residual
    between the twist averaged S_G and a given S_G.

    Parameters
    ----------
    raw_SF : list of :class:`pandas.DataFrame`
        A list of all the structure factors.
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

    for SFi in raw_SF:
        aSFi = SFi.groupby('G', as_index=False).mean()
        residuals.append(np.power(np.abs(SF['S_G'] - aSFi['S_G']), 2).sum())

        if not np.array_equal(aSFi['G'], SF['G']):
            raise RuntimeError('G value arrays are not equivlent between'
                               'the average SF and an individual SF.'
                               'This should not happen!')

    ispecial = np.argmin(residuals)

    return ispecial


def write_sfTA_csv(csv_file, directories, raw_SF):
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

    Returns
    -------
    None.
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


def write_individual_twist_average_csv(single_write, raw_SF):
    ''' Write out the average of the individual twist angles to a csv file.

    Parameters
    ----------
    single_write : str
        A user provided name to save the data to.
    raw_SF : list of :class:`pandas.DataFrame`
        A list of all the structure factors.

    Returns
    -------
    None.
    '''
    individual_averages = []

    for i, SFi in enumerate(raw_SF):
        itwist = np.repeat(i+1, np.unique(SFi['G']).shape[0])
        aSFi = pd.DataFrame({'Twist angle Num': itwist})

        group = SFi.groupby('G', as_index=False)
        aSFi = group.mean()
        aSFi[['V_G_error', 'S_G_error']] = group.sem()[['V_G', 'S_G']]

        aSFi = aSFi.sort_values('G').reset_index(drop=True)
        individual_averages.append(aSFi)

    pd.concat(individual_averages).to_csv(single_write, index=False)
    print(f' Saving individual averages to: {single_write}', file=sys.stderr)


def main(arguments):
    ''' Run structure factor twist averaging on cc4s outputs.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.

    Returns
    -------
    None.
    '''
    directories, options = parse_command_line_arguments(arguments)

    directories = clean_paths_and_simple_checks(directories)

    if options.mp2 or options.mp2_write is not None:
        yaml_out_files = find_yaml_outs(directories)
        mp2_df = extract_mp2_from_yaml(yaml_out_files)

        if options.mp2:
            mp2_df_str = mp2_df.to_string(index=False, float_format='%24.16f')
            print(mp2_df_str, file=sys.stderr)

        if options.mp2_write is not None:
            msg = f' Saving MP2 energies to: {options.mp2_write}'
            print(msg, file=sys.stderr)
            mp2_df.to_csv(options.mp2_write, index=False)

    if options.skip_sfta:
        return

    Gvector_files, Coulomb_files, S_G_files = find_SF_outputs(directories)

    raw_SF, SF = read_and_average_SF(Gvector_files, Coulomb_files, S_G_files)

    ispecial = find_special_twist_angle(raw_SF, SF)

    if options.legacy_write is not None:
        write_sfTA_csv(options.legacy_write, directories, raw_SF)

    if options.average:
        SF_str = SF.to_string(index=False, float_format='%24.16f')
        print(SF_str, file=sys.stderr)

    if options.average_write is not None:
        msg = f' Saving average structure factor to: {options.average_write}'
        print(msg, file=sys.stderr)
        SF.to_csv(options.average_write, index=False)

    if options.single_write is not None:
        write_individual_twist_average_csv(options.single_write, raw_SF)

    if options.sfta_plot is not None or options.variance_plot is not None:
        plot_SF(options.sfta_plot, options.variance_plot, raw_SF, SF, ispecial)

    print('\n Found Special Twist Angle:', file=sys.stderr)
    print(f' {directories[ispecial]}\n', file=sys.stderr)


if __name__ == '__main__':
    ScriptTimer.start()
    main(sys.argv[1:])
    ScriptTimer.stop()
