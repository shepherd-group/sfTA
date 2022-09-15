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
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from warnings import warn
from yaml import safe_load
from typing import TypeVar, List, Tuple, Type

Array = TypeVar('np.ndarray')
Dataframe = TypeVar('pd.core.frame.DataFrame')


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
    read_data_and_analyze(directories, msg)
        A data parsing wrapper for structure factor analysis.
    structure_factor_linear_combination()
        A function for combining structure factors in a linear fashion.
    do_basic_analysis()
        A simple wrapper function to save and plot a single structure factor.
    update_timing_report(msg)
        Adds a timing check point to the timing report dictionary.
    end_timing_report()
        Close out the timing report and print out to the user.
    '''
    def __init__(self, clargs: List[str], fmt: str = '%24.16f') -> None:
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

        if self.options.order_directories:
            directories = human_readable_reordering(self.options.directories)
            self.options.directories = directories
            self.options.addp = human_readable_reordering(self.options.addp)

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
            self.end_timing_report()
            return

        self.SFi, self.aSFi, self.aSF = self.read_data_and_analyze(directories)

        if self.options.basic_analysis is not None:
            self.do_basic_analysis()
            self.end_timing_report()
            return

        if self.options.addp is not None:
            self.structure_factor_linear_combination()

        special_data = find_special_twist_angle(self.aSFi, self.aSF,
                                                self.options.weighted_residual,
                                                self.options.anisotropic,
                                                self.options.upper_bound,
                                                self.options.lower_bound)
        self.ispecial, residuals = special_data
        self.update_timing_report(msg='Special twist analysis')

        residual_report = self.options.print_residuals
        if self.options.residual_write is not None or residual_report:
            residual_df = pd.DataFrame({
                    'Twist': np.arange(1, len(self.directories)+1),
                    'Path': self.directories,
                    'Residual': residuals,
                    'Rank': np.argsort(residuals).argsort() + 1,
                })
            if self.options.residual_write is not None:
                msg = ' Saving residuals data to:'
                print(f'{msg} {self.options.residual_write}', file=sys.stderr)
                residual_df.to_csv(self.options.residual_write, index=False)
                self.update_timing_report(msg='Residual saving')
            if residual_report:
                residual_df[' '] = [' ' for _ in range(len(residuals))]
                residual_df.loc[self.ispecial, ' '] = '<--- (Minimum)'
                res_str = residual_df.to_string(index=False, float_format=fmt)
                print(res_str, file=sys.stderr)
                self.update_timing_report(msg='Residual reporting')

        if self.options.legacy_write is not None:
            legacy_output = self.options.legacy_write
            write_sfTA_csv(legacy_output, self.directories, self.SFi)
            self.update_timing_report(msg='Raw structure factor storing')

        if self.options.average:
            SF_str = self.aSF.to_string(index=False, float_format=fmt)
            print(SF_str, file=sys.stderr)
            self.update_timing_report(msg='Twist average dumping')

        if self.options.special:
            SF_str = self.aSFi[self.ispecial]
            SF_str = SF_str.to_string(index=False, float_format=fmt)
            print(SF_str, file=sys.stderr)
            self.update_timing_report(msg='Special twist dumping')

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
                    self.aSF, self.ispecial, self.options.anisotropic)
            self.update_timing_report(msg='Structure factor plotting')

        if self.options.special_write is not None:
            special_output = self.options.special_write
            msg = ' Saving special twist angle structure factor to:'
            print(f'{msg} {special_output}', file=sys.stderr)
            self.aSFi[self.ispecial].to_csv(special_output, index=False)
            self.update_timing_report(msg='Special twist storing')

        self.end_timing_report()

        print(f'\nFound Special Twist Angle: {self.ispecial} '
              f'({self.ispecial + 1}) | zero-indexed (one-indexed)',
              file=sys.stderr)
        print(f'Path: {self.directories[self.ispecial]}\n', file=sys.stderr)

    def read_data_and_analyze(
                self,
                directories: List[str],
                amsg: str = ''
            ) -> Tuple[List[Dataframe], Dataframe]:
        ''' A wrapper for finding the ouput files, parsing the output files,
        and finally performing analysis on the output files. The resulting
        raw data and analyzed data are returned.

        Parameters
        ----------
        directories : list of str
            The locations of the structure factor data to be collected.
        amsg : str, default=''
            A message to append on to the timing report for the collection
            and analysis of data.

        Returns
        -------
        raw_SF : list of :class:`pandas.DataFrame`
            A list of all the structure factors.
        raw_aSF : list of :class:`pandas.DataFrame`
            A list of all the average structure factors
            for a given twist angle.
        SF : :class:`pandas.DataFrame`
            A data frame of the average structure factor.
        '''
        output_files = find_SF_outputs(directories)
        self.update_timing_report(msg=f'Structure factor output search{amsg}')
        sf_tuple = read_and_average_SF(*output_files, self.options.anisotropic)
        self.update_timing_report(msg='Structure factor parsing '
                                  f'and analysis{amsg}')
        return sf_tuple

    def structure_factor_linear_combination(self) -> None:
        ''' Calculate the linear combination of two structure factors
        provided by the user. This is performed for the raw, average individual
        and twist averaged structure factors. Only the S(G) and S(G) error
        terms are considered. The addition is defined by the addop parameter
        and the resulting data is stored in the standard arrays/dataframes.

        Raises
        ------
        RuntimeError
            If the number of directories and addition directories are not
            the same.
        RuntimeError
            If the G values are not the same between the structure factors
            from the directories and addition directories.
        '''
        terminate, fac, sk, ek = False, self.options.addop, 'S_G', 'S_G_error'

        print('Calculating the structure factor as a linear combination of:\n'
              f'    "directories" + {self.options.addop} x "addp"\n'
              'The post-analysis S(G) and S(G) errors saved in csv files will '
              'reflect this!\n', file=sys.stderr)

        if len(self.options.directories) != len(self.options.addp):
            raise RuntimeError('The provided number of directories '
                               f'(N={len(self.options.directories)}) is '
                               'not the same as the number of addition '
                               f'directories (N={len(self.options.addp)})!')

        def _SGADD(T1: Dataframe, T2: Dataframe, C: float) -> Array:
            ''' Private function to add two pandas columns with a constant.'''
            return T1.values.flatten() + C*T2.values.flatten()

        def _SGPRP(T1: Dataframe, T2: Dataframe, C: float) -> Array:
            ''' Private function to calculate the additive error for _SGADD.'''
            return (T1.values.flatten()**2.0
                    + (C*T2.values.flatten())**2.0)**0.5

        diff = self.read_data_and_analyze(self.options.addp, amsg=' (-addp)')
        d0, d1, d2 = diff

        if not np.array_equal(d2['G'], self.aSF['G']):
            terminate = True
        else:
            self.aSF[sk] = _SGADD(self.aSF[sk], d2[sk], fac)
            self.aSF[ek] = _SGPRP(self.aSF[ek], d2[ek], fac)

        for i in range(len(self.aSFi)):
            if not np.array_equal(d0[i]['G'], self.SFi[i]['G']):
                terminate = True
            else:
                self.SFi[i][sk] = _SGADD(self.SFi[i][sk], d0[i][sk], fac)

            if not np.array_equal(d1[i]['G'], self.aSFi[i]['G']):
                terminate = True
            else:
                self.aSFi[i][sk] = _SGADD(self.aSFi[i][sk], d1[i][sk], fac)
                self.aSFi[i][ek] = _SGPRP(self.aSFi[i][ek], d1[i][ek], fac)

        if terminate:
            raise RuntimeError('Structure factors used in addition '
                               'do not have matching G values!')
        else:
            self.update_timing_report(msg='Linear combination of S(G).')

    def do_basic_analysis(self) -> None:
        ''' Plot a single provided structure factor and store the structure
        factor in a csv file.

        Raises
        ------
        RuntimeError
            If there is more than a single twist angle provided.
        '''
        if len(self.options.directories) > 1:
            raise RuntimeError('The basic analysis options "-ba" only works '
                               'with a single directory, you provided '
                               f'{len(self.options.directories)}!')

        print(' Saving structure factor data to: '
              f'{self.options.basic_analysis}.csv', file=sys.stderr)
        self.aSFi[0].to_csv(f'{self.options.basic_analysis}.csv', index=False)
        plot_single_SF(self.options.basic_analysis, self.aSFi[0])
        self.update_timing_report(msg='Basic analysis data storing/plotting')

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
    parser.add_argument('-rw', '--residual-write', action='store',
                        default=None, type=str, dest='residual_write', help=''
                        'A file to write the residuals to in a csv format.')
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
    parser.add_argument('-ba', '--basic-analysis', action='store',
                        default=None, type=str, dest='basic_analysis',
                        help='Provide a file name to (1) store the structure '
                        'factor data and (2) plot the corresponding data. '
                        'If used, only a single calculation may be provided '
                        'for analysis, and many sfTA related parameters '
                        'will be ignored.')
    parser.add_argument('-ub', '--upper-bound', action='store', default=None,
                        type=float, dest='upper_bound', help='Set an upper '
                        'bound on the G values to use when calculating '
                        'the special twist, everything equal to or below this '
                        'is used and everything above is not.')
    parser.add_argument('-lb', '--lower-bound', action='store', default=None,
                        type=float, dest='lower_bound', help='Set a lower '
                        'bound on the G values to use when calculating '
                        'the special twist, everything equal to or above this '
                        'is used and everything below is not.')
    parser.add_argument('-addp', '--addition-paths', nargs='+',
                        dest='addp', help='Provide a set of structure '
                        'factor paths with structure factor data that will be '
                        'used to calculate an addition between two structure '
                        'factors. To define the addition see the addop '
                        'parameter. The number of paths must match that of '
                        'the directories provided for analysis. This option '
                        'must follow the main analysis directories.')
    parser.add_argument('-addop', '--addition-operator', default=1.0,
                        dest='addop', type=float, help='Used to alter the '
                        'linear combination of the directories structure '
                        'factors and the additive paths structure factors. '
                        'For example, one may calculate the binding structure '
                        'factor for a monolayer and bilayer by providing '
                        'the bilayer as the directories, the monolayer as the '
                        'additive path, and the operator as -2.0. The default '
                        'addition operator is 1.0.')
    parser.add_argument('-a', '--average', action='store_true', default=False,
                        dest='average', help='Print out the average structure '
                        'factor in a nice table to the standard error output.')
    parser.add_argument('-s', '--special', action='store_true', default=False,
                        dest='special', help='Print out the special twist '
                        'structure factor in a nice table.')
    parser.add_argument('-e', '--mp2', action='store_true', default=False,
                        dest='mp2', help='Pull the MP2 energies and print out '
                        'as a table to the standard error output.')
    parser.add_argument('-w', '--weighted-residual', action='store_true',
                        default=False, dest='weighted_residual', help='Use '
                        'a weight 1/|G|^2 for the difference when calculating '
                        'the residuals to find the special twist angle.')
    parser.add_argument('-n', '--anisotropic', action='store_true',
                        default=False, dest='anisotropic', help='Perform '
                        'structure factor twist averaging and select the '
                        'special twist angle by performing an anisotropic '
                        'twist averaging scheme.')
    parser.add_argument('-r', '--print-residuals', action='store_true',
                        default=False, dest='print_residuals', help='Report '
                        'the residuals from the structure factor analysis '
                        'to the standard error stream.')
    parser.add_argument('-o', '--order-directories', action='store_true',
                        default=False, dest='order_directories',
                        help='Attempt to order the provided directories '
                        'in a human readable fashion. Note, this has not been '
                        'extensively tested, so use at your own discretion '
                        'and consider checking the sorted results.')
    parser.add_argument('-k', '--skip-sfta', action='store_true',
                        default=False, dest='skip_sfta', help='Skip all forms '
                        'of sfTA analysis. I.E., overrides related settings!')
    parser.add_argument('directories', nargs='+', help='Paths containing '
                        'Structure Factor data to be analyzed.')
    parser.parse_args(args=None if arguments else ['--help'])

    options = parser.parse_args(arguments)

    def _ext_check(filename: str, ext: str) -> str:
        ''' A private function to ensure user provided
        filenames have the corresponding extension.
        '''
        if filename is not None and ext not in filename:
            filename += ext
        return filename

    options.average_write = _ext_check(options.average_write, '.csv')
    options.special_write = _ext_check(options.special_write, '.csv')
    options.single_write = _ext_check(options.single_write, '.csv')
    options.mp2_write = _ext_check(options.mp2_write, '.csv')
    options.residual_write = _ext_check(options.residual_write, '.csv')
    options.legacy_write = _ext_check(options.legacy_write, '.csv')
    options.sfta_plot = _ext_check(options.sfta_plot, '.png')
    options.difference_plot = _ext_check(options.difference_plot, '.png')
    options.variance_plot = _ext_check(options.variance_plot, '.png')

    return options


def plot_single_SF(plot_name: str, SF: Dataframe) -> None:
    ''' Creates a simple plot for a single structure factor.

    Parameters
    ----------
    plot_name : str
        A string for the structure factor plot name, default is None in
        which case no plot is created. The "png" file extension is added.
    SF : :class:`pandas.DataFrame`
        A data frame of the structure factor under consideration.

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

    print(' Saving structure factor plot to: '
          f'{plot_name}.png', file=sys.stderr)

    plt.clf()

    plt.plot(
            SF['G'],
            SF['S_G'],
            label='individual twist',
            marker='x',
            c='#02a642',
            mec='k',
        )

    plt.xlabel('G')
    plt.ylabel('S(G)')
    plt.legend(loc='best', ncol=1, handlelength=1.0, handletextpad=0.1)
    plt.savefig(f'{plot_name}.png', bbox_inches='tight')


def plot_SF(sfta_plot: str, difference_plot: str, variance_plot: str,
            raw_aSF: List[Dataframe], SF: Dataframe, ispecial: int,
            anisotropic: bool = False) -> None:
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
    anisotropic : bool, default=False
        Controls whether the structure factor averaging
        is done with or with G vector averaging/matching.

    Returns
    -------
    None.
    '''
    # TODO - WZV
    # This function needs to be moved to its own file
    # and the code should be cleaned/organized better.
    font = {'family': 'serif', 'sans-serif': 'Computer Modern Roman'}
    mpl.rc('font', **font)
    mpl.rc('savefig', dpi=300)
    mpl.rc('lines', lw=2, markersize=5)
    mpl.rc('legend', fontsize=8, numpoints=1)
    mpl.rc(('axes', 'xtick', 'ytick'), labelsize=8)
    mpl.rc('figure', dpi=300, figsize=(3.37, 3.37*(np.sqrt(5)-1)/2))

    if sfta_plot is not None:
        plt.clf()

        if anisotropic:
            sfta_plot = sfta_plot.replace('png', 'pdf')
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d', computed_zorder=False)

            S_G_min = 0.25*SF[SF['Gz'] == 0]['S_G'].min()

            for i, aSFi in enumerate(raw_aSF):
                z0 = aSFi['Gz'] == 0.0
                ax.scatter(
                        aSFi.loc[z0, 'Gx'],
                        aSFi.loc[z0, 'Gy'],
                        aSFi.loc[z0, 'S_G'],
                        label='individual' if i == 1 else '',
                        color='#02a642',
                        s=np.abs(aSFi.loc[z0, 'S_G']/S_G_min),
                        marker='^',
                        zorder=2,
                    )

                if i == ispecial:
                    ax.scatter(
                            aSFi.loc[z0, 'Gx'],
                            aSFi.loc[z0, 'Gy'],
                            aSFi.loc[z0, 'S_G'],
                            label='special',
                            color='#f26003',
                            s=np.abs(aSFi.loc[z0, 'S_G']/S_G_min),
                            marker='o',
                            edgecolors='k',
                            linewidths=0.1,
                            zorder=15,
                        )

            z0 = SF['Gz'] == 0.0
            ax.scatter(
                    SF.loc[z0, 'Gx'],
                    SF.loc[z0, 'Gy'],
                    SF.loc[z0, 'S_G'],
                    label='averaged',
                    color='#2c43fc',
                    s=np.abs(SF.loc[z0, 'S_G']/S_G_min),
                    marker='x',
                    zorder=16,
                )

            ax.set_xlabel(r'$\vec{G}_x$')
            ax.set_ylabel(r'$\vec{G}_y$')
            ax.set_zlabel(r'$S(\vec{G})$')
            ax.view_init(8, -55)
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3,
                       fontsize=6, handlelength=1.0, handletextpad=0.1)
            print(' Saving structure factor plot to: '
                  f'{sfta_plot}', file=sys.stderr)
            plt.savefig(sfta_plot, bbox_inches='tight')
        else:
            for i, aSFi in enumerate(raw_aSF):
                plt.plot(
                        aSFi['G'],
                        aSFi['S_G'],
                        label='individual twists' if i == 1 else '',
                        color='#02a642',
                    )

                if i == ispecial:
                    plt.plot(
                            aSFi['G'],
                            aSFi['S_G'],
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
                    np.sqrt(0.0*aSFi['S_G_error']**2 + SF['S_G_error']**2),
                    label='individual twists' if i == 1 else '',
                    color='#02a642',
                )

            if i == ispecial:
                plt.errorbar(
                        aSFi['G'],
                        aSFi['S_G'] - SF['S_G'],
                        np.sqrt(0.0*aSFi['S_G_error']**2 + SF['S_G_error']**2),
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
            directories: List[str],
        ) -> List[str]:
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
        warn(f'\nNon-directory paths provided: {removed_paths}'
             'These are removed!\n', stacklevel=2)

    if len(cleaned_directories) != 100:
        warn(f'\nThere are {len(cleaned_directories)},'
             ' not 100 calculations!\n', stacklevel=2)

    if np.unique(directories).shape[0] != len(directories):
        raise RuntimeError('Repeated directories found!')

    return cleaned_directories


def find_yaml_outs(directories: List[str]) -> List[str]:
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
        yaml_dict = safe_load(yaml_stream)

    return yaml_dict


def extract_mp2_from_yaml(yaml_out_files: List[str]) -> Dataframe:
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
    def _ekey_check(cstep: str, nkey: str, ekey: str) -> bool:
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
            if _ekey_check(step, 'CoupledCluster', 'correlation'):
                if imp2 == 0:
                    mp2_df['Ec'], mp2_df['Ed'], mp2_df['Ex'] = [], [], []

                mp2_df['Ec'].append(step['out']['energy']['correlation'])
                mp2_df['Ed'].append(step['out']['energy']['direct'])
                mp2_df['Ex'].append(step['out']['energy']['exchange'])

            if _ekey_check(step, 'FiniteSizeCorrection', 'correction'):
                if imp2 == 0:
                    mp2_df['FSC'] = []

                mp2_df['FSC'].append(step['out']['energy']['correction'])

            if _ekey_check(step, 'BasisSetCorrection', 'correction'):
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
            directories: List[str],
        ) -> Tuple[List[str]]:
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


def read_and_generate_Gvector_magnitudes(
            Gvector_file: str,
            anisotropic: bool = False,
        ) -> Array:
    ''' Read in a GridVectors.elements file generated by cc4s and calculate the
    G magnitudes for sFTA.

    Parameters
    ----------
    Gvector_file : string
        A file which contains the G vectors.
    anisotropic : bool, default=False
        If true, return the G vectors and the G magnitudes.

    Returns
    -------
    G : :class:`numpy.ndarray`
        An array of the G magnitudes, and optionally the G vectors.
    '''
    raw_g_xyz = np.loadtxt(Gvector_file, dtype=np.float64)
    N_G = int(raw_g_xyz.shape[0] / 3)
    g_xyz = raw_g_xyz.reshape((N_G, 3))

    if anisotropic:
        G = np.zeros((N_G, 4), dtype=np.float64)
        G[:, 0] = np.sqrt(np.einsum('ij,ij->i', g_xyz, g_xyz))
        G[:, 1:] = g_xyz
    else:
        G = np.sqrt(np.einsum('ij,ij->i', g_xyz, g_xyz))

    return G


def read_Vg(Coulomb_files: str) -> Array:
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


def read_Sg(S_G_file: str) -> Array:
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
            Gvector_files: List[str],
            Coulomb_files: List[str],
            S_G_files: List[str],
            anisotropic: bool = False,
        ) -> Tuple[List[Dataframe], Dataframe]:
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
    anisotropic : bool, default=False
        Controls whether the structure factor averaging
        is done with or with G vector averaging/matching.

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

        if anisotropic:
            G = read_and_generate_Gvector_magnitudes(files[0], anisotropic)
            Gkxyz = ['G', 'Gx', 'Gy', 'Gz']
            SFi = pd.DataFrame({
                    'index': np.arange(G[:, 0].shape[0]),
                    'G': G[:, 0],
                    'Gx': G[:, 1],
                    'Gy': G[:, 2],
                    'Gz': G[:, 3],
                })

            SFi['V_G'] = read_Vg(files[1])
            SFi['S_G'] = read_Sg(files[2])

            raw_SF.append(SFi)
        else:
            G = read_and_generate_Gvector_magnitudes(files[0], anisotropic)
            SFi = pd.DataFrame({
                    'G': G.round(10),
                })

            SFi['V_G'] = read_Vg(files[1])
            SFi['S_G'] = read_Sg(files[2])

            raw_SF.append(SFi)

            group = SFi.groupby('G')
            aSFi['S_G'] = group['S_G'].mean()
            aSFi['S_G_error'] = group['S_G'].sem()
            aSFi['V_G'] = group['V_G'].sum()
            aSFi['N_G'] = group['V_G'].count()
            aSFi.reset_index(drop=False, inplace=True)
            aSFi.sort_values(by='G', inplace=True)
            raw_aSF.append(aSFi)

    if anisotropic:
        group = pd.concat(raw_SF).groupby('index')
        SF[Gkxyz] = group[Gkxyz].mean()
        SF[[f'{k}_error' for k in Gkxyz]] = group[Gkxyz].sem()
    else:
        group = pd.concat(raw_SF).groupby('G')

    SF['S_G'] = group['S_G'].mean()
    SF['S_G_error'] = group['S_G'].sem()
    SF['V_G'] = group['V_G'].sum()/len(Coulomb_files)
    SF.reset_index(drop=False, inplace=True)

    if anisotropic:
        raw_aSF = raw_SF
    else:
        SF.sort_values(by='G', inplace=True)

    return (raw_SF, raw_aSF, SF)


def find_special_twist_angle(
            raw_aSF: List[Dataframe],
            SF: Dataframe,
            use_weighted_residuals: bool = False,
            anisotropic: bool = False,
            upper_bound: float = None,
            lower_bound: float = None,
        ) -> Tuple[int, List[float]]:
    ''' Find the twist angle corresponding to the minimum residual
    between the twist averaged S_G and a given S_G.

    Parameters
    ----------
    raw_aSF : list of :class:`pandas.DataFrame`
        A list of all the individual average structure factors.
    SF : :class:`pandas.DataFrame`
        A data frame of the average structure factor.
    use_weighted_residuals : bool, default=False
        Controls whether the difference is weighted by the 1/|G|^2 values
        when calculating the residual.
    anisotropic : bool, default=False
        Controls whether the structure factor averaging
        is done with or with G vector averaging/matching.
    upper_bound : float, default=None
        Set an upper bound on the G values to use when calculating
        the residuals to select the special twist angle. Everything
        equal to or below this value is considered in the residual
        calculation and everything above is not.
    lower_bound : float, default=None
        Similar to uppwer bound, sets a threshold in G, for which
        the structure factor residual is calculated on. The considered G
        values are only those equal to or greater than this threshold.

    Returns
    -------
    ispecial : integer
        The index of the special twist angle. The index is pythonic and
        matches the various lists used throughout.
    residuals : list of floats
        The residuals for the provided structure factors.

    Raises
    ------
    RuntimeError
        If the G vector residual is non-zero
    RuntimeError
        When the average and individual structure factor data sets
        have different G values.
    '''
    residuals = []

    def _LUMASK(ARR: Array, G: Array, GL: float, GU: float) -> Array:
        ''' Private function to apply S(G) masking based on GL/GU.'''
        if GL is not None and GU is not None:
            return ARR[np.logical_and(G >= GL, G <= GU)]
        elif GL is not None:
            return ARR[G >= GL]
        elif GU is not None:
            return ARR[G <= GU]
        else:
            return ARR

    if lower_bound is not None or upper_bound is not None:
        tmsg = '    '
        tmsg += f'{lower_bound:>.4f} =<' if lower_bound is not None else ''
        tmsg += ' G '
        tmsg += f'=< {upper_bound:>.4f}' if upper_bound is not None else ''
        tmsg += ' \n'
        print(f'Truncating structure factor to: \n{tmsg}'
              'for selecting the special twist!\n', file=sys.stderr)

    mean_G = SF['G'].values.flatten()
    mean_SG = SF['S_G'].values.flatten()
    mean_SG = _LUMASK(mean_SG, mean_G, lower_bound, upper_bound)

    for aSFi in raw_aSF:
        G = aSFi['G'].values.flatten()
        SG = aSFi['S_G'].values.flatten()
        SG = _LUMASK(SG, G, lower_bound, upper_bound)

        delta_S_G = np.power(np.abs(mean_SG - SG), 2)

        if anisotropic:
            delta_Gx = np.power(np.abs(SF['Gx'] - aSFi['Gx']), 2).sum()
            delta_Gy = np.power(np.abs(SF['Gy'] - aSFi['Gy']), 2).sum()
            delta_Gz = np.power(np.abs(SF['Gz'] - aSFi['Gz']), 2).sum()

            delta_S_G += delta_Gx + delta_Gy + delta_Gz
            if (abs(delta_Gx) + abs(delta_Gy) + abs(delta_Gz)) > 1E-12:
                raise RuntimeError('The \\vector{G} residual is non-zero!')

        if use_weighted_residuals:
            delta_S_G /= np.power(np.abs(mean_G), 2)

        residuals.append(delta_S_G.sum())

        if not anisotropic:
            if not np.array_equal(G, mean_G):
                raise RuntimeError('G value arrays are not equivlent between '
                                   'the average SF and an individual SF. '
                                   'This should not happen!')

    ispecial = np.argmin(residuals)

    return ispecial, residuals


def write_sfTA_csv(csv_file: str, directories: List[str],
                   raw_SF: List[Dataframe]) -> None:
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
        if 'G' in SFi.columns:
            itwist = np.repeat(i+1, SFi['G'].shape[0])
            dkeys = ['G', 'V_G', 'S_G']
        else:
            itwist = np.repeat(i+1, SFi['Gx'].shape[0])
            dkeys = ['Gx', 'Gy', 'Gz', 'V_G', 'S_G']
        oSFi = pd.DataFrame({'Twist angle Num': itwist})
        oSFi[dkeys] = SFi[dkeys]
        csv_SF.append(oSFi.sort_values(by='G').reset_index(drop=True))

        csv_mp['Twist angle Num'].append(i+1)
        csv_mp['directory'].append(directory)

    print(f' Saving structure factor data to: {csv_file}', file=sys.stderr)
    pd.concat(csv_SF).to_csv(csv_file, index=False)
    print(f' Saving twist angle index map to: {csv_twist}', file=sys.stderr)
    pd.DataFrame(csv_mp).to_csv(csv_twist, index=False)


def write_individual_twist_average_csv(
            single_write: str, raw_aSF: List[Dataframe],
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
        if 'G' in aSFi.columns:
            itwist = np.repeat(i+1, np.unique(aSFi['G']).shape[0])
        else:
            itwist = np.repeat(i+1, aSFi['Gx'].shape[0])
        aSFi.insert(0, 'Twist angle Num', itwist)
        individual_averages.append(aSFi)

    pd.concat(individual_averages).to_csv(single_write, index=False)
    print(f' Saving individual averages to: {single_write}', file=sys.stderr)


def human_readable_reordering(los: List[str]) -> List[str]:
    ''' Attempt to sort a list in a human readable fashion. This is done by
    first removing all non-numerical characters, and the remaining numerical
    characters are padded with zeros and inserted back into the original
    string. Thereafter the sorting procedure is trivial (hopefully).

    Parameters
    ----------
    los : list of strings
        The list of strings to attempt to sort in a human readable fashion.

    Returns
    -------
    sorted_los : list of strings
        The hopefully human sorted list of strings.
    '''
    if los is None:
        return None
    elif len(los) < 2:
        return los

    sorted_los = [k for k in los]

    padded_los = []
    for string in los:
        numeric_string = ''
        alphabetical_string = ''

        for character in string:
            if character.isdigit():
                numeric_string += character
                alphabetical_string += ' '
            else:
                numeric_string += ' '
                alphabetical_string += character

        string_lon = numeric_string.split()
        string_los = alphabetical_string.split()
        lon_len, los_len = len(string_lon), len(string_los)
        first_character_is_digit = string[0].isdigit()
        npad = max([len(number) for number in string_lon]) + 2

        padded_string = ''

        while lon_len > 0 or los_len > 0:
            if lon_len != 0 and los_len != 0:
                if first_character_is_digit:
                    padded_string += string_lon[0].zfill(npad)
                    string_lon.pop(0)
                    padded_string += string_los[0]
                    string_los.pop(0)
                else:
                    padded_string += string_los[0]
                    string_los.pop(0)
                    padded_string += string_lon[0].zfill(npad)
                    string_lon.pop(0)
            elif lon_len != 0 and los_len == 0:
                for number in string_lon:
                    padded_string += number.zfill(npad)
                string_lon = []
            elif lon_len == 0 and los_len != 0:
                for sub_string in string_los:
                    padded_string += sub_string
                string_los = []

            lon_len, los_len = len(string_lon), len(string_los)

        padded_los.append(padded_string)

    sorted_los = list(np.array(sorted_los)[np.argsort(padded_los)])

    return sorted_los


def main(arguments: List[str]) -> None:
    ''' Run structure factor twist averaging on cc4s outputs.

    Parameters
    ----------
    arguments : list of strings
        User provided command-line arguments.
    '''
    StructureFactor(arguments)


if __name__ == '__main__':
    main(sys.argv[1:])
