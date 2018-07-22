#!/usr/bin/env python
# #-*-coding:utf-8 -*-
#########################################################################
# File Name   :  perp.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-07-18 23:09
# purpose     :
#########################################################################

import argparse
import os
import sys
from typing import List

import numpy as np
from ase import io, units


def get_poscar_files(directory: str, recursive: bool) -> List:
    """Get the POSCAR file in the recursive mode

    :directory: TODO
    :returns: TODO

    """
    # walk directory (recursively) and return all poscar* files
    # return list of poscars' path
    sys.stderr.write(
        'searching directory %s for poscar* files ...\n' % directory)
    poscars = []
    if not recursive:
        for item in os.listdir(directory):
            if item.startswith('POSCAR'):
                poscars.append(os.path.join(directory, item))
    else:
        for root, subfolders, files in os.walk(directory):
            for item in files:
                if item.startswith('POSCAR'):
                    poscars.append(os.path.join(root, item))
    if len(poscars) == 0:
        sys.stderr.write(
            'could not find any poscar files in this directory.\n')
    else:
        sys.stderr.write('found the following files:\n')
        sys.stderr.write('  {}\n'.format(('\n  ').join(poscars)))
        return poscars
    return poscars


def Parser() -> "Namespace":
    """Parser file from command line
    :returns: TODO

    """
    parser = argparse.ArgumentParser(
        description='''get perturbation from the exist POSCAR''')

    parser.add_argument(
        '-r',
        '--recursive',
        action='store_true',
        required=False,
        help='scan recursively for OUTCAR files')
    parser.add_argument(
        '-out',
        '--output',
        type=str,
        default='iter',
        help='Set the output iter parent directory of the POSCAR')
    parser.add_argument(
        '-ra',
        '--ratio',
        type=float,
        default=0.2,
        help='Set the configuration change ratio')
    parser.add_argument(
        '-div',
        '--divides',
        type=int,
        default=100,
        help='Set the divides of the density')
    parser.add_argument(
        '-den',
        '--density',
        type=float,
        default=0.88,
        help='Set the center density of the whole phase graph')
    parser.add_argument(
        'files', type=str, nargs='*', help='list of POSCAR files (plain)')

    args = parser.parse_args()
    return args


def uniq(seq: List) -> List:
    # return unique list without changing the order
    # from http://stackoverflow.com/questions/480214
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


def perp() -> None:
    """TODO: set the 

    :arg1: TODO
    :returns: TODO

    """
    args = Parser()

    # determine all poscar files
    poscars: List = []

    if args.files:
        for item in args.files:
            if os.path.isdir(item):
                poscars += get_poscar_files(item, args.recursive)
            else:
                poscars.append(item)

    # Get the unique list
    poscars = uniq(poscars)

    # Get the ratio
    ratio = args.ratio

    # Get the density divides
    divides = args.divides

    # Get the centering density
    goal_density = args.density

    # Get the output directory
    # output_dir = os.path.join(args.output, 'iter')
    output_dir = os.path.join('.', args.output)

    # now start change

    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # for debug
    # print(poscars)

    for set_index, poscar_file in enumerate(poscars):
        # create path for set
        set_path = os.path.join(output_dir, f'set_{set_index}')
        if not os.path.exists(set_path):
            os.makedirs(set_path)

        configuration = io.read(poscar_file)
        cell = configuration.cell
        positions = np.copy(configuration.positions)
        # density unit g/cm^3
        density = np.sum(configuration.get_masses()) * \
            units._amu / configuration.get_volume() * 10**27

        # print('density', density)
        # print('mass', configuration.get_masses())
        # print('goal_density', goal_density)

        for new_config_index in range(divides + 1):
            density_ratio = goal_density / density * \
                        (((new_config_index - divides // 2) / divides *
                           ratio) + 1)
            # print("density_ratio", density_ratio)
            configuration.cell = cell / pow(density_ratio, 1 / 3)
            configuration.positions = positions / pow(density_ratio, 1 / 3)
            print(positions)

            # Create configuration path
            config_path = os.path.join(set_path, f'config_{new_config_index}')
            if not os.path.exists(config_path):
                os.makedirs(config_path)

            poscar_path = os.path.join(config_path, 'POSCAR')
            io.write(poscar_path, configuration, 'vasp')


if __name__ == "__main__":
    perp()
