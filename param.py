#!/usr/bin/env python3
# #-*-coding:utf-8 -*-
#########################################################################
# File Name   :  param.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-07-20 03:34
# purpose     :
#########################################################################

import argparse
import json
import os
from typing import Dict

from deepmd_lib import deepmd_iter
from lmp_lib import lmp_iter
from vasp_lib import vasp_iter


def parser() -> object:
    """Parser file from command line
    :returns: 

    """
    parser = argparse.ArgumentParser(
        description='''Deepmd ab-inito genrator''')

    parser.add_argument(
        'file', type=str, help='Get the json configuration file')

    parser.add_argument(
        '-c',
        '--continue_train',
        action='store_true',
        help='continue from the generator_checkpoint.json')

    args = parser.parse_args()
    return args


def generator():
    """General iterations
    Implement the first iteration in the vasp_iter() function
    get initial poscar from vasp_sys_dir
    :returns: 

    """
    args: object = parser()

    data = {}
    with open(args.file, "r") as json_file:
        data: Dict = json.load(json_file)

    # number_iter = data['numb_iter']
    vasp_data = data['vasp']
    deepmd_data = data['deepmd']
    lmp_data = data['lmp']

    # Specify total iteration number defined from lammps
    iter_number_list = []
    iter_number = 0
    for model_devi_job in lmp_data['model_devi_jobs']:
        if model_devi_job['ensemble'] == 'npt':
            iter_number += model_devi_job['temps_divides'] * \
                           model_devi_job['press_divides']
            iter_number_list.append(iter_number)
        elif model_devi_job['ensemble'] == 'nvt':
            iter_number += model_devi_job['temps_divides']
            iter_number_list.append(iter_number)

    ##############################################################
    # General iteration
    # Implement the first iteration in the vasp_iter() function
    # get initial poscar from vasp_sys_dir
    ##############################################################

    # initial index
    iter_index = 0
    need_continue = args.continue_train
    initial_status = ''  # initial checkpoint shouldn't change

    # Check whether have to jump to a loop
    if args.continue_train:
        # load json file
        with open('generator_checkpoint.json') as generate_ckpt:
            ckpt = json.load(generate_ckpt)
            iter_index = ckpt['iter_index']
            initial_status = ckpt['status']

    # ################################################
    # HACK
    # This implement is not so elegent
    # First step of the whole loop
    # ################################################
    iter_dir = f'iter_{iter_index}'
    # Start create iteration dir
    if not os.path.exists(iter_dir):
        os.makedirs(iter_dir)

    if need_continue:
        # status is vasp
        if initial_status == 'vasp':

            # Now process vasp iteration
            vasp_dir = os.path.join(iter_dir, 'vasp')
            if not os.path.exists(vasp_dir):
                os.makedirs(vasp_dir)
            vasp_iter(iter_index, vasp_data, (initial_status == 'vasp'))

        if initial_status == 'vasp' or initial_status == 'deepmd':

            # Now process deepmd iteration
            deepmd_dir = os.path.join(iter_dir, 'deepmd')
            if not os.path.exists(deepmd_dir):
                os.makedirs(deepmd_dir)
            deepmd_iter(iter_index, deepmd_data, (initial_status == 'deepmd'))

        if initial_status == 'vasp' or initial_status == 'deepmd' or \
                initial_status == 'lammps':
            # Now process lammps iteration
            #
            # Fix lmp_iter function
            lmp_dir = os.path.join(iter_dir, 'lammps')
            if not os.path.exists(lmp_dir):
                os.makedirs(lmp_dir)
            #
            # Maybe lmp_iter should have continue mode
            lmp_iter(iter_index, lmp_data, deepmd_data,
                     (initial_status == 'lammps'))

            # Finally continue the loop
            iter_index += 1

    # enter full loop
    while iter_index < iter_number:
        iter_dir = f'iter_{iter_index}'

        # Start create iteration dir
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)

        # Now process vasp iteration
        vasp_dir = os.path.join(iter_dir, 'vasp')
        if not os.path.exists(vasp_dir):
            os.makedirs(vasp_dir)
        vasp_iter(iter_index, vasp_data, False)

        # Now process deepmd iteration
        deepmd_dir = os.path.join(iter_dir, 'deepmd')
        if not os.path.exists(deepmd_dir):
            os.makedirs(deepmd_dir)
        deepmd_iter(iter_index, deepmd_data, False)

        # Now process lammps iteration

        lmp_dir = os.path.join(iter_dir, 'lammps')
        if not os.path.exists(lmp_dir):
            os.makedirs(lmp_dir)
        lmp_iter(iter_index, lmp_data, deepmd_data, False)

        # Finally continue the loop
        iter_index += 1


if __name__ == "__main__":
    generator()
