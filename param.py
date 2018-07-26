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
import random
import shutil
import subprocess
import sys
import time
from itertools import compress
from multiprocessing import Pool
from typing import Dict, List, Set

import numpy as np
from ase import io

import cessp2force_lin
import convert2raw
import perp


class cd:
    """Context manager for changing the current working directory"""

    def __init__(self, newPath):
        """__init__

        :param newPath:
        """
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        """__enter__"""
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        """__exit__

        :param etype:
        :param value:
        :param traceback:
        """
        os.chdir(self.savedPath)


def get_poscar_files(directory: str, recursive: bool) -> List:
    """Get the POSCAR file in the recursive mode

    :directory: 
    :returns: 

    """
    # walk directory (recursively) and return all poscar* files
    # return list of poscar path
    sys.stderr.write(
        'searching directory %s for poscar* files ...\n' % directory)
    poscars = []
    if not recursive:
        for item in os.listdir(directory):
            if item.startswith('POSCAR'):
                poscars.append(os.path.join(directory, item))
    else:
        for root, subfolders, files in os.walk(directory):
            print(files)
            for item in files:
                if item.startswith('POSCAR'):
                    poscars.append(os.path.join(root, item))
    if len(poscars) == 0:
        sys.stderr.write(
            'could not find any poscar files in this directory.\n')
    else:
        sys.stderr.write('found the following files:\n')
        # sys.stderr.write('  {}\n'.format(('\n  ').join(poscars)))
        return poscars
    return poscars


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


def unique(seq: List) -> List:
    # return unique list without changing the order
    # from http://stackoverflow.com/questions/480214
    seen: Set = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


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
            lmp_iter(iter_index, lmp_data, deepmd_data, (initial_status == 'lammps'))

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


def vasp_kpoints_generate(vasp_config_dir: str, vasp_data: Dict):
    """: Docstring for vasp_kpoints_generate.

    :vasp_config_dir: str
    :vasp_data: Dict: 
    :returns: 

    """
    vasp_kpoints_path = os.path.join(vasp_config_dir, 'KPOINTS')
    kmesh: List = vasp_data['params']['kmesh']
    vasp_kpoints_str = \
        f"""System with Gamma grid
            0 0 0
            Gamma
            {kmesh[0]} {kmesh[1]} {kmesh[2]}
            0 0 0
        """

    with open(vasp_kpoints_path, 'w') as vasp_kpoints_file:
        vasp_kpoints_file.write(vasp_kpoints_str)


def vasp_incar_kpoints_generate(vasp_data: Dict, vasp_config_dir: str,
                                vasp_dir: str):
    """Generate INCAR for vasp

    :vasp_params: Dict: 
    :returns: 

    """
    vasp_incar_path = os.path.join(vasp_config_dir, 'INCAR')

    vasp_kspace_str = ''
    if vasp_data['params']["using_kspacing_not_kmesh"]:
        vasp_kspace_str = \
            f"""# Add kspacing and kgamma tags
                KSPACING	= {vasp_data['params']['kspacing']}
                KGAMMA	=	.TRUE.
            """
    else:
        vasp_kpoints_generate(vasp_config_dir, vasp_data)

    vasp_incar_str = \
        f"""SYSTEM =  H2 # electronic degrees
            PREC  = Accurate                 # chose Low only after tests
            ENCUT = {vasp_data['params']['encut']}
            EDIFF = {vasp_data['params']['ediff']} # default is large
            ALGO = Fast
            LREAL = {vasp_data['params']['lreal']} # real space projection
            ISMEAR = -1 ; SIGMA = 0.130    # Fermi smearing: 1500 K 0.086 10-3
            ISYM = 0                       # no symmetry
            ISIF = 0                       # Not need stress tensor
            NCORE = {vasp_data['params']['ncore']}   # default choose 4
            NPAR = {vasp_data['params']['npar']}     # default choose 7
            KPAR = {vasp_data['params']['kpar']}     # default choose 4

        {vasp_kspace_str}

            # (do little writing to save disc space)
            IBRION = -1 ; LCHARG = .FALSE. ; LWAVE = .FALSE.
        """

    with open(vasp_incar_path, 'w') as vasp_incar_file:
        vasp_incar_file.write(vasp_incar_str)


def vasp_potcar_generate(vasp_data: Dict, vasp_config_dir: str, vasp_dir: str):
    """: Docstring for vasp_potcar_generate.

    :vasp_params: Dict: 
    :vasp_config_dir: str: 
    :returns: 

    """
    vasp_potcar_path = os.path.join(vasp_config_dir, 'POTCAR')
    vasp_backup_potcar_path = os.path.join(vasp_dir, 'POTCAR')
    # Check the potcar if is generated
    if not os.path.exists(vasp_backup_potcar_path):
        # Concat POTCAR
        with open(vasp_backup_potcar_path, 'w') as vasp_potcar_file:
            for fname in vasp_data['potential']:
                with open(fname) as infile:
                    for line in infile:
                        vasp_potcar_file.write(line)
    # print(vasp_backup_potcar_path)
    # if the potcar is generated
    shutil.copyfile(vasp_backup_potcar_path, vasp_potcar_path)


def vasp_run(vasp_data: Dict, vasp_config_dir: str):
    """Run vasp file

    :vasp_data: Dict: 
    :vasp_config_dir: str: 
    :returns: 

    """
    with cd(vasp_config_dir):
        # we are in ~/Library
        vasp_start = time.time()
        subprocess.run(
            ["mpirun", "-np", f"{vasp_data['np']}", f"{vasp_data['command']}"])
        vasp_end = time.time()
        print(f"""vasp calculation in {vasp_config_dir} has been done
                   calculation time is {vasp_end-vasp_start}s

                   """)


def vasp_iter(iter_index: int, vasp_data: Dict, need_continue: bool):
    """Do vasp iteration

    :iter_index: int: 
    :vasp_data: Dict: 
    :returns: 

    """
    iter_dir = f'iter_{iter_index}'
    vasp_dir = os.path.join(iter_dir, 'vasp')
    vasp_previous_dir = ''

    # iter_index is 0 (initial condition)
    if iter_index == 0:
        vasp_previous_dir = vasp_data['sys_dir']
    # General iter_index
    # ATTENTION:
    # VASP poscar is in lammps dir
    else:
        iter_previous_dir = f'iter_{iter_index - 1}'
        vasp_previous_dir = os.path.join(iter_previous_dir, 'lammps',
                                         'generated_poscar')

    # Get poscar
    # Divide poscar to several sets
    # Determinal how many set are generated
    # Bug fixed poscar_set is a file name instead of path
    poscar_set_list = [
        get_poscar_files(os.path.join(vasp_previous_dir, poscar_set), True)
        for poscar_set in os.listdir(vasp_previous_dir)
    ]
    # for poscar_set in os.listdir(vasp_previous_dir):
    #     print(poscar_set)
    # print("poscar_set_list")
    # print(poscar_set_list)
    # print("vasp_previous_dir")
    # print(vasp_previous_dir)

    # Prepare data for vasp
    # vasp_backup_potcar_path = ''

    vasp_set_index = 0
    vasp_config_index = 0
    if os.path.exists('generator_checkpoint.json'):
        with open('generator_checkpoint.json') as generate_ckpt:
            ckpt = json.load(generate_ckpt)
            if need_continue:
                vasp_config_index = ckpt['config_index']
                vasp_set_index = ckpt['set_index']

    while vasp_set_index < len(poscar_set_list):
        while vasp_config_index < len(poscar_set_list[vasp_set_index]):
            # BugFixed:
            # Need updata poscar in the previous dir
            poscar_file = poscar_set_list[vasp_set_index][vasp_config_index]
            vasp_set_dir = os.path.join(vasp_dir, f'set_{vasp_set_index}')
            if not os.path.exists(vasp_set_dir):
                os.makedirs(vasp_set_dir)
            vasp_config_dir = os.path.join(vasp_set_dir,
                                           f'config_{vasp_config_index}')
            if not os.path.exists(vasp_config_dir):
                os.makedirs(vasp_config_dir)
            # prepare POSCAR
            vasp_poscar_dst_path = os.path.join(vasp_config_dir, 'POSCAR')
            shutil.copyfile(poscar_file, vasp_poscar_dst_path)
            # prepare POTCAR
            vasp_potcar_generate(vasp_data, vasp_config_dir, vasp_dir)
            # prepare INCAR
            vasp_incar_kpoints_generate(vasp_data, vasp_config_dir, vasp_dir)
            # Don't need KPOINTS (have specified kspaceing)
            # Now run vasp

            # Update checkpoint
            vasp_update_checkpoint(vasp_set_index, vasp_config_index,
                                   iter_index)

            vasp_run(vasp_data, vasp_config_dir)

            # Update outcar status
            vasp_outcar_check(vasp_set_index, vasp_config_index, iter_index,
                              vasp_data)

            # Finally update the poscar
            vasp_config_index += 1

        # print('vasp_set_index', vasp_set_index)
        # print('vasp_config_index', vasp_config_index)
        # print(len(poscar_set_list))
        vasp_set_index += 1
        # Bug Fixed
        # Manually set vasp_config_index to 0
        vasp_config_index = 0


def vasp_update_checkpoint(vasp_set_index, vasp_config_index: int,
                           iter_index: int):
    """: Docstring for vasp_update_checkpoint.
    :returns: 

    """
    ckpt: Dict = dict()
    # if iter_index == 0 and vasp_config_index == 0:
    with open('generator_checkpoint.json', 'w') as generate_ckpt:
        ckpt['status'] = 'vasp'
        ckpt['config_index'] = vasp_config_index
        ckpt['set_index'] = vasp_set_index
        ckpt['iter_index'] = iter_index
        json.dump(ckpt, generate_ckpt, indent=2)
    # else:
    #     with open('generator_checkpoint.json', 'r') as generate_ckpt:
    #         ckpt = json.load(generate_ckpt)
    #         ckpt['status'] = 'vasp'
    #         ckpt['config_index'] = vasp_config_index
    #         ckpt['iter_index'] = iter_index

    #     # Write to the same file
    #     with open('generator_checkpoint.json', 'w') as generate_ckpt:
    #         json.dump(ckpt, generate_ckpt)


def vasp_outcar_check(vasp_set_index: int, vasp_config_index: int,
                      iter_index: int, vasp_data: Dict):
    """Check outcar of every configuration and save to log

    :vasp_config_index: int: 
    :iter_index: int: 
    :returns: 

    """
    iter_dir = f'iter_{iter_index}'
    vasp_dir = os.path.join(iter_dir, 'vasp')
    vasp_config_dir = os.path.join(vasp_dir, f'set_{vasp_set_index}',
                                   f'config_{vasp_config_index}')
    outcar_path = os.path.join(vasp_config_dir, 'OUTCAR')
    outcar_check_path = 'outcar_check.json'
    vasp_outcar_trust = vasp_data['outcar_trust']
    vasp_outcar_devi_trust = vasp_data['outcar_devi_trust']

    outcar_check_dict: Dict = dict()
    # Pay atteontion to the condition !!!!!!
    # Not read file at the first step
    if not (iter_index == 0 and vasp_set_index == 0
            and vasp_config_index == 0):
        with open(outcar_check_path) as outcar_check_file:
            outcar_check_dict = json.load(outcar_check_file)

    set_dict: Dict = dict()
    drift_dict: Dict = dict()

    if f'iter_{iter_index}' in outcar_check_dict:
        set_dict = outcar_check_dict[f'iter_{iter_index}']
    # print(set_dict)

    if f'set_{vasp_set_index}' in set_dict:
        drift_dict = set_dict[f'set_{vasp_set_index}']
    # print(drift_dict)

    # Parse OUTCAR
    with open(outcar_path) as outcar_file:
        for line in outcar_file:
            if line.lstrip().startswith('POSITION'):
                force_x: List = []  # noqa
                force_y: List = []  # noqa
                force_z: List = []  # noqa
                outcar_file.readline()
                line = outcar_file.readline()
                while not line.lstrip().startswith('-----------'):
                    pos_for_field = line.split()
                    # print(pos_for_field)
                    force_x.append(float(pos_for_field[3]))
                    force_y.append(float(pos_for_field[4]))
                    force_z.append(float(pos_for_field[5]))
                    line = outcar_file.readline()

                # Start calculate standard deviation
                force_x_devi = float(np.std(force_x))
                force_y_devi = float(np.std(force_y))
                force_z_devi = float(np.std(force_z))

                drift_devi_flag = \
                    force_x_devi < vasp_outcar_devi_trust and \
                    force_y_devi < vasp_outcar_devi_trust and \
                    force_z_devi < vasp_outcar_devi_trust

                line = outcar_file.readline()

                # Start calculate total drift
                total_dirft = line.split()
                drift_flag = \
                    abs(float(total_dirft[2])) < vasp_outcar_trust \
                    and abs(float(total_dirft[3])) < vasp_outcar_trust \
                    and abs(float(total_dirft[4])) < vasp_outcar_trust

                config_dict = dict()
                config_dict['drift'] = drift_flag
                config_dict['drift_data'] = [
                    float(total_dirft[2]),
                    float(total_dirft[3]),
                    float(total_dirft[4])
                ]
                config_dict['devi_force'] = drift_devi_flag
                config_dict['devi_force_data'] = [
                    force_x_devi, force_y_devi, force_z_devi
                ]
                drift_dict[f'config_{vasp_config_index}'] = config_dict

    # print(drift_dict)
    # Finally Update dict
    # 
    # HAVE BUG
    set_dict[f'set_{vasp_set_index}'] = drift_dict
    outcar_check_dict[f'iter_{iter_index}'] = set_dict

    # Update json file
    with open(outcar_check_path, 'w') as outcar_check_file:
        json.dump(outcar_check_dict, outcar_check_file, indent=2)


def deepmd_raw_generate(vasp_dir: str, deepmd_dir: str, deepmd_data: Dict):
    """Generate raw data for deepmd

    :vasp_dir: str: 
    :deepmd_dir: str: 
    :deepmd_data: Dict: 
    :returns: 

    """
    vasp_set_list = [
        os.path.join(vasp_dir, vasp_set_dir_name)
        for vasp_set_dir_name in os.listdir(vasp_dir)
        if vasp_set_dir_name.startswith('set')
    ]
    vasp_set_list_absolute = [
        os.path.abspath(vasp_set) for vasp_set in vasp_set_list
    ]
    deepmd_set_dir_list = [
        os.path.join(deepmd_dir, 'data', f'deepmd_set_{set_index}')
        for set_index in range(len(vasp_set_list))
    ]
    deepmd_set_dir_list_absolute = [
        os.path.abspath(deepmd_set) for deepmd_set in deepmd_set_dir_list
    ]

    # print(test_configs_path_absolute)
    # print(os.path.exists(test_configs_path_absolute))
    # 
    # HACK multiprocess never done
    # process = Pool(8)

    for set_index, deepmd_set_absolute in enumerate(
            deepmd_set_dir_list_absolute):
        if not os.path.exists(deepmd_set_absolute):
            os.makedirs(deepmd_set_absolute)
        with cd(deepmd_set_absolute):
            # Generate test_configs
            total_configs = cessp2force_lin.param_interface(
                vasp_set_list_absolute[set_index], True)
            # Generate raw dir
            test_configs_absolute = os.path.abspath('test.configs')
            convert2raw.param_interface(test_configs_absolute)
            print('generate_raw')
            if not deepmd_data['set_number']:
                set_number = 8
            else:
                set_number = deepmd_data['set_number']
            # Generate set
            set_size: int = 50 * (total_configs // set_number // 50)
            print(f'set size is {set_size}')
            # # Remove set
            # print("remove set")
            for set_dir in os.listdir('.'):
                if set_dir.startswith('set') and os.path.isdir(set_dir):
                    shutil.rmtree(set_dir)
            code = subprocess.run(["../../../../raw_to_set.sh", f"{set_size}"])
            print(f'return code {code}')

    # Don't need copy set file, can specified in the json file
    # # Copy set directory to correponding deepmd_graph_dir
    # set_dir_lists = [set_dir for set_dir
    #                  in os.listdir(deepmd_dir)
    #                  if set_dir.startswith('set')]
    # for set_dir in set_dir_lists:
    #     dirname = Path(set_dir).name
    #     deepmd_graph_dir_set = os.path.join(deepmd_graph_dir, dirname)
    #     copytree(set_dir, deepmd_graph_dir_set, symlinks=False, ignore=None)


def deepmd_clear_raw_test_configs(deepmd_dir: str):
    """Remove extra test_configs and raw file

    :deepmd_dir: str: 
    :returns: 

    """
    with cd(deepmd_dir):
        raw_file_list = [raw for raw in os.listdir('.') if raw.endswith('raw')]
        for raw_file in raw_file_list:
            os.remove(raw_file)
        test_configs = 'test.configs'
        os.remove(test_configs)


def deepmd_json_param(deepmd_graph_dir: str, deepmd_data: Dict,
                      iter_index: int):
    """Generate json file for deepmd training

    :deepmd_graph_dir: str: 
    :deepmd_data: Dict: 
    :returns: 

    """
    # Specify more parameter option from json file

    # Generate a random number as a random seed
    deepmd_data['training_params']['seed'] = random.randint(0, 2147483647)
    # Change batch size
    if iter_index == 0:
        deepmd_data['training_params']['batch_size'] = deepmd_data[
            'init_batch_size']
    else:
        deepmd_data['training_params']['batch_size'] = deepmd_data[
            'sys_batch_size']

    # Deepmd version update, not set restart in json file but set in command line
    # # decide whether restart
    # if iter_index == 0:
    #     deepmd_data['training_params']['restart'] = False
    # else:
    #     deepmd_data['training_params']['restart'] = True

    # set system path
    # Bug Fixed
    # Now use relative path
    with cd(deepmd_graph_dir):
        sets_system_relative_path = os.path.join('..', 'data')
        sets_system_list = [
            os.path.join(sets_system_relative_path, deepmd_set_dir)
            for deepmd_set_dir in os.listdir(sets_system_relative_path)
        ]
    deepmd_data['training_params']['systems'] = sets_system_list

    deepmd_json_path = os.path.join(deepmd_graph_dir, 'deepmd.json')

    # Create if not have graph dir
    if not os.path.exists(deepmd_graph_dir):
        os.makedirs(deepmd_graph_dir)

    with open(deepmd_json_path, 'w') as deepmd_json:
        json.dump(deepmd_data['training_params'], deepmd_json, indent=2)


def deepmd_mv_ckpt(iter_index: int, graph_index: int):
    """: Docstring for deepmd_mv_ckpt.

    :iter_index: int: 
    :returns: 

    """
    iter_dir = f'iter_{iter_index}'
    # mv when not initial
    if iter_index != 0:
        iter_previous_dir = f'iter_{iter_index - 1}'
        iter_previous_graph_dir = os.path.join(iter_previous_dir, 'deepmd',
                                               f'graph_{graph_index}')
        iter_graph_dir = os.path.join(iter_dir, 'deepmd',
                                      f'graph_{graph_index}')
        for model_ckpt in os.listdir(iter_previous_graph_dir):
            if model_ckpt.startswith('model.ckpt'):
                shutil.move(model_ckpt, iter_graph_dir)


def deepmd_run(iter_index: int, deepmd_graph_dir: str, deepmd_data: Dict,
               need_continue: bool):
    """Train and freeze the graph in the deepmd_graph_dir

    :deepmd_graph_dir: str: 
    :deepmd_data: Dict: 
    :returns: 

    """
    dp_train_path = os.path.join(deepmd_data['deepmd_bin_path'], 'dp_train')
    dp_frz_path = os.path.join(deepmd_data['deepmd_bin_path'], 'dp_frz')
    print(f'Now start training in the deepmd_graph_dir {deepmd_graph_dir}\n')
    with cd(deepmd_graph_dir):
        deepmd_json_path = os.path.join('.', 'deepmd.json')
        # Not set OMP number, use the default

        print("enter_traina_dir", file=sys.stderr)
        print("need_continue_run", need_continue, file=sys.stderr)
        # Check if restart
        if not need_continue:
            if iter_index == 0:
                subprocess.run([dp_train_path, deepmd_json_path])
            else:
                subprocess.run([
                    dp_train_path, deepmd_json_path, '--initial', 'model.ckpt'
                ])
        else:
            subprocess.run(
                [dp_train_path, deepmd_json_path, '--restart', 'model.ckpt'])
        # Start freeze model
        print(f'Now start freezing the graph in the {deepmd_graph_dir}\n')
        subprocess.run([dp_frz_path])
        print(f'Freezing end\n')


def deepmd_iter(iter_index: int, deepmd_data: Dict, need_continue: bool):
    """Do deepmd iteration

    :iter_index: int: 
    :deepmd_data: Dict: 
    :returns: 

    """
    iter_dir = os.path.join('.', f'iter_{iter_index}')
    vasp_dir = os.path.join(iter_dir, 'vasp')
    deepmd_dir = os.path.join(iter_dir, 'deepmd')

    # 
    # DONE
    # Not need continue function
    # use multiprocess instead

    # Prepare set files (generated from vasp)
    deepmd_raw_generate(vasp_dir, deepmd_dir, deepmd_data)

    process = Pool(deepmd_data['numb_models'])
    for graph_index in range(deepmd_data['numb_models']):
        deepmd_graph_dir = os.path.join(deepmd_dir, f'graph_{graph_index}')
        if not os.path.exists(deepmd_graph_dir):
            os.makedirs(deepmd_graph_dir)
        print("need_continue", need_continue)
        if need_continue:
            process.apply_async(
                deepmd_single_process_continue_iter,
                args=(graph_index, deepmd_graph_dir, deepmd_data, iter_index,
                      need_continue))
        else:
            process.apply_async(
                deepmd_single_process_initial_iter,
                args=(graph_index, deepmd_graph_dir, deepmd_data, iter_index,
                      need_continue))
    process.close()
    process.join()
    print('All subprocess done')

    # # Do some cleaning to save disk space
    # deepmd_clear_raw_test_configs(deepmd_dir)


def deepmd_single_process_initial_iter(graph_index: int, deepmd_graph_dir: str,
                                       deepmd_data: Dict, iter_index: int,
                                       need_continue: bool):
    """auxiliary function to do single process deepmd

    :graph_index: int: 
    :deepmd_graph_dir: str: 
    :deepmd_data: Dict: 
    :iter_index: int: 
    :need_continue: bool: 
    :returns: 

    """
    # Generate json
    deepmd_json_param(deepmd_graph_dir, deepmd_data, iter_index)
    # move previous model.ckpt if is not initial
    deepmd_mv_ckpt(iter_index, graph_index)
    # update deepmd check point
    deepmd_update_checkpoint(iter_index)
    # Traning and freezing the model
    deepmd_run(iter_index, deepmd_graph_dir, deepmd_data, need_continue)


def deepmd_single_process_continue_iter(deepmd_graph_dir: str,
                                        deepmd_data: Dict,
                                        iter_index: int,
                                        need_continue: bool):
    """deepmd_single_process function for continue mode

    :param need_continue:
    :param deepmd_graph_dir:
    :type deepmd_graph_dir: str
    :param deepmd_data:
    :type deepmd_data: Dict
    :param iter_index:
    :type iter_index: int
    """
    # Traning and freezing the model
    deepmd_run(iter_index, deepmd_graph_dir, deepmd_data, need_continue)


def deepmd_update_checkpoint(iter_index: int):
    """: Docstring for deepmd_update_checkpoint.
    :returns: 

    """
    ckpt: Dict = dict()
    with open('generator_checkpoint.json', 'r') as generate_ckpt:
        ckpt = json.load(generate_ckpt)
        ckpt['status'] = 'deepmd'
        ckpt['config_index'] = -1  # multiprocessing don't need graph_index
        ckpt['set_index'] = -1
        ckpt['iter_index'] = iter_index

    # Dump to the same file and erase the former
    with open('generator_checkpoint.json', 'w') as generate_ckpt:
        json.dump(ckpt, generate_ckpt, indent=2)


def lmp_in_generate(lmp_config_index: int,
                    model_devi_job_index: int,
                    model_devi_jobs: List,
                    lmp_config_dir: str,
                    deepmd_data: Dict):
    """Generate in.deepmd file for lammps

    :lmp_data: Dict: 
    :lmp_config_dir: str: 
    :returns: 

    """
    # Get corresponding parameters from iter_index
    # Initial some parameters
    fix_str = ''
    temps = 2000  # unit K
    press = 3000000  # unit bar
    temps_damp = 0.1
    press_damp = 1
    trj_freq = model_devi_jobs[model_devi_job_index]['trj_freq']

    # Get corresponding temparature and pressure

    temps_lo = model_devi_jobs[model_devi_job_index]['temps_lo']
    temps_hi = model_devi_jobs[model_devi_job_index]['temps_hi']
    temps_divides = model_devi_jobs[model_devi_job_index]['temps_divides']
    if model_devi_jobs[model_devi_job_index]['temps_damp']:
        temps_damp = model_devi_jobs[model_devi_job_index]['temps_damp']

    # Get the iter_index after minus the iteration before this job

    if model_devi_jobs[model_devi_job_index]['ensemble'] == 'npt':
        # Get pressure parameters
        press_lo = model_devi_jobs[model_devi_job_index]['press_lo']
        press_hi = model_devi_jobs[model_devi_job_index]['press_hi']
        press_divides = model_devi_jobs[model_devi_job_index]['press_divides']

        # config_index in a job
        temps_group = lmp_config_index / temps_divides
        press_group = lmp_config_index % temps_divides
        temps = temps_lo + (temps_hi - temps_lo) * temps_group / (
                temps_divides - 1)
        press = (press_lo + (press_hi - press_lo) * press_group /
                 (press_divides - 1)) * 10000  # result is bar

        if model_devi_jobs[model_devi_job_index]['press_damp']:
            press_damp = model_devi_jobs[model_devi_job_index]['press_damp']

        # Set fix command
        fix_str = f'fix 1 all npt temp $t $t ${{td}} tri $p $p ${{pd}}'
    elif model_devi_jobs[model_devi_job_index]['ensemble'] == 'nvt':
        temps_group = lmp_config_index % temps_divides
        temps = temps_lo + (temps_hi - temps_lo) * temps_group / (
                temps_divides - 1)
        # Set fix command
        fix_str = f'fix 1 all nvt temp $t $t ${{td}}'
    else:
        print("ensemble not set", file=sys.stderr)
        exit(1)

    # Generate pair_style command
    graph_path_list = [
        os.path.join('..', '..', 'deepmd', f'graph_{graph_numb}',
                     'frozen_model.pb')
        for graph_numb in range(deepmd_data['numb_models'])
    ]
    graph_path_list_str = ' '.join(graph_path_list)
    pair_style_str = f'pair_style deepmd {graph_path_list_str}'

    lmp_params_str = \
        f'''
        # 3D copper block simulation
        boundary    p p p
        units       metal
        atom_style  atomic

        # geometry
        box tilt large
        read_data	data.pos

        # deepmd potential

        {pair_style_str}
        pair_coeff

        neighbor       2 bin
        neigh_modify   delay 0

        #Langevin random seed
        variable r equal 57085
        variable t equal {temps}
        variable p equal {press}
        variable td equal {temps_damp}
        variable pd equal {press_damp}

        # initialize
        velocity all create $t 28459 rot yes dist gaussian mom yes
        reset_timestep 0

        # fixes
        {fix_str}

        timestep 0.001

        # output
        thermo_style  custom step temp press
        thermo  10

        # trajectory
        dump		atom_traj all xyz {trj_freq} dump.trajectory

        # execution
        run 	 2000
        '''

    lmp_in_path = os.path.join(lmp_config_dir, 'in.deepmd')
    with open(lmp_in_path, mode='w') as lmp_in_file:
        lmp_in_file.write(lmp_params_str)


def lmp_pos_generate(poscar_path: str, lmp_config_dir: str) -> None:
    """Generate lammps pos file from POSCAR

    :poscar_path: str: 
    :lmp_config_dir: str: 
    :returns: 

    """
    lmp_pos_path = os.path.join(lmp_config_dir, 'data.pos')
    subprocess.run(
        ['./VASP-poscar2lammps.awk', poscar_path, '>', lmp_pos_path])


def lmp_run(lmp_config_dir: str, lmp_data: Dict) -> None:
    """Run lammps with command

    :lmp_config_dir: str: 
    :lmp_data: Dict: 
    :returns: 

    """
    with cd(lmp_config_dir):
        subprocess.run([
            'mpirun', '-np', f"{lmp_data['np']}", f"{lmp_data['command']}",
            '-in', 'in.deepmd'
        ])


def lmp_iter(iter_index: int, lmp_data: Dict, deepmd_data: Dict, need_continue: bool):
    """Generate specific configurations in an iteration

    :param deepmd_data:
    :param lmp_data:
    :param iter_index:
    :param need_continue:
    :returns:

    """
    # Randomly choose one figuration from the same iteration vasp dir
    iter_dir: str = f'iter_{iter_index}'
    vasp_dir = os.path.join(iter_dir, 'vasp')
    lmp_dir = os.path.join(iter_dir, 'lammps')
    # lmp_config_dir is just a alias
    poscar_list = perp.get_poscar_files(vasp_dir, True)

    model_devi_jobs_list = lmp_data['model_devi_jobs']

    # lammps continue module
    job_index = 0
    lmp_config_index = 0
    if os.path.exists('generator_checkpoint.json'):
        with open('generator_checkpoint.json') as generate_ckpt:
            ckpt = json.load(generate_ckpt)
            if need_continue:
                lmp_config_index = ckpt['config_index']
                job_index = ckpt['job_index']

    while job_index < len(model_devi_jobs_list):
        model_devi_job = model_devi_jobs_list[job_index]
        # Create job dir
        lmp_job_path = os.path.join(lmp_dir, f'job_{job_index}')
        if not os.path.exists(lmp_job_path):
            os.makedirs(lmp_job_path)

        lmp_config_number_in_job = model_devi_job['temps_divides']
        if model_devi_job['ensemble'] == 'npt':
            # Get pressure parameters
            press_divides = model_devi_job['press_divides']
            lmp_config_number_in_job *= press_divides

        # generate a random number list in every job
        random_config_randomlist = random.sample(range(len(poscar_list)),
                                                 lmp_config_number_in_job)
        random_choosed_config_list = [poscar_list[random_config_index]
                                      for random_config_index in random_config_randomlist]
        # Now create config dir in each job dir
        # generate with random poscar file list
        # DONE
        while lmp_config_index < lmp_config_number_in_job:
            lmp_config_dir = os.path.join(lmp_job_path, f'config_{lmp_config_index}')
            if not os.path.exists(lmp_config_dir):
                os.makedirs(lmp_config_dir)

            lmp_pos_generate(random_choosed_config_list[lmp_config_index], lmp_config_dir)
            # generate in.deepmd config file
            lmp_in_generate(lmp_config_index, job_index,
                            lmp_data['model_devi_jobs'], lmp_config_dir, deepmd_data)
            # update lammps check point
            lmp_update_checkpoint(iter_index, lmp_config_index, job_index)
            # Run lammps command
            lmp_run(lmp_config_dir, lmp_data)
            # choose bad configurations
            lmp_parse_dump2poscar(lmp_config_number_in_job, lmp_config_dir, lmp_data, job_index)

            # Update config index
            lmp_config_index += 1

        # Update job index
        job_index += 1


def lmp_get_bad_config_mask(model_devi_job_index: int,
                            lmp_config_dir: str,
                            lmp_data: Dict,
                            lmp_config_numbers: int) -> List:
    """Get the bad config list from model_deviation

    :param lmp_data:
    :param lmp_config_dir:
    :param model_devi_job_index:
    :param lmp_config_numbers:
    :returns:   len(bad_configs) = config_numbers
                bad_configs is a ***BOOL*** mask determine which to choose
    """
    bad_configs_mask: List = []
    bad_configs_list: List = []
    bad_configs_numbers = 0
    # parse model_devi.out file
    model_devi_out_path = os.path.join(lmp_config_dir, 'model_devi.out')
    model_devi_f_trust_lo = lmp_data['model_devi_f_trust_lo']
    model_devi_f_trust_hi = lmp_data['model_devi_f_trust_hi']

    with open(model_devi_out_path, 'r') as model_devi_out_file:
        for line_number, line in enumerate(model_devi_out_file):
            if line_number != 0:
                line_field: List = line.split(line)
                if float(line_field[5]) < model_devi_f_trust_lo:
                    bad_configs_mask.append(False)
                elif float(line_field[5]) >= model_devi_f_trust_lo and \
                        float(line_field[5]) <= model_devi_f_trust_hi:
                    bad_configs_mask.append(True)
                    bad_configs_list.append(line_field[5])
                    bad_configs_numbers += 1
                else:
                    bad_configs_mask.append(False)

    max_choose = lmp_data["model_devi_jobs"][model_devi_job_index]["nchoose"]
    need_choose = max_choose // lmp_config_numbers
    if bad_configs_numbers > need_choose:
        # choose the max choosing configurations
        bad_configs_array = np.array(bad_configs_list)
        max_choose_configs_index = np.argsort(bad_configs_array)[-max_choose:]
        # Choose configs less than the max choosing numbers
        choosed_bad_configs_mask = np.array(bad_configs_mask)[
            max_choose_configs_index]
        bad_configs_mask = choosed_bad_configs_mask.tolist()

        # The else is not so frequently
        # Not so frequently

    return bad_configs_mask


def lmp_parse_dump2poscar(lmp_config_numbers: int,
                          lmp_config_dir: str,
                          lmp_data: Dict,
                          model_devi_job_index: int):
    """Change the dump file to the xyz file

    :param model_devi_job_index:
    :param lmp_config_numbers:
    :param lmp_data To get lammps parameters
    :param lmp_config_dir:
    :returns:

    """
    dump_file_path = os.path.join(lmp_config_dir, 'dump.trajectory')
    dump_xyz_path = os.path.join(lmp_config_dir, 'dump.xyz')
    with open(dump_file_path, 'r') as dump_file:
        with open(dump_xyz_path, 'w') as dump_xyz_file:
            for line in dump_file:
                line_field = line.split()
                if len(line_field) == 4:
                    element_type: str = lmp_data['element_map'][int(
                        line_field[0])]
                    line_field[0] = element_type
                dump_xyz_file.write(' '.join(line_field))
                line = ' '.join(line_field) + '\n'
                dump_xyz_file.write(line)

    # Read xyz file

    # Read all frames of the lammps dump file
    configs_list: List = io.read(dump_xyz_path, index=':', format='xyz')

    # Get bad configs
    lmp_bad_configs_mask: List = lmp_get_bad_config_mask(model_devi_job_index, lmp_config_dir, lmp_data,
                                                         lmp_config_numbers)
    # make poscar dir
    generated_poscar_dir = os.path.join(lmp_config_dir, 'generated_poscar')
    if not os.path.exists(generated_poscar_dir):
        os.makedirs(generated_poscar_dir)

    for bad_config_index, bad_config in enumerate(
            compress(configs_list, lmp_bad_configs_mask)):
        bad_config_dir = f"bad_config_{bad_config_index}"
        if not os.path.exists(bad_config_dir):
            os.makedirs(bad_config_dir)
            # Generate POSCAR
            poscar_path = os.path.join(bad_config_dir, 'POSCAR')
            io.write(poscar_path, bad_config, format='vasp')


def lmp_update_checkpoint(iter_index: int, lmp_config_index: int, job_index: int):
    """  Update checkpoint of the lammps part
    :param lmp_config_index:
    :param iter_index:
    :param job_index:
    :returns: 

    """
    with open('generator_checkpoint.json', 'r') as generate_ckpt:
        ckpt = json.load(generate_ckpt)
        ckpt['status'] = 'lammps'
        ckpt['config_index'] = lmp_config_index
        ckpt['job_index'] = job_index
        ckpt['iter_index'] = iter_index

    # Update the json
    with open('generator_checkpoint.json', 'w') as generate_ckpt:
        json.dump(ckpt, generate_ckpt, indent=2)


def copytree(src, dst, symlinks=False, ignore=None):
    """update of the shutil coptree function

    """
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


if __name__ == "__main__":
    generator()
