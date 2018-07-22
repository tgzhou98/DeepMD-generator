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
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

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


def parser() -> object:
    """Parser file from command line
    :returns: TODO

    """
    parser = argparse.ArgumentParser(
        description='''Deepmd ab-inito genrator''')

    parser.add_argument(
        'file', type=str, help='Get the json configuration file')

    args = parser.parse_args()
    return args


def uniq(seq: List) -> List:
    # return unique list without changing the order
    # from http://stackoverflow.com/questions/480214
    seen = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


def generator():
    """General iterations
    Implement the first iteration in the vasp_iter() function
    get initial poscar from vasp_sys_dir
    :returns: TODO

    """
    args = parser()

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

    for iter_index in range(iter_number):
        iter_dir = f'iter_{iter_index}'

        # Get job index
        model_devi_job_index = 0
        for iter_accumulates_index in len(iter_number_list):
            if iter_index < iter_number_list[iter_accumulates_index]:
                model_devi_job_index = iter_accumulates_index
        # iteration before this job
        iter_before_this_job = 0
        if model_devi_job_index == 0:
            iter_before_this_job = 0
        else:
            iter_before_this_job = iter_number_list[model_devi_job_index - 1]

        # Start create iteration dir
        if not os.path.exists(iter_dir):
            os.makedirs(iter_dir)

        # Now process vasp iteration
        vasp_dir = os.path.join(iter_dir, 'vasp')
        if not os.path.exists(vasp_dir):
            os.makedirs(iter_dir)
        vasp_iter(iter_index, vasp_data)

        # Now process deepmd iteration
        deepmd_dir = os.path.join(iter_dir, 'deepmd')
        if not os.path.exists(deepmd_dir):
            os.makedirs(deepmd_dir)
        deepmd_iter(iter_index, deepmd_data)

        # Now process lammps iteration
        lmp_dir = os.path.join(iter_dir, 'lammps')
        if not os.path.exists(lmp_dir):
            os.makedirs(lmp_dir)
        lmp_iter(iter_index, model_devi_job_index, iter_before_this_job,
                 lmp_data, deepmd_data)


def vasp_incar_generate(vasp_data: Dict, vasp_config_dir: str):
    """Generate INCAR for vasp

    :vasp_params: Dict: TODO
    :returns: TODO

    """
    vasp_incar_path = os.path.join(vasp_config_dir, 'INCAR')
    with open(vasp_incar_path, 'w') as vasp_incar_file:
        vasp_incar_str = \
            f"""SYSTEM =  H2 # electronic degrees
            PREC  = Accurate                 # chose Low only after tests
            ENCUT = {vasp_data['params']['encut']}
            EDIFF = {vasp_data['params']['ediff']} # do not use default (too large)
            ALGO = Fast
            LREAL = A                      # real space projection
            ISMEAR = -1 ; SIGMA = 0.130    # Fermi smearing: 1500 K 0.086 10-3
            ISYM = 0                       # no symmetry
            ISIF = 0                       # Not need stress tensor
            NCORE = {vasp_data['params']['ncore']}   # default choose 4
            NPAR = {vasp_data['params']['npar']}     # default choose 7
            KPAR = {vasp_data['params']['kpar']}     # default choose 4

            # Add kspacing and kgamma tags
            KSPACING	= {vasp_data['params']['kspacing']}
            KGAMMA	=	.TRUE.

            # (do little writing to save disc space)
            IBRION = -1 ; LCHARG = .FALSE. ; LWAVE = .FALSE.
        """
        vasp_incar_file.write(vasp_incar_str)


def vasp_potcar_generate(vasp_data: Dict, vasp_config_dir: str,
                         vasp_dir: str) -> str:
    """TODO: Docstring for vasp_potcar_generate.

    :vasp_params: Dict: TODO
    :vasp_config_dir: str: TODO
    :returns: TODO

    """
    vasp_potcar_path = os.path.join(vasp_config_dir, 'POTCAR')
    vasp_backup_potcar_path = os.path.join(vasp_dir, 'POTCAR')
    # Check the potcar if is generated
    if not os.path.exists(vasp_backup_potcar_path):
        # Concat POTCAR
        with open(vasp_potcar_path, 'w') as vasp_potcar_file:
            for fname in vasp_data['potential']:
                with open(fname) as infile:
                    for line in infile:
                        vasp_potcar_file.write(line)
    # if the potcar is generated
    shutil.copyfile(vasp_backup_potcar_path, vasp_potcar_path)
    return vasp_backup_potcar_path


def vasp_run(vasp_data: Dict, vasp_config_dir: str):
    """Run vasp file

    :vasp_data: Dict: TODO
    :vasp_config_dir: str: TODO
    :returns: TODO

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


def vasp_iter(iter_index: int, vasp_data: Dict):
    """Do vasp iteration

    :iter_index: int: TODO
    :vasp_data: Dict: TODO
    :returns: TODO

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
        vasp_previous_dir = os.path.join(iter_previous_dir, 'lammps')

    # Get poscar
    poscars = get_poscar_files(vasp_previous_dir, True)

    # Prepare data for vasp
    vasp_backup_potcar_path = ''
    for poscar_index, poscar_file in enumerate(poscars):
        vasp_config_dir = os.path.join(vasp_dir, f'config_{poscar_index}')
        if not os.path.exists(vasp_config_dir):
            os.makedirs(vasp_config_dir)
        # prepare POSCAR
        vasp_poscar_dst_path = os.path.join(vasp_config_dir, 'POSCAR')
        shutil.copyfile(poscar_file, vasp_poscar_dst_path)
        # prepare POTCAR
        vasp_backup_potcar_path = vasp_potcar_generate(
            vasp_data, vasp_config_dir, vasp_dir)
        # prepare INCAR
        vasp_incar_generate(vasp_data, vasp_config_dir)
        # Don't need KPOINTS (have specified kspaceing)
        # Now run vasp
        vasp_run(vasp_data, vasp_config_dir)

    # Remove Backup POTCAR
    os.remove(vasp_backup_potcar_path)


def deepmd_raw_generate(vasp_dir: str, deepmd_dir: str, deepmd_data: Dict):
    """Generate raw data for deepmd

    :vasp_dir: str: TODO
    :deepmd_dir: str: TODO
    :deepmd_data: Dict: TODO
    :returns: TODO

    """
    deepmd_dir_absolute = os.path.abspath(deepmd_dir)
    vasp_dir_absolute = os.path.abspath(vasp_dir)
    test_configs_path_absolute = os.path.join(deepmd_dir_absolute,
                                              'test.configs')
    if not os.path.exists(test_configs_path_absolute):
        with cd(deepmd_dir_absolute):
            # Generate test_configs
            total_configs = cessp2force_lin.param_interface(
                vasp_dir_absolute, True)
            # Generate raw dir
            convert2raw.param_interface(test_configs_path_absolute)
            if not deepmd_data['set_number']:
                set_number = 8
            else:
                set_number = deepmd_data['set_number']
            # Generate set
            set_size: int = 50 * (total_configs // set_number // 50)
            print(f'set size is {set_size}')
            subprocess.run(["./raw_to_set.sh"], set_size)

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

    :deepmd_dir: str: TODO
    :returns: TODO

    """
    with cd(deepmd_dir):
        raw_file_list = [raw for raw in os.listdir('.') if raw.endswith('raw')]
        for raw_file in raw_file_list:
            os.remove(raw_file)
        test_configs = os.path.join(deepmd_dir, 'test.configs')
        os.remove(test_configs)


def deepmd_json_param(deepmd_graph_dir: str, deepmd_data: Dict,
                      iter_index: int):
    """Generate json file for deepmd training

    :deepmd_graph_dir: str: TODO
    :deepmd_data: Dict: TODO
    :returns: TODO

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

    # decide whether restart
    if iter_index == 0:
        deepmd_data['training_params']['restart'] = False
    else:
        deepmd_data['training_params']['restart'] = True

    # set system path
    sets_system_path = os.path.join(deepmd_graph_dir, '..')
    deepmd_data['training_params']['systems'] = sets_system_path

    deepmd_json_path = os.path.join(deepmd_graph_dir, 'deepmd.json')
    with cd(deepmd_graph_dir):
        with open(deepmd_json_path, 'w') as deepmd_json:
            json.dump(deepmd_data['training_params'], deepmd_json)


def deepmd_mv_ckpt(iter_index: int, graph_index: int):
    """TODO: Docstring for deepmd_mv_ckpt.

    :iter_index: int: TODO
    :returns: TODO

    """
    iter_dir = f'iter_{iter_index}'
    # mv when not inital
    if iter_index != 0:
        iter_previous_dir = f'iter_{iter_index - 1}'
        iter_previous_graph_dir = os.path.join(iter_previous_dir, 'deepmd',
                                               f'graph_{graph_index}')
        iter_graph_dir = os.path.join(iter_dir, 'deepmd',
                                      f'graph_{graph_index}')
        for model_ckpt in os.listdir(iter_previous_graph_dir):
            if model_ckpt.startswith('model.ckpt'):
                shutil.move(model_ckpt, iter_graph_dir)


def deepmd_run(deepmd_graph_dir: str, deepmd_data: Dict):
    """Train and freeze the graph in the deepmd_graph_dir

    :deepmd_graph_dir: str: TODO
    :deepmd_data: Dict: TODO
    :returns: TODO

    """
    dp_train_path = os.path.join(deepmd_data['deepmd_bin_path'], 'dp_train')
    dp_frz_path = os.path.join(deepmd_data['deepmd_bin_path'], 'dp_frz')
    print(f'Now start training in the deepmd_graph_dir {deepmd_graph_dir}\n')
    with cd(deepmd_graph_dir):
        deepmd_json_path = os.path.join('.', 'deepmd.json')
        deepmd_train_start_time = time.time()
        # Not set OMP number, use the default
        subprocess.run([dp_train_path, deepmd_json_path])
        deepmd_train_end_time = time.time()
        print(
            f'Traning end, take {deepmd_train_end_time - deepmd_train_start_time}s\n'
        )
        # Start freeze model
        print(
            f'Now start freezing the graph in the deepmd_graph_dir {deepmd_graph_dir}\n'
        )
        subprocess.run([dp_frz_path])
        print(f'Freezing end\n')


def deepmd_iter(iter_index: int, deepmd_data: Dict):
    """Do deepmd iteration

    :iter_index: int: TODO
    :deepmd_data: Dict: TODO
    :returns: TODO

    """
    iter_dir = os.path.join('.', f'iter_{iter_index}')
    vasp_dir = os.path.join(iter_dir, 'vasp')
    deepmd_dir = os.path.join(iter_dir, 'deepmd')
    for graph_index in range(deepmd_data['numb_models']):
        deepmd_graph_dir = os.path.join(deepmd_dir, f'graph_{graph_index}')
        # Prepare set files (generated from vasp)
        deepmd_raw_generate(vasp_dir, deepmd_dir, deepmd_data)
        # Do some cleaning to save disk space
        deepmd_clear_raw_test_configs(deepmd_dir)
        # Generate json
        deepmd_json_param(deepmd_graph_dir, deepmd_data, iter_index)
        # move previous model.ckpt if is not initial
        deepmd_mv_ckpt(iter_index, graph_index)
        # Traning and freezing the model
        deepmd_run(deepmd_graph_dir, deepmd_data)


def lmp_in_generate(iter_index: int, model_devi_job_index: int,
                    iter_before_this_job: int, model_devi_jobs: List,
                    lmp_config_dir: str, deepmd_data: Dict):
    """Generate in.deepmd file for lammps

    :lmp_data: Dict: TODO
    :lmp_config_dir: str: TODO
    :returns: TODO

    """
    # Get corresponding parameters from iter_index
    # Initial some parameters
    fix_str = ''
    temps = 2000  # unit K
    press = 3000000  # unit bar
    temps_damp = 0.1
    press_damp = 1

    # Get corresponding temparature and pressure

    temps_lo = model_devi_jobs[model_devi_job_index]['temps_lo']
    temps_hi = model_devi_jobs[model_devi_job_index]['temps_hi']
    temps_divides = model_devi_jobs[model_devi_job_index]['temps_divides']
    if model_devi_jobs[model_devi_job_index]['temps_damp']:
        temps_damps = model_devi_jobs[model_devi_job_index]['temps_damp']

    # Get the iter_index after minus the iteration before this job
    new_iter_index = iter_index - iter_before_this_job

    if model_devi_jobs[model_devi_job_index]['ensemble'] == 'npt':
        press_lo = model_devi_jobs[model_devi_job_index]['press_lo']
        press_hi = model_devi_jobs[model_devi_job_index]['press_hi']
        press_divides = model_devi_jobs[model_devi_job_index]['press_divides']
        temps_group = new_iter_index // temps_divides
        press_group = new_iter_index % temps_divides
        temps = temps_lo + (temps_hi - temps_lo) * temps_group / (
            temps_divides - 1)
        press = (press_lo + (press_hi - press_lo) * press_group /
                 (press_divides - 1)) * 10000  # result is bar

        if model_devi_jobs[model_devi_job_index]['press_damp']:
            press_damps = model_devi_jobs[model_devi_job_index]['press_damp']

        # Set fix command
        fix_str = f'fix 1 all npt temp $t $t ${{td}} iso $p $p ${{pd}}'
    elif model_devi_jobs[model_devi_job_index]['ensemble'] == 'nvt':
        temps_group = new_iter_index % temps_divides
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
        dump		atom_traj all xyz 10 dump.trajectory

        # execution
        run 	 2000
        '''

    lmp_in_path = os.path.join(lmp_config_dir, 'in.deepmd')
    with open(lmp_in_path, mode='w') as lmp_in_file:
        lmp_in_file.write(lmp_params_str)


def lmp_pos_generate(poscar_path: str, lmp_config_dir: str) -> None:
    """Generate lammps pos file from POSCAR

    :poscar_path: str: TODO
    :lmp_config_dir: str: TODO
    :returns: TODO

    """
    lmp_pos_path = os.path.join(lmp_config_dir, 'data.pos')
    subprocess.run(
        ['./VASP-poscar2lammps.awk', poscar_path, '>', lmp_pos_path])


def lmp_run(lmp_config_dir: str, lmp_data: Dict) -> None:
    """Run lammps with command

    :lmp_config_dir: str: TODO
    :lmp_data: Dict: TODO
    :returns: TODO

    """
    with cd(lmp_config_dir):
        subprocess.run([
            'mpirun', '-np', f"{lmp_data['np']}", f"{lmp_data['command']}",
            '-in', 'in.deepmd'
        ])


def lmp_iter(iter_index: int, model_devi_job_index: int,
             iter_before_this_job: int, lmp_data: Dict,
             deepmd_data: Dict):
    """Generate specific configurations in an iteration

    :iter_index: int: TODO
    :model_devi_job_index: int: TODO
    :iter_before_this_job: int: TODO
    :deepmd_data: Dict: TODO
    :returns: TODO

    """
    # Randomly choose one figuration from the same iteration vasp dir
    iter_dir = f'iter_{iter_index}'
    vasp_dir = os.path.join(iter_dir, 'vasp')
    lmp_dir = os.path.join(iter_dir, 'lammps')
    # lmp_config_dir is just a alias
    lmp_config_dir = lmp_dir
    poscar_list = perp.get_poscar_files(vasp_dir, True)

    # generate a random number
    random_config_index = random.randint(0, len(poscar_list))
    # generate with random pos file
    lmp_pos_generate(poscar_list[random_config_index], lmp_config_dir)
    # generate in.deepmd config file
    lmp_in_generate(iter_index, model_devi_job_index, iter_before_this_job,
                    lmp_data['model_devi_jobs'], lmp_config_dir, deepmd_data)
    # Run lammps command 
    lmp_run(lmp_config_dir, lmp_data)


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
