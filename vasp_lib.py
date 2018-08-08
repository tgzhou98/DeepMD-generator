#!/usr/bin/env python3
# #-*-coding:utf-8 -*-
#########################################################################
# File Name   :  param.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-07-20 03:34
# purpose     :
#########################################################################
import os
import shutil
from json import load, dump
from subprocess import run
from time import time
from typing import Dict, List

import numpy as np

import auxiliary


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
    with auxiliary.cd(vasp_config_dir):
        # we are in ~/Library
        vasp_start = time()
        run(
            ["mpirun", "-np", f"{vasp_data['np']}", f"{vasp_data['command']}"])
        vasp_end = time()
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
    # Divide poscar to several sets
    # Determine  how many set are generated

    # Bug fixed
    # Now vasp can detect previous lammps dir and concat sets from different jobs
    job_list: List[str] = [os.path.join(vasp_previous_dir, lmp_job_dir) for lmp_job_dir
                           in os.listdir(vasp_previous_dir)
                           if lmp_job_dir.startswith('job')]
    # Assume that each job has the same set numbers
    poscar_set_list: List[List[str]] = [[] for _ in range(len(job_list[0]))]
    for lmp_job_dir in job_list:
        lmp_set_list_in_job: List[List[str]] = [
            auxiliary.get_poscar_files(os.path.join(lmp_job_dir, set_dir_name), True)
            for set_dir_name in os.listdir(lmp_job_dir) if set_dir_name.startswith('set')
        ]
        poscar_set_list = auxiliary.list_concat(poscar_set_list, lmp_set_list_in_job)

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
            ckpt = load(generate_ckpt)
            if need_continue:
                vasp_config_index = ckpt['config_index']
                vasp_set_index = ckpt['set_index']

    while vasp_set_index < len(poscar_set_list):
        while vasp_config_index < len(poscar_set_list[vasp_set_index]):
            # BugFixed:
            # Need update poscar in the previous dir
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
        dump(ckpt, generate_ckpt, indent=2)
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
        if os.path.exists(outcar_check_path):
            with open(outcar_check_path) as outcar_check_file:
                outcar_check_dict = load(outcar_check_file)

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
        dump(outcar_check_dict, outcar_check_file, indent=2)
