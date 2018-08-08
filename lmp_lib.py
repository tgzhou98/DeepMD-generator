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
import random
import subprocess
import sys
from itertools import compress
from json import load, dump
from typing import Dict, List

import numpy as np
from ase import io, data

import auxiliary


def lmp_in_generate(model_devi_job_index: int, lmp_config_dir: str, deepmd_data: Dict, model_devi_jobs: List,
                    lmp_shuffled_index):
    """Generate in.deepmd file for lammps

    :param lmp_shuffled_index:
    :param deepmd_data:
    :param lmp_config_dir:
    :param model_devi_jobs:
    :param model_devi_job_index:
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
    run_time = model_devi_jobs[model_devi_job_index]['nsteps']

    # Get corresponding temperature and pressure

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
        temps_group = lmp_shuffled_index / temps_divides
        press_group = lmp_shuffled_index % temps_divides
        temps = temps_lo + (temps_hi - temps_lo) * temps_group / (
                temps_divides - 1)
        press = (press_lo + (press_hi - press_lo) * press_group /
                 (press_divides - 1)) * 10000  # result is bar

        if model_devi_jobs[model_devi_job_index]['press_damp']:
            press_damp = model_devi_jobs[model_devi_job_index]['press_damp']

        # Set fix command
        fix_str = f'fix 1 all npt temp $t $t ${{td}} tri $p $p ${{pd}}'
    elif model_devi_jobs[model_devi_job_index]['ensemble'] == 'nvt':
        temps_group = lmp_shuffled_index % temps_divides
        temps = temps_lo + (temps_hi - temps_lo) * temps_group / (
                temps_divides - 1)
        # Set fix command
        fix_str = f'fix 1 all nvt temp $t $t ${{td}}'
    else:
        print("ensemble not set", file=sys.stderr)
        exit(1)

    # Generate pair_style command
    graph_path_list = [
        os.path.join('..', '..', '..', '..', 'deepmd', f'graph_{graph_numb}',
                     'frozen_model.pb')
        for graph_numb in range(deepmd_data['numb_models'])
    ]
    graph_path_list_str = ' '.join(graph_path_list)
    graph_path_list_str += f" {model_devi_jobs[model_devi_job_index]['trj_freq']} model_devi.out"
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
        dump lmp_config all atom {trj_freq} dump.atom

        # execution
        run 	 {run_time}
        '''

    lmp_in_path = os.path.join(lmp_config_dir, 'in.deepmd')
    with open(lmp_in_path, mode='w') as lmp_in_file:
        lmp_in_file.write(lmp_params_str)


def lmp_pos_generate(poscar_path: str, lmp_config_dir: str, lmp_data: Dict) -> None:
    """Generate lammps pos file from POSCAR

    :param lmp_data:
    :poscar_path: str:
    :lmp_config_dir: str:
    :returns:

    """
    lmp_pos_path = os.path.join(lmp_config_dir, 'data.pos')
    with open(lmp_pos_path, 'w') as lmp_pos_file:
        subprocess.run(
            ['./VASP-poscar2lammps.awk', poscar_path],
            stdout=lmp_pos_file
        )

    # TODO
    # Add mass to lammps pos file
    with open(lmp_pos_path, 'a') as lmp_pos_file:
        lmp_pos_file.write('\nMasses\n\n')
        for atom_order, atom_element in enumerate(lmp_data['element_map']):
            atom_mass = data.atomic_masses[data.atomic_numbers[atom_element]]
            lmp_pos_file.write(f'{atom_order + 1} {atom_mass}\n')


def lmp_run(lmp_config_dir: str, lmp_data: Dict) -> None:
    """Run lammps with command

    :param lmp_data:
    :param lmp_config_dir
    :type lmp_config_dir str
    :lmp_data: Dict:
    :return None

    """
    with auxiliary.cd(lmp_config_dir):
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
    poscar_set_list: List[List[str]] = [auxiliary.get_poscar_files(os.path.join(vasp_dir, set_dir), True)
                                        for set_dir in os.listdir(vasp_dir)
                                        if set_dir.startswith('set')]

    model_devi_jobs_list = lmp_data['model_devi_jobs']

    # lammps continue module
    job_index = 0
    set_index = 0
    lmp_config_index = 0
    config_shuffle_random_list_in_job: List = list()

    if os.path.exists('generator_checkpoint.json'):
        with open('generator_checkpoint.json') as generate_ckpt:
            ckpt = load(generate_ckpt)
            if need_continue:
                if 'config_index' in ckpt:
                    lmp_config_index = ckpt['config_index']
                if 'job_index' in ckpt:
                    job_index = ckpt['job_index']
                if 'set_index' in ckpt:
                    set_index = ckpt['set_index']
                if 'random_shuffle_list' in ckpt:
                    config_shuffle_random_list_in_job = ckpt['random_shuffle_list']

    while job_index < len(model_devi_jobs_list):
        # predefine some job information
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

        # lmp_set_dir_list and lmp_config_number_in_job is the factor of lmp_config_number_in_set
        lmp_config_number_in_set = lmp_config_number_in_job // len(poscar_set_list)
        lmp_config_number_in_job = (lmp_config_number_in_job // len(poscar_set_list)) * len(poscar_set_list)

        # Be careful!!
        # This random list is to provide a random list for shuffle of temps and press
        if not need_continue:
            config_shuffle_random_list_in_job = random.sample(range(lmp_config_number_in_job),
                                                              lmp_config_number_in_job)
        else:
            if set_index == 0 and lmp_config_index == 0:
                config_shuffle_random_list_in_job = random.sample(range(lmp_config_number_in_job),
                                                                  lmp_config_number_in_job)

        while set_index < len(poscar_set_list):
            # generate a random number list in every set
            poscar_list = poscar_set_list[set_index]

            # BUG FIXED
            # Be careful!!
            # This random list is to choose some configuration from previous vasp directories randomly
            random_config_random_list = random.sample(range(len(poscar_list)),
                                                      lmp_config_number_in_set)
            random_choosed_config_list: List[str] = [poscar_list[random_config_index]
                                                     for random_config_index in random_config_random_list]
            # Now create config dir in each set dir
            # generate with random poscar file list
            # DONE

            while lmp_config_index < lmp_config_number_in_set:
                # Different initial atom type and numbers have different set_index
                # configs in one set
                lmp_config_dir = os.path.join(lmp_job_path, f'set_{set_index}', f'config_{lmp_config_index}')
                if not os.path.exists(lmp_config_dir):
                    os.makedirs(lmp_config_dir)

                # determine  random shuffle index
                # BUG
                # Can't correctly continue from the checkpoint
                # Fixed
                # Not save random shuffle list checkpoint file
                lmp_shuffled_index = config_shuffle_random_list_in_job[
                    lmp_config_index + set_index * lmp_config_number_in_set]

                lmp_pos_generate(random_choosed_config_list[lmp_config_index], lmp_config_dir, lmp_data)
                # generate in.deepmd config file
                lmp_in_generate(job_index, lmp_config_dir, deepmd_data, lmp_data['model_devi_jobs'], lmp_shuffled_index)
                # update lammps check point
                lmp_update_checkpoint(iter_index, lmp_config_index, job_index, set_index,
                                      config_shuffle_random_list_in_job)
                # Run lammps command
                lmp_run(lmp_config_dir, lmp_data)
                # choose bad configurations
                lmp_parse_dump2poscar(lmp_config_number_in_job, lmp_config_dir, lmp_data, job_index)

                # Update config index
                lmp_config_index += 1

            # BUG FIXED
            # have to set lmp_config_index to 0 again
            lmp_config_index = 0
            set_index += 1

        # have to set set_index to 0 again
        set_index = 0
        # Update job index
        job_index += 1

    # Finally set job_index to 0
    # NO use


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
    # The length of the bad_config_mask > bad_configs_devi_list and bad_configs_index
    bad_configs_devi_list: List = []
    bad_configs_index_list: List = []
    bad_configs_numbers = 0
    lmp_step_numbers = 0
    # parse model_devi.out file
    model_devi_out_path = os.path.join(lmp_config_dir, 'model_devi.out')
    model_devi_f_trust_lo = lmp_data['model_devi_f_trust_lo']
    model_devi_f_trust_hi = lmp_data['model_devi_f_trust_hi']

    with open(model_devi_out_path, 'r') as model_devi_out_file:
        for line_number, line in enumerate(model_devi_out_file):
            if line_number != 0:
                # Lammps run how many steps
                lmp_step_numbers += 1

                line_field: List = line.split()
                if model_devi_f_trust_lo <= float(line_field[4]) <= model_devi_f_trust_hi:
                    # bad_configs_mask.append(True)
                    bad_configs_devi_list.append(line_field[4])
                    bad_configs_index_list.append(line_number - 1)
                    bad_configs_numbers += 1

    max_choose = lmp_data["model_devi_jobs"][model_devi_job_index]["nchoose"]
    need_choose = max_choose // lmp_config_numbers
    bad_configs_mask = [False] * lmp_step_numbers

    if bad_configs_numbers > need_choose:
        # choose the max choosing configurations
        bad_configs_array = np.array(bad_configs_devi_list)
        # Bug fixed, slicing index is need choose
        max_choose_configs_index = np.argsort(bad_configs_array)[-need_choose:]
        # Choose configs less than the max choosing numbers
        for bad_choosed_index in max_choose_configs_index.tolist():
            # Secondary choose
            bad_configs_index = bad_configs_index_list[bad_choosed_index]
            bad_configs_mask[bad_configs_index] = True
    else:
        # Choose configs more than the max choosing numbers
        for bad_configs_index in bad_configs_index_list:
            bad_configs_mask[bad_configs_index] = True

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
    dump_file_path = os.path.join(lmp_config_dir, 'dump.atom')
    # Read all frames of the lammps dump file
    configs_list: List = io.read(dump_file_path, index=':', format='lammps-dump')

    # Get bad configs
    lmp_bad_configs_mask: List = lmp_get_bad_config_mask(model_devi_job_index, lmp_config_dir, lmp_data,
                                                         lmp_config_numbers)
    # make poscar dir
    generated_poscar_dir = os.path.join(lmp_config_dir, 'generated_poscar')
    if not os.path.exists(generated_poscar_dir):
        os.makedirs(generated_poscar_dir)

    with auxiliary.cd(generated_poscar_dir):
        for bad_config_index, bad_config in enumerate(
                compress(configs_list, lmp_bad_configs_mask)):
            bad_config_dir = f"bad_config_{bad_config_index}"

            if not os.path.exists(bad_config_dir):
                os.makedirs(bad_config_dir)
            # The next block shouldn't be indent!!
            # Fixed
            # Generate POSCAR
            poscar_path = os.path.join(bad_config_dir, 'POSCAR')
            print("write file to :", os.path.abspath(poscar_path), file=sys.stderr)
            io.write(poscar_path, bad_config, format='vasp', direct=True)


def lmp_update_checkpoint(iter_index: int, lmp_config_index: int, job_index: int, set_index: int,
                          config_shuffle_random_list_in_job: List):
    """  Update checkpoint of the lammps part
    :param config_shuffle_random_list_in_job:
    :param set_index:
    :param lmp_config_index:
    :param iter_index:
    :param job_index:
    :returns:

    """
    with open('generator_checkpoint.json', 'r') as generate_ckpt:
        ckpt = load(generate_ckpt)
        ckpt['status'] = 'lammps'
        ckpt['config_index'] = lmp_config_index
        ckpt['job_index'] = job_index
        ckpt['set_index'] = set_index
        ckpt['iter_index'] = iter_index
        ckpt['random_shuffle_list'] = config_shuffle_random_list_in_job

    # Update the json
    with open('generator_checkpoint.json', 'w') as generate_ckpt:
        dump(ckpt, generate_ckpt, indent=2)
