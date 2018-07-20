#!/usr/bin/env python3
##-*-coding:utf-8 -*-
#########################################################################
# File Name   :  param.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn 
# created at  :  2018-07-20 03:34
# purpose     :  
#########################################################################


import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import os
from shutil import copyfile
from typing import List, Dict


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
    """Do many iterations
    :returns: TODO

    """
    args = Parser()

    with open(args.file, "r") as json_file:
        data = json.load(json_file)


    number_iter = data['numb_iter']

    ##############################################################
    # Inital Condition
    # Do the first iteration
    # get initial poscar from vasp_sys_dir
    ##############################################################
    poscars = get_poscar_files(data['vasp_sys_dir'], True)
    iter_dir = f'iter_0'
    if not os.path.exists(iter_dir):
        os.makedirs(iter_dir)

    vasp_dir = os.path.join(iter_dir, 'vasp')
    if not os.path.exist(vasp_dir):
        os.makedirs(vasp_dir)
    # Prepare data for vasp
    for poscar_index, poscar_file in enumerate(poscars):
        vasp_config_dir = os.path.join(vasp_dir, f'config_{poscar_index}')
        if not os.path.exist(vasp_config_dir):
            os.makedirs(vasp_config_dir)
        # prepare POSCAR
        vasp_poscar_dst_path = os.path.join(vasp_config_dir, 'POSCAR')
        copyfile(poscar_file, vasp_poscar_dst_path)
        # prepare POTCAR


    ##############################################################
    # General iteration 
    # Do the first iteration
    # get initial poscar from vasp_sys_dir
    ##############################################################
    for i in range(1, numb_iter):
        iter_dir = f'iter_{i}'
        vasp_iter(data['vasp_sys_dir'],
                  data['vasp_command'],
                  data['vasp_np'],
                  data['vasp_potential'],
                  data['vasp_params'])





    print(len(data['init_data_sys']))
    print(len(data['init_batch_size']))
    print(len(data['sys_batch_size']))


def vasp_incar_generate(vasp_params: Dict, vasp_config_dir: str):
    """Generate INCAR for vasp

    :vasp_params: Dict: TODO
    :returns: TODO

    """
    vasp_incar_path = os.path.join(vasp_config_dir, 'INCAR')
    with open(vasp_incar_path, 'w') as vasp_incar_file:
        vasp_incar_str = \
            f'''SYSTEM =  H2 # electronic degrees
            PREC  = Accurate                 # chose Low only after tests
            ENCUT = {vasp_params['encut']}
            EDIFF = {vasp_params['ediff']} # do not use default (too large)
            ALGO = Fast
            LREAL = A                      # real space projection
            ISMEAR = -1 ; SIGMA = 0.130    # Fermi smearing: 1500 K 0.086 10-3
            ISYM = 0                       # no symmetry 
            NCORE = {vasp_params['ncore']}   # default choose 4
            NPAR = {vasp_params['npar']}     # default choose 7
            KPAR = {vasp_params['kpar']}     # default choose 4

            # Add kspacing and kgamma tags
            KSPACING	= {vasp_params['kspacing']}
            KGAMMA	=	.TRUE.

            # (do little writing to save disc space)
            IBRION = -1 ; LCHARG = .FALSE. ; LWAVE = .FALSE.
        '''
        vasp_incar_file.write(vasp_incar_str)


def vasp_potcar_generate(vasp_params: Dict,
                         vasp_config_dir: str,
                         vasp_dir: str):
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
        with open(vasp_incar_path, 'w') as vasp_potcar_file:
            for fname in vasp_params['vasp_potential']:
                with open(fname) as infile:
                    for line in infile:
                        vasp_potcar_file.write(line)
    # if the potcar is generated
    copyfile(vasp_backup_potcar_path, vasp_potcar_path)


def vasp_iter(vasp_sys_dir: str,
              vasp_command: str,
              vasp_np: int,
              vasp_potential: List,
              vasp_params: Dict
              ):
    """TODO: Docstring for vasp_iter.

    :sys_configs: List: TODO
    :: TODO
    :returns: TODO

    """
    pass

def deepmd_iter(deepmd_sys_dir: str,
                init_batch_size: List,
                sys_batch_size: List,
                default_training_param: Dict):
    """TODO: Docstring for deepmd_iter.

    :init_data_sys_dir: TODO
    :init_data_sys: TODO
    :returns: TODO

    """


def lmp_iter(lmp_command: str,
             model_devi_np: int,
             model_devi_trust: float,
             model_devi_f_trust_lo: float,
             model_devi_f_trust_hi: float,
             model_devi_e_trust_lo: float,
             model_devi_e_trust_hi: float
             ):
    """TODO: Docstring for lmp_iter.

    :lmp_command: str
    :model_devi_np: int: TODO
    :model_devi_trust: float: TODO
    :model_devi_f_trust_lo: float
    :model_devi_f_trust_hi: float
    :model_devi_e_trust_lo: float
    :model_devi_e_trust_hi: float: TODO
    :returns: TODO

    """
    pass


if __name__=="__main__":

