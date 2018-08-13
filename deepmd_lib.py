#!/usr/bin/env python3
# #-*-coding:utf-8 -*-
#########################################################################
# File Name   :  param.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn
# created at  :  2018-07-20 03:34
# purpose     :
#########################################################################
import json
import os
import random
import shutil
import subprocess
import sys
from multiprocessing import Pool
from typing import Dict, List

import auxiliary
import cessp2force_lin
import convert2raw


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
        with auxiliary.cd(deepmd_set_absolute):
            # Generate test_configs
            total_configs = cessp2force_lin.param_interface(
                vasp_set_list_absolute[set_index], True)
            # Generate raw dir
            test_configs_absolute = os.path.abspath('test.configs')
            convert2raw.param_interface(test_configs_absolute)
            print('generate_raw')
            if 'max_set_number' not in deepmd_data:
                max_set_number = 10
            else:
                max_set_number = deepmd_data['set_number']
            # Generate set
            # TODO
            # DIVIDE 10 is a magic number, but I don't know how to choose

            numb_test = deepmd_data['training_params']['numb_test']
            set_size = -1
            for train_set_number in range(1, max_set_number):
                configs_in_set = (total_configs - numb_test) // train_set_number
                if configs_in_set > numb_test:
                    continue
                else:
                    set_size = (total_configs - numb_test) // (train_set_number - 1)

            # Check whether set_size is updated
            if set_size == -1:
                print("making set is unsuccessful", file=sys.stderr)
                tb = sys.exc_info()
                raise Exception("foo occurred").with_traceback(tb)

            print(f'set size is {set_size}')

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
    with auxiliary.cd(deepmd_dir):
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

    # specify json file path
    deepmd_json_path = os.path.join(deepmd_graph_dir, 'deepmd.json')

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
    sets_system_list: List[str] = list()
    # FIXED
    # train from the sets in previous and current iter
    with auxiliary.cd(deepmd_graph_dir):
        for exist_sets_iter_index in range(iter_index + 1):
            deepmd_data_root_path = os.path.join('..', '..', '..', f'iter_{exist_sets_iter_index}', 'deepmd', 'data')
            sets_system_list += [
                os.path.join(deepmd_data_root_path, deepmd_set_dir)
                for deepmd_set_dir in os.listdir(deepmd_data_root_path)
                if deepmd_set_dir.startswith('deepmd_set')
            ]
    deepmd_data['training_params']['systems'] = sets_system_list

    # Create if not have graph dir
    if not os.path.exists(deepmd_graph_dir):
        os.makedirs(deepmd_graph_dir)

    with open(deepmd_json_path, 'w') as deepmd_json:
        json.dump(deepmd_data['training_params'], deepmd_json, indent=2)


def deepmd_cp_ckpt(iter_index: int, graph_index: int):
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
                shutil.copy2(os.path.join(iter_previous_graph_dir, model_ckpt), iter_graph_dir)


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
    with auxiliary.cd(deepmd_graph_dir):
        deepmd_json_path = os.path.join('.', 'deepmd.json')
        # Not set OMP number, use the default

        print("enter_traina_dir", file=sys.stderr)
        print("need_continue_run", need_continue, file=sys.stderr)
        # Check if restart
        if not need_continue:
            # Now don't need --init-model parameter in dp_train
            subprocess.run([dp_train_path, deepmd_json_path])
            print("new model", file=sys.stderr)
        else:
            subprocess.run(
                [dp_train_path, deepmd_json_path, '--restart', 'model.ckpt'])
            print("restart-model", file=sys.stderr)
        # Start freeze model
        print(f'Now start freezing the graph in the {deepmd_graph_dir}\n', file=sys.stderr)
        subprocess.run([dp_frz_path])
        print(f'Freezing end\n', file=sys.stderr)


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
        lcurve_path = os.path.join(deepmd_graph_dir, f'lcurve.out')

        # This is the HACK
        # TODO
        # i don't provide two initial and continue commands to distinguish the initial and restart command
        # Just delete the directory and redo
        if need_continue and os.path.exists(lcurve_path):
            process.apply_async(
                deepmd_single_process_continue_iter,
                args=(graph_index, deepmd_graph_dir, deepmd_data, iter_index,
                      need_continue and os.path.exists(lcurve_path)))
        else:
            process.apply_async(
                deepmd_single_process_initial_iter,
                args=(graph_index, deepmd_graph_dir, deepmd_data, iter_index,
                      need_continue and os.path.exists(lcurve_path)))

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
    deepmd_cp_ckpt(iter_index, graph_index)
    # update deepmd check point
    deepmd_update_checkpoint(iter_index)
    # Training and freezing the model
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
    # Training and freezing the model
    deepmd_run(iter_index, deepmd_graph_dir, deepmd_data, need_continue)


def deepmd_update_checkpoint(iter_index: int):
    """  Deepmd update the checkpoint
    :type iter_index: int

    """
    # Now update will preserve the previous steps information
    with open('generator_checkpoint.json', 'r') as generate_ckpt:
        ckpt = json.load(generate_ckpt)
        ckpt['status'] = 'deepmd'
        ckpt['config_index'] = 0  # multiprocessing don't need graph_index
        ckpt['set_index'] = 0
        ckpt['iter_index'] = iter_index
        ckpt['dump_to_poscar'] = False

    # Dump to the same file and erase the former
    with open('generator_checkpoint.json', 'w') as generate_ckpt:
        json.dump(ckpt, generate_ckpt, indent=2)
