#!/usr/bin/env python
##-*-coding:utf-8 -*-
#########################################################################
# File Name   :  hello.py
# author      :   Tiangang Zhou
# e-Mail      :  tg_zhou@pku.edu.cn 
# created at  :  2018-07-22 23:21
# purpose     :  
#########################################################################


import os

from ase import io


def lmp_parse_dump2poscar(lmp_config_dir: str):
    """Change the dump file to the xyz file

    :dump_file_path: str: TODO
    :lmp_data: Dict: TODO
    :returns: TODO

    """
    dump_file_path = os.path.join(lmp_config_dir, 'dump.trajectory')
    dump_xyz_path = os.path.join(lmp_config_dir, 'dump.xyz')
    with open(dump_file_path, 'r') as dump_file:
        with open(dump_xyz_path, 'w') as dump_xyz_file:
            for line in dump_file:
                line_field = line.strip().split()
                if len(line_field) == 4:
                    element_type: str = "H"
                    line_field[0] = element_type
                line = ' '.join(line_field) + '\n'
                dump_xyz_file.write(line)

    # Read xyz file

    configs = io.read('dump.xyz', index=':')
    for config in configs:
        print(config.positions)
        print(len(config.positions))
    print(len(configs))



if __name__=="__main__":
    lmp_parse_dump2poscar('.')
