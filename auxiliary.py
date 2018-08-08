import os
import shutil
import sys
from typing import List, Set


def unique(seq: List) -> List:
    # return unique list without changing the order
    # from http://stackoverflow.com/questions/480214
    seen: Set = set()
    seen_add = seen.add
    return [x for x in seq if x not in seen and not seen_add(x)]


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


def list_concat(list1: List, list2: List):
    return [(a_ + b_) for a_, b_ in zip(list1, list2)]
