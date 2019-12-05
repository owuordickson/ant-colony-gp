# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@license: "MIT"
@version: "1.0"
@email: "owuordickson@gmail.com"
@created: "5 November 2019"

"""


import os
import multiprocessing as mp


class InitParallel:

    @staticmethod
    def get_num_cores():
        num_cores = InitParallel.get_slurm_cores()
        if not num_cores:
            num_cores = mp.cpu_count()
        return num_cores

    @staticmethod
    def get_slurm_cores():
        try:
            cores = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            return cores
        except ValueError:
            str_cores = str(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            temp = str_cores.split('(', 1)
            cpus = int(temp[0])
            str_nodes = temp[1]
            temp = str_nodes.split('x', 1)
            str_temp = str(temp[1]).split(')', 1)
            nodes = int(str_temp[0])
            cores = cpus * nodes
            return cores
        except KeyError:
            return False
