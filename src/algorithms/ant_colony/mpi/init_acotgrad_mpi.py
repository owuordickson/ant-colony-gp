# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Joseph Orero and Anne Laurent,"
@license: "MIT"
@version: "2.0"
@email: "owuordickson@gmail.com"
@created: "19 November 2019"

Usage:
    $python3 init_acotgrad.py -f ../data/DATASET.csv -c 0 -s 0.5 -r 0.5 -p 1

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity

"""


import sys
import os
from optparse import OptionParser
from src.algorithms.ant_colony.mpi.aco_tgrad_mpi import Dataset_t, T_GradACO


def write_file(data, path):
    with open(path, 'w') as f:
        f.write(data)
        f.close()


def fetch_patterns(self, step):
    step += 1  # because for-loop is not inclusive from range: 0 - max_step
    # 1. Transform data
    d_set = self.d_set
    attr_data, time_diffs = self.transform_data(step)

    # 2. Execute aco-graank for each transformation
    attr_size = len(attr_data[self.attr_cols[0]])
    for col in d_set.attr_cols:
        col_data = np.array(attr_data[col], dtype=float)
        is_valid, temp = d_set.construct_bins(self, col, attr_size, col_data)

    ac = GradACOt(d_set, time_diffs)
    list_gp = ac.run_ant_colony()
    # print("\nPheromone Matrix")
    # print(ac.p_matrix)
    if len(list_gp) > 0:
        return list_gp
    return False


if __name__ == "__main__":
    if not sys.argv:
        # pType = sys.argv[1]
        file_path = sys.argv[1]
        min_sup = sys.argv[2]
        allow_eq = sys.argv[3]
        ref_col = sys.argv[4]
        min_rep = sys.argv[5]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             #default='../data/DATASET2.csv',
                             #default='../data/rain_temp2013-2015.csv',
                             default='data/rain_temp2013-2015.csv',
                             #default='../data/Directio.csv',
                             type='string')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
        optparser.add_option('-e', '--allowEqual',
                             dest='allowEq',
                             help='allow equal',
                             default=None,
                             type='int')
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=1,
                             type='int')
        optparser.add_option('-r', '--minRepresentativity',
                             dest='minRep',
                             help='minimum representativity',
                             default=0.5,
                             type='float')
        (options, args) = optparser.parse_args()
        inFile = None
        if options.file is None:
            print('No data-set filename specified, system with exit')
            print("Usage: $python3 init_acotgrad.py -f filename.csv -c refColumn -s minSup  -r minRep")
            sys.exit('System will exit')
        else:
            inFile = options.file
        file_path = inFile
        min_sup = options.minSup
        allow_eq = options.allowEq
        ref_col = options.refCol
        min_rep = options.minRep

    import time
    import numpy as np
    from pathlib import Path
    import h5py
    from mpi4py import MPI

    start = time.time()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    h5_file = str(Path(file_path).stem) + str('.h5')
    # if not os.path.exists(h5_file):  # for parallel
    #    h5f = h5py.File(h5_file, 'w', driver='mpio', comm=comm)
    # else:
    #    h5f = h5py.File(h5_file, 'r', driver='mpio', comm=comm)

    if rank == 0:
        # master process
        print("master process " + str(rank) + " started ...")

        if os.path.exists(h5_file):  # to be removed
            # read data set from h5 file
            h5f = h5py.File(h5_file, 'r')  # to be removed
            h5f.close()  # to be removed
        else:
            # create new data set from csv file
            d_set = None
        t_aco = T_GradACO(d_set, ref_col, min_sup, min_rep)
        steps = np.arange(t_aco.max_step)

        # determine the size of each sub-task
        ave, res = divmod(steps.size, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]

        # determine the starting and ending indices of each sub-task
        starts = [sum(counts[:p]) for p in range(nprocs)]
        ends = [sum(counts[:p + 1]) for p in range(nprocs)]

        # converts data into a list of arrays
        steps = [steps[starts[p]:ends[p]] for p in range(nprocs)]
        # swapping steps so that Process 0 has the smallest data
        steps[0], steps[-1] = steps[-1], steps[0]
    else:
        # worker process
        print("worker process " + str(rank) + " started ...")
        # d_set = None
        t_aco = None
        steps = None
    steps = comm.scatter(steps, root=0)
    t_aco = comm.bcast(t_aco, root=0)

    # fetch TGPs
    lst_tgp = list()
    if rank == 0:  # if-else to be removed
        h5f = h5py.File(h5_file, 'r+')  # to be removed
        for s in steps:
            tgp = t_aco.fetch_patterns(s, h5f)
            lst_tgp.append(tgp)
        h5f.close()  # to be removed
    else:
        for s in steps:
            # tgp = t_aco.fetch_patterns(s, h5f)
            tgp = False
            lst_tgp.append(tgp)


    lst_tgp = comm.gather(lst_tgp, root=0)
    if rank == 0:
        d_set = t_aco.d_set
        wr_line = "Algorithm: ACO-TGRAANK (3.0) \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(min_sup) + '\n'
        wr_line += "Minimum representativity: " + str(min_rep) + '\n'
        wr_line += "Multi-core execution: True" + '\n'
        wr_line += "Number of cores: " + str(nprocs) + '\n'
        wr_line += "Number of tasks: " + str(t_aco.max_step) + '\n\n'

        for txt in d_set.title:
            col = int(txt[0])
            if col == ref_col:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '**' + '\n')
            else:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + file_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        # lst_tgp is gathered from all processes
        for obj in lst_tgp:
            for pats in obj:
                if pats:
                    for tgp in pats:
                        wr_line += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                                    ' | ' + str(tgp.time_lag.to_string()) + '\n')

        end = time.time()
        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += str(wr_line)
        f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
        # write_file(wr_text, f_name)
        print(wr_text)
