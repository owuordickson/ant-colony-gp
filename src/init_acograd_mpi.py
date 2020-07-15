# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler, Edmond Menya, and Anne Laurent,"
@license: "MIT"
@version: "3.2"
@email: "owuordickson@gmail.com"
@created: "20 June 2020"

Optimized using MPI and Parallel HDF5

Usage:
    $mpirun -n 4 python init_acograd_mpi.py -f ../data/DATASET.csv -s 0.5

Description:
    f -> file path (CSV)
    s -> minimum support

"""

import sys
from optparse import OptionParser
import numpy as np
from mpi4py import MPI
import h5py
from pathlib import Path
import os
import gc
from algorithms.ant_colony.mpi.aco_grad_mpi import Dataset_mpi, GradACO_mpi
from algorithms.common.profile_cpu import Profile
from algorithms.ant_colony.hdf5.aco_grad_h5 import GradACO_h5
# from src.algorithms.ant_colony.cython.cyt_aco_grad import GradACO


def write_file(data, path):
    with open(path, 'w') as f:
        f.write(data)
        f.close()

# ------------------------- main method ---------------------------------------------


if __name__ == "__main__":
    if not sys.argv:
        pType = sys.argv[1]
        filePath = sys.argv[2]
        # refCol = sys.argv[3]
        minSup = sys.argv[3]
        allowEq = sys.argv[4]
    else:
        optparser = OptionParser()
        optparser.add_option('-f', '--inputFile',
                             dest='file',
                             help='path to file containing csv',
                             # default=None,
                             #default='../data/DATASET.csv',
                             #default='../data/DATASET3.csv',
                             #default='../data/Omnidir.csv',
                             default='../data/FluTopicData-testsansdate-blank.csv',
                             #default='data/FluTopicData-testsansdate-blank.csv',
                             #default='../data/FARSmiss.csv',
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
        optparser.add_option('-c', '--cores',
                             dest='numCores',
                             help='number of cores',
                             default=1,
                             type='int')
        (options, args) = optparser.parse_args()

        if options.file is None:
            print("Usage: $python init_acograd.py -f filename.csv ")
            sys.exit('System will exit')
        else:
            filePath = options.file
        minSup = options.minSup
        allowEq = options.allowEq
        numCores = options.numCores

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    h5_file = str(Path(filePath).stem) + str('_mpi.h5')
    exists = os.path.exists(h5_file)
    if not exists:  # for parallel
        h5f = h5py.File(h5_file, 'a', driver='mpio', comm=comm)
    else:
        h5f = h5py.File(h5_file, 'r+', driver='mpio', comm=comm)

    start = MPI.Wtime()
    # fetch data from csv and distribute work among processes
    if rank == 0:
        # master process
        print("master process " + str(rank) + " started ...")

        if exists:
            # read data set from h5 file
            print("reading")
            d_set = Dataset_mpi(min_sup=minSup, h5f=h5f)
            attr_data = None
        else:
            # create new data set from csv file
            d_set = Dataset_mpi(file_path=filePath, min_sup=minSup, eq=allowEq)
            attr_data = d_set.data.copy().T
            d_set.attr_size = len(attr_data[d_set.attr_cols[0]])
        a_count = d_set.attr_cols

        # determine the size of each sub-task
        ave, res = divmod(a_count.size, nprocs)
        counts = [ave + 1 if p < res else ave for p in range(nprocs)]

        # determine the starting and ending indices of each sub-task
        starts = [sum(counts[:p]) for p in range(nprocs)]
        ends = [sum(counts[:p + 1]) for p in range(nprocs)]

        # converts data into a list of arrays
        a_count = [a_count[starts[p]:ends[p]] for p in range(nprocs)]
        # swapping steps so that Process 0 has the smallest data
        a_count[0], a_count[-1] = a_count[-1], a_count[0]
    else:
        # worker process
        print("worker process " + str(rank) + " started ...")
        a_count = None
        d_set = None
        attr_data = None
    a_count = comm.scatter(a_count, root=0)
    d_set = comm.bcast(d_set, root=0)
    attr_data = comm.bcast(attr_data, root=0)

    # store in h5 file
    if not exists:
        print("writing")
        grp = h5f.require_group('dataset')
        grp.create_dataset('title', data=d_set.title)
        # data = np.array(d_set.data.copy()).astype('S')
        # grp.create_dataset('data', data=data)
        grp.create_dataset('time_cols', data=d_set.time_cols)
        grp.create_dataset('attr_cols', data=d_set.attr_cols)
        grp.create_dataset('size', data=np.array([d_set.column_size, d_set.size, d_set.attr_size]))

        n = d_set.attr_size
        d_set.step_name = 'step_' + str(int(d_set.size - d_set.attr_size))
        ds = d_set.step_name + '/p_matrix'
        grp.create_dataset(ds, (d_set.column_size, 3), dtype='f4')
        invalid_bins = list()
        for col in d_set.attr_cols:
            col_data = np.array(attr_data[col], dtype=float)
            incr = np.array((col, '+'), dtype='i, S1')
            decr = np.array((col, '-'), dtype='i, S1')
            temp_pos = Dataset_mpi.bin_rank(col_data, equal=d_set.equal)
            supp = float(np.sum(temp_pos)) / float(n * (n - 1.0) / 2.0)

            if supp < d_set.thd_supp:
                invalid_bins.append(incr)
                invalid_bins.append(decr)
            else:
                ds = d_set.step_name + '/valid_bins/' + str(col) + '_pos'
                grp.create_dataset(ds, data=temp_pos)
                ds = d_set.step_name + '/valid_bins/' + str(col) + '_neg'
                grp.create_dataset(ds, data=temp_pos.T)
        d_set.invalid_bins = np.array(invalid_bins)
        ds = d_set.step_name + '/invalid_bins'
        grp.create_dataset(ds, data=d_set.invalid_bins)
        attr_data = None
        d_set.data = None
        gc.collect()

    # fetch GPs
    ac = GradACO_mpi(d_set, h5f)
    lst_gp = ac.run_ant_colony()
    ds = 'dataset/' + d_set.step_name + '/p_matrix'
    h5f[ds][...] = ac.p_matrix

    # gather all patterns from all processes
    lst_tgp = comm.gather(lst_gp, root=0)
    end = MPI.Wtime()
    # display results and save to file
    if rank == 0:
        wr_line = "Algorithm: ACO-GRAANK (3.2)\n"
        wr_line += "   - MPI & H5Py parallel implementation \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(minSup) + '\n'
        wr_line += "Number of processes: " + str(nprocs) + '\n'
        wr_line += "Number of patterns: " + str(len(lst_gp)) + '\n\n'

        for txt in d_set.title:
            try:
                wr_line += (str(txt.key) + '. ' + str(txt.value.decode()) + '\n')
            except AttributeError:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + filePath + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        for gp in lst_gp:
            wr_line += (str(gp.to_string()) + ' : ' + str(gp.support) + '\n')

        wr_line += "\nPheromone Matrix\n"
        wr_line += str(ac.p_matrix)

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += str(wr_line)
        f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
        write_file(wr_text, f_name)
        print(wr_text)
    h5f.close()
