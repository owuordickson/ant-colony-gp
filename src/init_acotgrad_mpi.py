# -*- coding: utf-8 -*-
"""
@author: "Dickson Owuor"
@credits: "Thomas Runkler and Anne Laurent,"
@license: "MIT"
@version: "3.2"
@email: "owuordickson@gmail.com"
@created: "15 June 2020"

Optimized using MPI and Parallel HDF5

Usage:
    $mpirun -n 4 python init_acotgrad_mpi.py -f ../data/DATASET.csv -c 0 -s 0.5 -r 0.5

Description:
    f -> file path (CSV)
    c -> reference column
    s -> minimum support
    r -> representativity

Credits:
    http://docs.h5py.org/en/latest/mpi.html
    https://www.hdfgroup.org/downloads/hdf5/
    https://www.kth.se/blogs/pdc/2019/08/parallel-programming-in-python-mpi4py-part-1/
    http://what-when-how.com/Tutorial/topic-97chepqdi/Python-and-HDF5-137.html

"""


import sys
import os
from optparse import OptionParser
import numpy as np
from pathlib import Path
import h5py
from mpi4py import MPI
from algorithms.ant_colony.mpi.aco_tgrad_mpi import Dataset_t, T_GradACO, GradACOt


def write_file(data, path):
    with open(path, 'w') as f:
        f.write(data)
        f.close()


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
                             # default='../data/rain_temp2013-2015.csv',
                             default='../data/Directio.csv',
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

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    h5_file = str(Path(file_path).stem) + str('_mpi.h5')
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
            d_set = Dataset_t(min_sup=min_sup, eq=allow_eq, h5f=h5f)
        else:
            # create new data set from csv file
            d_set = Dataset_t(file_path=file_path, min_sup=min_sup, eq=allow_eq)
        t_aco = T_GradACO(d_set, ref_col, min_rep)
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

    # store in h5 file
    if not exists:
        print("writing")
        d_set = t_aco.d_set
        grp = h5f.require_group('dataset')
        grp.create_dataset('title', data=d_set.title)
        data = np.array(d_set.data.copy()).astype('S')
        grp.create_dataset('data', data=data)
        grp.create_dataset('time_cols', data=d_set.time_cols)
        grp.create_dataset('attr_cols', data=d_set.attr_cols)
        grp.create_dataset('size', data=np.array([d_set.column_size, d_set.size]))
        for st in range(t_aco.max_step):
            st += 1
            step_name = 'step_' + str(st)
            ds = step_name + '/valid_bins'
            n = (d_set.size - st)
            dt = np.dtype("({0}, {0})bool".format(n, n))
            grp.create_dataset(ds, (d_set.column_size,), dtype=dt, chunks=True)

            ds = step_name + '/time_diffs'
            grp.create_dataset(ds, (n, 2), dtype='f', chunks=True)
            # grp.create_dataset(ds, (n, 2), dtype='f4, i4')

            ds = step_name + '/p_matrix'
            grp.create_dataset(ds, (d_set.column_size, 3), dtype='f4', chunks=True)

            ds = step_name + '/attr_size'
            grp.create_dataset(ds, (1,), dtype='i4')
        data = None

    # fetch TGPs
    lst_tgp = list()
    for step in steps:
        s = step
        step += 1  # because for-loop is not inclusive from range: 0 - max_step
        d_set = t_aco.d_set
        d_set.step_name = 'step_' + str(step)

        if exists:
            # read from h5 file
            time_diffs = h5f['dataset/' + d_set.step_name + '/time_diffs'][:]
            # p_matrix = h5f['dataset/' + d_set.step_name + '/p_matrix'][:]
            d_set.attr_size = h5f['dataset/' + d_set.step_name + '/attr_size'][0]

            temp_bins = h5f['dataset/' + d_set.step_name + '/valid_bins'][:]
            invalid_bins = []
            for col in range(len(temp_bins)):
                total = np.sum(temp_bins[col])
                if total <= 0:
                    incr = np.array((col, '+'), dtype='i, S1')
                    decr = np.array((col, '-'), dtype='i, S1')
                    invalid_bins.append(incr)
                    invalid_bins.append(decr)
            d_set.invalid_bins = np.array(invalid_bins)
        else:
            # write to h5 file
            # 1. Transform data for each step
            attr_data, time_diffs = t_aco.transform_data(step)

            # 2. fetch all valid/invalid bins and store in d_set
            d_set.attr_size = len(attr_data[d_set.attr_cols[0]])
            n = d_set.attr_size
            # valid_bins = list()  # to be removed
            invalid_bins = list()  # to be removed
            for col in d_set.attr_cols:
                col_data = np.array(attr_data[col], dtype=float)
                is_valid, res = t_aco.construct_bins(col, n, col_data)
                if is_valid:
                    # store in valid bins
                    ds = d_set.step_name + '/valid_bins'
                    grp[ds][col] = res
                else:
                    # store in invalid bins
                    for temp in res:
                        invalid_bins.append(temp)
            d_set.invalid_bins = np.array(invalid_bins)

            ds = d_set.step_name + '/time_diffs'
            grp[ds][...] = time_diffs

            ds = d_set.step_name + '/attr_size'
            grp[ds][0] = d_set.attr_size

        # 3. Execute aco-graank
        ac = GradACOt(d_set, time_diffs, h5f)
        tgps = ac.run_ant_colony()  # needs to read h5 file
        if len(tgps) > 0:
            lst_tgp.append(tgps)

        ds = 'dataset/' + d_set.step_name + '/p_matrix'
        h5f[ds][...] = ac.p_matrix

    # gather all patterns from all processes
    lst_tgp = comm.gather(lst_tgp, root=0)
    end = MPI.Wtime()

    # display results and save to file
    if rank == 0:
        d_set = t_aco.d_set
        wr_line = "Algorithm: ACO-TGRAANK (3.2) \n"
        wr_line += "   - MPI4Py & H5Py implementation \n"
        wr_line += "No. of (dataset) attributes: " + str(d_set.column_size) + '\n'
        wr_line += "No. of (dataset) tuples: " + str(d_set.size) + '\n'
        wr_line += "Minimum support: " + str(min_sup) + '\n'
        wr_line += "Minimum representativity: " + str(min_rep) + '\n'
        wr_line += "Multi-core execution: True" + '\n'
        wr_line += "Number of cores: " + str(nprocs) + '\n'
        wr_line += "Number of tasks: " + str(t_aco.max_step) + '\n'

        for txt in d_set.title:
            col = int(txt[0])
            if col == ref_col:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '**' + '\n')
            else:
                wr_line += (str(txt[0]) + '. ' + str(txt[1].decode()) + '\n')

        wr_line += str("\nFile: " + file_path + '\n')
        wr_line += str("\nPattern : Support" + '\n')

        # lst_tgp is gathered from all processes
        count = 0
        for obj in lst_tgp:
            for pats in obj:
                if pats:
                    for tgp in pats:
                        count += 1
                        wr_line += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
                                    ' | ' + str(tgp.time_lag.to_string()) + '\n')
        wr_line += "\n\n Number of patterns: " + str(count) + '\n'

        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += str(wr_line)
        f_name = str('res_aco_t' + str(end).replace('.', '', 1) + '.txt')
        write_file(wr_text, f_name)
        print(wr_text)

    h5f.close()
