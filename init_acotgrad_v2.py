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
from optparse import OptionParser
from mpi4py import MPI
from src.algorithms.ant_colony.aco_tgrad_v2 import T_GradACO


def write_file(data, path):
    with open(path, 'w') as f:
        f.write(data)
        f.close()


if __name__ == "__main__":
    if not sys.argv:
        # pType = sys.argv[1]
        file_path = sys.argv[1]
        ref_col = sys.argv[2]
        min_sup = sys.argv[3]
        min_rep = sys.argv[4]
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
        optparser.add_option('-c', '--refColumn',
                             dest='refCol',
                             help='reference column',
                             default=1,
                             type='int')
        optparser.add_option('-s', '--minSupport',
                             dest='minSup',
                             help='minimum support value',
                             default=0.5,
                             type='float')
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
        ref_col = options.refCol
        min_sup = options.minSup
        min_rep = options.minRep

    import time
    start = time.time()
    # res_text = init_algorithm(file_path, ref_col, min_sup, min_rep, cores)
    # end = time.time()
    # wr_text = ("Run-time: " + str(end - start) + " seconds\n")
    # wr_text += str(res_text)
    # f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
    # write_file(wr_text, f_name)
    # print(wr_text)

    import numpy as np
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    if rank == 0:
        # master process
        print("master process " + str(rank))
        # data = np.arange(nprocs)
        eq = False
        t_aco = T_GradACO(file_path, eq, ref_col, min_sup, min_rep)
    else:
        # worker process
        print("worker process " + str(rank))
        t_aco = None

    t_aco = comm.bcast(t_aco, root=0)
    # t_aco.d_set.init_h5_groups(comm=comm)
    # lst_tgp = t_aco.fetch_patterns(rank)

    if rank == 0:
        t_aco.d_set.init_h5_groups()
        lst_tgp = t_aco.fetch_patterns(rank+1)

        for i in range(1, nprocs):
            req = comm.irecv(source=i, tag=rank)
            data = req.wait()
            print(data)

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

        for obj in lst_tgp:
            wr_line += (str(obj.to_string()) + ' : ' + str(obj.support) +
                        ' | ' + str(obj.time_lag.to_string()) + '\n')
            # if obj:
            # for tgp in obj:
            #    wr_line += (str(tgp.to_string()) + ' : ' + str(tgp.support) +
            #    ' | ' + str(tgp.time_lag.to_string()) + '\n')

        end = time.time()
        wr_text = ("Run-time: " + str(end - start) + " seconds\n")
        wr_text += str(wr_line)
        f_name = str('res_aco' + str(end).replace('.', '', 1) + '.txt')
        # write_file(wr_text, f_name)
        print(wr_text)
    else:
        lst_tgp = t_aco.max_step * rank
        # lst_tgp = tgp.fetch_patterns(rank)
        req = comm.isend(lst_tgp, dest=0, tag=0)
        req.wait()

